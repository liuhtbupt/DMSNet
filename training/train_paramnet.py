import os
import sys
import csv
import math
import random
import itertools
import copy
import zipfile
from pathlib import Path

import h5py
import numpy as np
from numpy.lib import format as npformat

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

try:
    import swanlab
except ImportError:
    swanlab = None

RELEASE_ROOT = Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

from models.paramnet import VelocityEnhancedParamNet, count_parameters


# ======================================================================
# 1. Dataset
# ======================================================================
class MultiTargetParamDataset(Dataset):
    def __init__(
        self,
        mat_path,
        high_shape=(64, 112, 32),
        low_shape=(64, 14, 4),
    ):
        super().__init__()

        self.mat_path = mat_path
        self.high_shape = high_shape
        self.low_shape = low_shape
        self._file = None

        with h5py.File(self.mat_path, "r") as f:
            self.K_list_all = np.array(f["K_list"], dtype=np.int64).squeeze()
            self.num_samples = int(self.K_list_all.shape[0])

            self.Y_phys_all = self._fix_label_layout(
                np.array(f["Y_phys"], dtype=np.float32),
                field_dim=4,
                name="Y_phys",
            )
            self.Kmax = int(self.Y_phys_all.shape[1])

            self.P_h_all = np.array(f["P_h_clean"], dtype=np.float32).squeeze()
            if "P_l_clean" in f:
                self.P_l_all = np.array(f["P_l_clean"], dtype=np.float32).squeeze()
            else:
                self.P_l_all = np.ones(self.num_samples, dtype=np.float32)

            self.cfg = self._read_cfg(f)

            if "Y_bin0_debug" in f:
                self.Y_bin_all = self._fix_label_layout(
                    np.array(f["Y_bin0_debug"], dtype=np.int64),
                    field_dim=3,
                    name="Y_bin0_debug",
                )
            else:
                self.Y_bin_all = self._make_bins_from_phys(self.Y_phys_all, self.cfg)

            if "Y_res_debug" in f:
                self.Y_delta_phys_all = self._fix_label_layout(
                    np.array(f["Y_res_debug"], dtype=np.float32),
                    field_dim=3,
                    name="Y_res_debug",
                )
            else:
                self.Y_delta_phys_all = self._make_delta_phys_from_phys(
                    self.Y_phys_all,
                    self.Y_bin_all,
                    self.cfg,
                )

            widths = np.array(
                [
                    (self.cfg["r_max"] - self.cfg["r_min"]) / self.cfg["Nbin_r"],
                    (self.cfg["v_max"] - self.cfg["v_min"]) / self.cfg["Nbin_v"],
                    (self.cfg["theta_max"] - self.cfg["theta_min"]) / self.cfg["Nbin_theta"],
                ],
                dtype=np.float32,
            )
            self.Y_delta_norm_all = self.Y_delta_phys_all / widths.reshape(1, 1, 3)
            self.Y_delta_norm_all = np.clip(self.Y_delta_norm_all, -0.5, 0.5)

            exist_all = self.K_list_all > 0
            valid_h = exist_all & np.isfinite(self.P_h_all) & (self.P_h_all > 0)
            self.P_h_ref = float(np.median(self.P_h_all[valid_h])) if np.any(valid_h) else 1.0
            self.P_h_ref = max(self.P_h_ref, 1e-30)

            print("\n========== Param Dataset Info ==========")
            print("file:", mat_path)
            print("num_samples:", self.num_samples)
            print("Kmax:", self.Kmax)
            print("K distribution:", {int(k): int(np.sum(self.K_list_all == k)) for k in range(self.Kmax + 1)})
            print("Y_phys:", self.Y_phys_all.shape)
            print("Y_bin:", self.Y_bin_all.shape)
            print("X_h_real shape:", f["X_h_real"].shape)
            print("X_l_real shape:", f["X_l_real"].shape)
            print("cfg:", self.cfg)
            if "input_scale" in f.attrs:
                print(f"HDF5 input_scale = {float(f.attrs['input_scale']):.4e}")
            if "raw_P_h_ref" in f.attrs:
                print(f"HDF5 raw_P_h_ref = {float(f.attrs['raw_P_h_ref']):.4e}")
            print(f"P_h_ref = {self.P_h_ref:.4e}")
            print("========================================\n")

    def _read_cfg(self, f):
        defaults = {
            "r_min": 50.0,
            "r_max": 300.0,
            "v_min": -30.0,
            "v_max": 30.0,
            "theta_min": -math.pi / 3.0,
            "theta_max": math.pi / 3.0,
            "Nbin_r": 32,
            "Nbin_v": 32,
            "Nbin_theta": 32,
        }
        cfg = dict(defaults)
        if "cfg" not in f:
            return cfg

        for k in cfg:
            if k in f["cfg"]:
                value = np.array(f["cfg"][k]).squeeze()
                cfg[k] = int(value) if k.startswith("Nbin_") else float(value)
        return cfg

    def _fix_label_layout(self, arr, field_dim, name):
        if arr.ndim != 3:
            raise ValueError(f"{name} should be 3D, got {arr.shape}")

        N = self.num_samples
        if arr.shape[0] == N and arr.shape[2] == field_dim:
            return arr.copy()
        if arr.shape[0] == field_dim and arr.shape[2] == N:
            return np.transpose(arr, (2, 1, 0)).copy()
        if arr.shape[0] == N and arr.shape[1] == field_dim:
            return np.transpose(arr, (0, 2, 1)).copy()
        if arr.shape[1] == field_dim and arr.shape[2] == N:
            return np.transpose(arr, (2, 0, 1)).copy()

        raise ValueError(f"Unexpected {name} layout: {arr.shape}, field_dim={field_dim}, N={N}")

    @staticmethod
    def _make_bins_from_phys(y_phys, cfg):
        active = y_phys[:, :, 0] > 0.5
        y_bin = np.zeros(y_phys.shape[:2] + (3,), dtype=np.int64)
        spans = np.array(
            [
                cfg["r_max"] - cfg["r_min"],
                cfg["v_max"] - cfg["v_min"],
                cfg["theta_max"] - cfg["theta_min"],
            ],
            dtype=np.float32,
        )
        mins = np.array([cfg["r_min"], cfg["v_min"], cfg["theta_min"]], dtype=np.float32)
        nbins = np.array([cfg["Nbin_r"], cfg["Nbin_v"], cfg["Nbin_theta"]], dtype=np.float32)
        cont = (y_phys[:, :, 1:4] - mins.reshape(1, 1, 3)) / spans.reshape(1, 1, 3) * nbins.reshape(1, 1, 3)
        bins = np.floor(cont).astype(np.int64)
        bins[:, :, 0] = np.clip(bins[:, :, 0], 0, int(cfg["Nbin_r"]) - 1)
        bins[:, :, 1] = np.clip(bins[:, :, 1], 0, int(cfg["Nbin_v"]) - 1)
        bins[:, :, 2] = np.clip(bins[:, :, 2], 0, int(cfg["Nbin_theta"]) - 1)
        y_bin[active] = bins[active]
        return y_bin

    @staticmethod
    def _make_delta_phys_from_phys(y_phys, y_bin, cfg):
        mins = np.array([cfg["r_min"], cfg["v_min"], cfg["theta_min"]], dtype=np.float32)
        widths = np.array(
            [
                (cfg["r_max"] - cfg["r_min"]) / cfg["Nbin_r"],
                (cfg["v_max"] - cfg["v_min"]) / cfg["Nbin_v"],
                (cfg["theta_max"] - cfg["theta_min"]) / cfg["Nbin_theta"],
            ],
            dtype=np.float32,
        )
        centers = mins.reshape(1, 1, 3) + (y_bin.astype(np.float32) + 0.5) * widths.reshape(1, 1, 3)
        delta = y_phys[:, :, 1:4] - centers
        delta[y_phys[:, :, 0] <= 0.5] = 0.0
        return delta.astype(np.float32)

    def __len__(self):
        return self.num_samples

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def _ensure_open(self):
        if self._file is None:
            self._file = h5py.File(self.mat_path, "r")

    def _read_4d_sample(self, name, idx, expected_shape):
        d = self._file[name]
        s = d.shape
        Nc, Ms, Nr = expected_shape

        if len(s) == 4 and s[0] == Nc and s[1] == Ms and s[2] == Nr:
            return np.array(d[:, :, :, idx], dtype=np.float32)
        if len(s) == 4 and s[0] == self.num_samples and s[1] == Nr and s[2] == Ms and s[3] == Nc:
            arr = np.array(d[idx, :, :, :], dtype=np.float32)
            return np.transpose(arr, (2, 1, 0))
        if len(s) == 4 and s[0] == self.num_samples and s[1] == Nc and s[2] == Ms and s[3] == Nr:
            return np.array(d[idx, :, :, :], dtype=np.float32)

        raise ValueError(f"Unexpected {name} shape={s}, expected={expected_shape}")

    def __getitem__(self, idx):
        self._ensure_open()

        Xh_r = self._read_4d_sample("X_h_real", idx, self.high_shape)
        Xh_i = self._read_4d_sample("X_h_imag", idx, self.high_shape)
        Xl_r = self._read_4d_sample("X_l_real", idx, self.low_shape)
        Xl_i = self._read_4d_sample("X_l_imag", idx, self.low_shape)

        X_h = np.stack([Xh_r, Xh_i], axis=0).astype(np.float32)
        X_l = np.stack([Xl_r, Xl_i], axis=0).astype(np.float32)

        return {
            "X_h": torch.from_numpy(X_h),
            "X_l": torch.from_numpy(X_l),
            "K": torch.tensor(int(self.K_list_all[idx]), dtype=torch.long),
            "Y_phys": torch.from_numpy(self.Y_phys_all[idx].astype(np.float32)),
            "Y_bin": torch.from_numpy(self.Y_bin_all[idx].astype(np.int64)),
            "Y_delta_norm": torch.from_numpy(self.Y_delta_norm_all[idx].astype(np.float32)),
            "P_h": torch.tensor(max(float(self.P_h_all[idx]), 1e-30), dtype=torch.float32),
            "idx": torch.tensor(idx, dtype=torch.long),
        }


class FixedSNRGridDataset(Dataset):
    """Expose every clean scene once at every configured training SNR."""

    def __init__(self, base_dataset, snr_values):
        super().__init__()
        self.base_dataset = base_dataset
        self.snr_values = tuple(float(value) for value in snr_values)
        if len(self.base_dataset) <= 0:
            raise ValueError("base_dataset must not be empty")
        if not self.snr_values:
            raise ValueError("snr_values must not be empty")

    def __len__(self):
        return len(self.base_dataset) * len(self.snr_values)

    def __getitem__(self, index):
        base_size = len(self.base_dataset)
        snr_index, base_index = divmod(int(index), base_size)
        sample = dict(self.base_dataset[base_index])
        sample["snr_db"] = torch.tensor(
            self.snr_values[snr_index],
            dtype=torch.float32,
        )
        sample["base_idx"] = torch.tensor(base_index, dtype=torch.long)
        return sample


def _read_npy_header_from_zip(zf, member):
    with zf.open(member, "r") as f:
        version = npformat.read_magic(f)
        shape, fortran_order, dtype = npformat._read_array_header(f, version)
    return shape, fortran_order, dtype


def _read_exact(stream, nbytes):
    chunks = []
    remaining = int(nbytes)
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(f"Unexpected EOF while reading {nbytes} bytes from npz member")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _stream_complex_member_to_ri_npy(npz_path, member, out_path, chunk_rows=4):
    out_path = Path(out_path)
    if out_path.exists():
        return

    with zipfile.ZipFile(npz_path, "r") as zf, zf.open(member, "r") as f:
        version = npformat.read_magic(f)
        shape, fortran_order, dtype = npformat._read_array_header(f, version)
        if fortran_order:
            raise ValueError(f"{member} is Fortran ordered; expected C order.")
        if dtype != np.dtype(np.complex128):
            raise ValueError(f"{member} dtype={dtype}, expected complex128.")
        if len(shape) != 4:
            raise ValueError(f"{member} shape={shape}, expected [N,Nc,Ms,Nr].")

        n_samples = int(shape[0])
        sample_shape = tuple(int(v) for v in shape[1:])
        elems_per_row = int(np.prod(sample_shape))
        bytes_per_row = elems_per_row * dtype.itemsize

        print(f"Converting {member} -> {out_path}")
        print(f"  source shape={shape}, output shape={(n_samples, 2, *sample_shape)}, chunk_rows={chunk_rows}")

        with open(out_path, "wb") as out_f:
            header = {
                "descr": np.dtype(np.float32).str,
                "fortran_order": False,
                "shape": (n_samples, 2, *sample_shape),
            }
            npformat.write_array_header_2_0(out_f, header)

            for start in range(0, n_samples, int(chunk_rows)):
                rows = min(int(chunk_rows), n_samples - start)
                buf = _read_exact(f, rows * bytes_per_row)
                arr = np.frombuffer(buf, dtype=dtype).reshape((rows, *sample_shape))
                out_chunk = np.empty((rows, 2, *sample_shape), dtype=np.float32)
                out_chunk[:, 0, ...] = arr.real
                out_chunk[:, 1, ...] = arr.imag
                out_f.write(out_chunk.tobytes(order="C"))

                if start == 0 or (start // int(chunk_rows)) % 250 == 0 or start + rows == n_samples:
                    print(f"  converted rows {start + rows}/{n_samples}")


def ensure_sionna_param_cache(npz_path, cache_dir, cfg, convert_chunk_rows=4):
    npz_path = Path(npz_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not npz_path.exists():
        raise FileNotFoundError(f"Sionna npz not found: {npz_path}")

    xh_path = cache_dir / "X_h_ri.npy"
    xl_path = cache_dir / "X_l_ri.npy"
    k_path = cache_dir / "K_list.npy"
    ph_path = cache_dir / "P_h_clean.npy"
    pl_path = cache_dir / "P_l_clean.npy"
    y_phys_path = cache_dir / "Y_phys.npy"
    y_bin_path = cache_dir / "Y_bin.npy"
    y_delta_norm_path = cache_dir / "Y_delta_norm.npy"
    marker = cache_dir / "PARAM_CACHE_READY.txt"

    required = [xh_path, xl_path, k_path, ph_path, pl_path, y_phys_path, y_bin_path, y_delta_norm_path]
    if marker.exists() and all(p.exists() for p in required):
        print(f"Using existing Sionna ParamNet cache: {cache_dir}")
        return cache_dir

    with zipfile.ZipFile(npz_path, "r") as zf:
        for member in ("X_h.npy", "X_l.npy", "K_list.npy"):
            if member not in zf.namelist():
                raise KeyError(f"{npz_path} missing required member: {member}")
        xh_shape, xh_fortran, xh_dtype = _read_npy_header_from_zip(zf, "X_h.npy")
        xl_shape, xl_fortran, xl_dtype = _read_npy_header_from_zip(zf, "X_l.npy")

    print("\n========== Sionna ParamNet Cache ==========")
    print("npz:", npz_path)
    print("cache:", cache_dir)
    print("X_h:", xh_shape, xh_dtype, "fortran=", xh_fortran)
    print("X_l:", xl_shape, xl_dtype, "fortran=", xl_fortran)
    print("==========================================\n")

    if not k_path.exists() or not ph_path.exists() or not pl_path.exists() or not y_phys_path.exists():
        with np.load(npz_path, allow_pickle=False) as z:
            K = np.asarray(z["K_list"], dtype=np.int64).reshape(-1)
            np.save(k_path, K)
            if "P_h_clean" in z:
                P_h = np.asarray(z["P_h_clean"], dtype=np.float32).reshape(-1)
            else:
                P_h = np.ones_like(K, dtype=np.float32)
            if "P_l_clean" in z:
                P_l = np.asarray(z["P_l_clean"], dtype=np.float32).reshape(-1)
            else:
                P_l = np.ones_like(P_h, dtype=np.float32)
            np.save(ph_path, P_h)
            np.save(pl_path, P_l)

            if "Y_rva" in z:
                Y_rva = np.asarray(z["Y_rva"], dtype=np.float32)[:, :, :3]
            elif "Y" in z:
                Y_rva = np.asarray(z["Y"], dtype=np.float32)[:, :, :3]
            else:
                raise KeyError("Sionna npz must contain Y_rva or Y")

            target_mask = np.zeros(Y_rva.shape[:2], dtype=np.float32)
            if "target_mask" in z:
                target_mask[:] = np.asarray(z["target_mask"], dtype=np.float32)
            else:
                for i, kk in enumerate(K):
                    target_mask[i, : int(kk)] = 1.0
            Y_phys = np.concatenate([target_mask[..., None], Y_rva], axis=2).astype(np.float32)
            Y_phys[target_mask <= 0.5, 1:4] = 0.0
            np.save(y_phys_path, Y_phys)

    Y_phys = np.load(y_phys_path, mmap_mode="r")
    if not y_bin_path.exists() or not y_delta_norm_path.exists():
        Y_phys_arr = np.asarray(Y_phys, dtype=np.float32)
        Y_bin = MultiTargetParamDataset._make_bins_from_phys(Y_phys_arr, cfg)
        Y_delta_phys = MultiTargetParamDataset._make_delta_phys_from_phys(Y_phys_arr, Y_bin, cfg)
        widths = np.array(
            [
                (cfg["r_max"] - cfg["r_min"]) / cfg["Nbin_r"],
                (cfg["v_max"] - cfg["v_min"]) / cfg["Nbin_v"],
                (cfg["theta_max"] - cfg["theta_min"]) / cfg["Nbin_theta"],
            ],
            dtype=np.float32,
        )
        Y_delta_norm = np.clip(Y_delta_phys / widths.reshape(1, 1, 3), -0.5, 0.5).astype(np.float32)
        np.save(y_bin_path, Y_bin.astype(np.int64))
        np.save(y_delta_norm_path, Y_delta_norm)

    _stream_complex_member_to_ri_npy(npz_path, "X_l.npy", xl_path, chunk_rows=max(int(convert_chunk_rows), 16))
    _stream_complex_member_to_ri_npy(npz_path, "X_h.npy", xh_path, chunk_rows=int(convert_chunk_rows))

    marker.write_text(f"source={npz_path}\n", encoding="utf-8")
    return cache_dir


class SionnaParamMemmapDataset(Dataset):
    def __init__(self, npz_path, cache_dir, cfg, convert_chunk_rows=4, target_p_h_ref=1.0):
        super().__init__()
        self.npz_path = str(npz_path)
        self.cfg = dict(cfg)
        self.cache_dir = ensure_sionna_param_cache(npz_path, cache_dir, self.cfg, convert_chunk_rows)
        self.xh_path = Path(self.cache_dir) / "X_h_ri.npy"
        self.xl_path = Path(self.cache_dir) / "X_l_ri.npy"
        self.k_path = Path(self.cache_dir) / "K_list.npy"
        self.ph_path = Path(self.cache_dir) / "P_h_clean.npy"
        self.y_phys_path = Path(self.cache_dir) / "Y_phys.npy"
        self.y_bin_path = Path(self.cache_dir) / "Y_bin.npy"
        self.y_delta_norm_path = Path(self.cache_dir) / "Y_delta_norm.npy"

        self.X_h = None
        self.X_l = None
        self.K_list_all = None
        self.P_h_all = None
        self.Y_phys_all = None
        self.Y_bin_all = None
        self.Y_delta_norm_all = None
        self._ensure_arrays_open()

        self.num_samples = int(self.K_list_all.shape[0])
        self.Kmax = int(self.Y_phys_all.shape[1])
        exist_all = np.asarray(self.K_list_all) > 0
        p_h_all = np.asarray(self.P_h_all, dtype=np.float32)
        valid_h = exist_all & np.isfinite(p_h_all) & (p_h_all > 0)
        raw_p_h_ref = float(np.median(p_h_all[valid_h])) if np.any(valid_h) else 1.0
        raw_p_h_ref = max(raw_p_h_ref, 1e-30)
        self.input_scale = math.sqrt(max(float(target_p_h_ref), 1e-30) / raw_p_h_ref)
        self.P_h_scale = self.input_scale ** 2
        self.P_h_ref = max(raw_p_h_ref * self.P_h_scale, 1e-30)

        print("\n========== Sionna Param Dataset Info ==========")
        print("file:", self.npz_path)
        print("cache:", self.cache_dir)
        print("num_samples:", self.num_samples)
        print("Kmax:", self.Kmax)
        print("K distribution:", {int(k): int(np.sum(np.asarray(self.K_list_all) == k)) for k in range(self.Kmax + 1)})
        print("Y_phys:", self.Y_phys_all.shape)
        print("Y_bin:", self.Y_bin_all.shape)
        print("X_h memmap shape:", self.X_h.shape)
        print("X_l memmap shape:", self.X_l.shape)
        print("cfg:", self.cfg)
        print(f"raw_P_h_ref = {raw_p_h_ref:.4e}")
        print(f"input_scale = {self.input_scale:.4e}")
        print(f"P_h_ref = {self.P_h_ref:.4e}")
        print("==============================================\n")

    def __len__(self):
        return self.num_samples

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in ["X_h", "X_l", "K_list_all", "P_h_all", "Y_phys_all", "Y_bin_all", "Y_delta_norm_all"]:
            state[key] = None
        return state

    def _ensure_arrays_open(self):
        if self.X_h is None:
            self.X_h = np.load(self.xh_path, mmap_mode="r")
        if self.X_l is None:
            self.X_l = np.load(self.xl_path, mmap_mode="r")
        if self.K_list_all is None:
            self.K_list_all = np.load(self.k_path, mmap_mode="r")
        if self.P_h_all is None:
            self.P_h_all = np.load(self.ph_path, mmap_mode="r")
        if self.Y_phys_all is None:
            self.Y_phys_all = np.load(self.y_phys_path, mmap_mode="r")
        if self.Y_bin_all is None:
            self.Y_bin_all = np.load(self.y_bin_path, mmap_mode="r")
        if self.Y_delta_norm_all is None:
            self.Y_delta_norm_all = np.load(self.y_delta_norm_path, mmap_mode="r")

    def __getitem__(self, idx):
        self._ensure_arrays_open()
        X_h = np.asarray(self.X_h[idx], dtype=np.float32) * np.float32(self.input_scale)
        X_l = np.asarray(self.X_l[idx], dtype=np.float32) * np.float32(self.input_scale)
        return {
            "X_h": torch.from_numpy(X_h),
            "X_l": torch.from_numpy(X_l),
            "K": torch.tensor(int(self.K_list_all[idx]), dtype=torch.long),
            "Y_phys": torch.from_numpy(np.asarray(self.Y_phys_all[idx], dtype=np.float32)),
            "Y_bin": torch.from_numpy(np.asarray(self.Y_bin_all[idx], dtype=np.int64)),
            "Y_delta_norm": torch.from_numpy(np.asarray(self.Y_delta_norm_all[idx], dtype=np.float32)),
            "P_h": torch.tensor(max(float(self.P_h_all[idx]) * self.P_h_scale, 1e-30), dtype=torch.float32),
            "idx": torch.tensor(idx, dtype=torch.long),
        }


# ======================================================================
# 2. AWGN and training utilities
# ======================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def add_awgn_to_batch(X_h, X_l, P_h, K_true, snr_db, P_h_ref):
    if snr_db is None:
        return X_h, X_l

    eps = 1e-30
    B = X_h.shape[0]
    device = X_h.device
    dtype = X_h.dtype

    P_h = torch.clamp(P_h.to(device=device, dtype=dtype).view(B), min=eps)
    exist_mask = K_true.to(device=device).view(B) > 0
    P_ref = torch.full((B,), float(P_h_ref), device=device, dtype=dtype).clamp_min(eps)
    P_for_noise = torch.where(exist_mask, P_h, P_ref)

    snr_db = torch.as_tensor(snr_db, device=device, dtype=dtype)
    if snr_db.ndim == 0:
        snr_db = snr_db.expand(B)
    else:
        snr_db = snr_db.view(B)

    sigma2 = P_for_noise / (10.0 ** (snr_db / 10.0))
    noise_std = torch.sqrt(sigma2 / 2.0)

    return (
        X_h + noise_std.view(B, 1, 1, 1, 1) * torch.randn_like(X_h),
        X_l + noise_std.view(B, 1, 1, 1, 1) * torch.randn_like(X_l),
    )


def get_train_snr_min(args):
    if args.train_snr_mode == "fixed_grid":
        return float(min(args.train_snr_list))
    if not args.use_snr_curriculum:
        return float(args.train_snr_min)

    epoch = max(int(getattr(args, "current_epoch", 1)), 1)
    progress = min(max((epoch - 1) / max(int(args.snr_curriculum_epochs) - 1, 1), 0.0), 1.0)
    return float(args.snr_curriculum_start_min) + progress * (
        float(args.train_snr_min) - float(args.snr_curriculum_start_min)
    )


def prepare_batch(batch, device, args, train=True):
    X_h = batch["X_h"].to(device, non_blocking=True)
    X_l = batch["X_l"].to(device, non_blocking=True)
    K_true = batch["K"].to(device, non_blocking=True)
    Y_phys = batch["Y_phys"].to(device, non_blocking=True)
    Y_bin = batch["Y_bin"].to(device, non_blocking=True)
    Y_delta_norm = batch["Y_delta_norm"].to(device, non_blocking=True)
    P_h = batch["P_h"].to(device, non_blocking=True)

    if train and args.train_use_awgn:
        if args.train_snr_mode == "uniform":
            snr_db = torch.empty(X_h.shape[0], device=device).uniform_(
                get_train_snr_min(args),
                float(args.train_snr_max),
            )
        elif args.train_snr_mode == "choice":
            snr_db = random.choice(args.train_snr_list)
        elif args.train_snr_mode == "fixed_grid":
            if "snr_db" not in batch:
                raise KeyError("fixed_grid training requires batch['snr_db']")
            snr_db = batch["snr_db"].to(device, non_blocking=True)
        else:
            raise ValueError(f"Unknown train_snr_mode: {args.train_snr_mode}")

        X_h, X_l = add_awgn_to_batch(X_h, X_l, P_h, K_true, snr_db, args.P_h_ref)

    if (not train) and (args.val_snr_db is not None):
        X_h, X_l = add_awgn_to_batch(X_h, X_l, P_h, K_true, args.val_snr_db, args.P_h_ref)

    return X_h, X_l, K_true, Y_phys, Y_bin, Y_delta_norm


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(dataset, batch_size, shuffle, num_workers, drop_last, seed, prefetch_factor):
    generator = torch.Generator()
    generator.manual_seed(seed)

    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


def make_random_split(dataset, train_ratio, seed):
    rng = np.random.default_rng(int(seed))
    indices = rng.permutation(len(dataset))
    n_train = int(round(float(train_ratio) * len(dataset)))
    n_train = min(max(n_train, 1), len(dataset) - 1)
    train_indices = indices[:n_train].tolist()
    val_indices = indices[n_train:].tolist()
    return Subset(dataset, train_indices), Subset(dataset, val_indices), train_indices, val_indices


def print_split_distribution(dataset, indices, name):
    k_all = np.asarray(dataset.K_list_all)
    k_sel = k_all[np.asarray(indices, dtype=np.int64)]
    dist = {int(k): int(np.sum(k_sel == k)) for k in range(dataset.Kmax + 1)}
    print(f"{name}: n={len(indices)}, K distribution={dist}")


def one_hot_count(K_true, Kmax):
    return F.one_hot(K_true.clamp(0, Kmax), num_classes=Kmax + 1).float()


def get_teacher_prob(epoch, args):
    if epoch <= args.teacher_forcing_epochs:
        return 1.0
    if epoch >= args.teacher_forcing_epochs + args.mix_decay_epochs:
        return float(args.min_teacher_prob)
    progress = (epoch - args.teacher_forcing_epochs) / max(args.mix_decay_epochs, 1)
    return 1.0 + progress * (float(args.min_teacher_prob) - 1.0)


@torch.no_grad()
def build_training_count_prior(model, X_h, X_l, K_true, args, epoch):
    count_out = model._run_count_net(X_h, X_l)
    pred_prior = torch.softmax(count_out["K_logits"], dim=1)

    teacher_prior = one_hot_count(K_true, args.Kmax).to(pred_prior)
    teacher_prob = get_teacher_prob(epoch, args)
    if teacher_prob >= 1.0:
        return teacher_prior, count_out, teacher_prob

    use_teacher = torch.rand(K_true.shape[0], device=K_true.device) < teacher_prob

    if args.pred_prior_type == "soft":
        pred_for_train = pred_prior
    elif args.pred_prior_type == "hard":
        pred_for_train = one_hot_count(torch.argmax(pred_prior, dim=1), args.Kmax).to(pred_prior)
    else:
        raise ValueError(f"Unknown pred_prior_type: {args.pred_prior_type}")

    prior = torch.where(use_teacher.view(-1, 1), teacher_prior, pred_for_train)
    return prior, count_out, teacher_prob


# ======================================================================
# 3. Hungarian loss
# ======================================================================
def robust_match(cost):
    P, T = cost.shape
    if P == 0 or T == 0:
        return []

    cost_np = cost.detach().float().cpu().clone()
    cost_np[~torch.isfinite(cost_np)] = 1e9
    cost_np = cost_np.numpy()

    if P >= T:
        best_val = float("inf")
        best_perm = None
        for pred_perm in itertools.permutations(range(P), T):
            val = sum(float(cost_np[p, t]) for t, p in enumerate(pred_perm))
            if val < best_val:
                best_val = val
                best_perm = pred_perm
        return [(int(best_perm[t]), int(t)) for t in range(T)] if best_perm is not None else []

    best_val = float("inf")
    best_perm = None
    for true_perm in itertools.permutations(range(T), P):
        val = sum(float(cost_np[p, t]) for p, t in enumerate(true_perm))
        if val < best_val:
            best_val = val
            best_perm = true_perm
    return [(int(p), int(best_perm[p])) for p in range(P)] if best_perm is not None else []


@torch.no_grad()
def build_matching_pairs(out, Y_phys, Y_bin, Y_delta_norm, args):
    B, P = out["p_logit"].shape
    logp_r = F.log_softmax(out["r_logits"], dim=-1)
    logp_v = F.log_softmax(out["v_logits"], dim=-1)
    logp_th = F.log_softmax(out["theta_logits"], dim=-1)
    pred_soft_r = soft_bin_expectation(out["r_logits"])
    pred_soft_v = soft_bin_expectation(out["v_logits"])
    pred_soft_th = soft_bin_expectation(out["theta_logits"])
    pred_bins = out["pred_bins"].float()
    obj_cost = F.binary_cross_entropy_with_logits(
        out["p_logit"],
        torch.ones_like(out["p_logit"]),
        reduction="none",
    )

    all_pairs = []
    for b in range(B):
        true_idx = torch.nonzero(Y_phys[b, :, 0] > 0.5, as_tuple=False).squeeze(-1)
        T = int(true_idx.numel())
        if T == 0:
            all_pairs.append([])
            continue

        cost = torch.zeros(P, T, device=out["p_logit"].device)
        true_bin = Y_bin[b, true_idx].long()
        true_delta = Y_delta_norm[b, true_idx]

        for j in range(T):
            rb = int(true_bin[j, 0].item())
            vb = int(true_bin[j, 1].item())
            tb = int(true_bin[j, 2].item())

            cost_cls = (
                args.match_lambda_r * (-logp_r[b, :, rb])
                + args.match_lambda_v * (-logp_v[b, :, vb])
                + args.match_lambda_theta * (-logp_th[b, :, tb])
            )
            true_cont = true_bin[j].float() + true_delta[j].float()
            pred_soft = torch.stack(
                [pred_soft_r[b], pred_soft_v[b], pred_soft_th[b]],
                dim=-1,
            )
            denom = torch.tensor(
                [
                    max(args.Nbin_r - 1, 1),
                    max(args.Nbin_v - 1, 1),
                    max(args.Nbin_theta - 1, 1),
                ],
                device=out["p_logit"].device,
                dtype=out["p_logit"].dtype,
            ).view(1, 3)
            cost_geo = F.smooth_l1_loss(
                pred_soft / denom,
                true_cont.to(pred_soft).view(1, 3).expand_as(pred_soft) / denom,
                reduction="none",
            ).mean(dim=1)
            cost_arg = (
                (pred_bins[b] - true_bin[j].float().view(1, 3))
                .abs()
                .clamp_max(float(args.match_arg_clip_bins))
                / float(args.match_arg_clip_bins)
            ).mean(dim=1)
            cost[:, j] = (
                args.match_lambda_cls * cost_cls
                + args.match_lambda_geo * cost_geo
                + args.match_lambda_arg * cost_arg
                + args.match_lambda_obj * obj_cost[b]
            )

        local_pairs = robust_match(cost)
        all_pairs.append([(p, int(true_idx[t].item())) for p, t in local_pairs])

    return all_pairs


def weighted_mean(x, w, eps=1e-12):
    return (x * w).sum() / (w.sum() + eps)


def neighbor_soft_cross_entropy(logits, target, neighbor_weight=0.0):
    neighbor_weight = float(neighbor_weight)
    if neighbor_weight <= 0.0:
        return F.cross_entropy(logits, target, reduction="none")

    num_classes = logits.shape[-1]
    idx = torch.arange(target.numel(), device=target.device)
    target_dist = torch.zeros_like(logits)
    main_weight = max(1.0 - 2.0 * neighbor_weight, 1e-6)
    target_dist[idx, target] = main_weight

    left = target > 0
    if left.any():
        target_dist[idx[left], target[left] - 1] += neighbor_weight

    right = target < num_classes - 1
    if right.any():
        target_dist[idx[right], target[right] + 1] += neighbor_weight

    target_dist = target_dist / target_dist.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return -(target_dist * F.log_softmax(logits, dim=-1)).sum(dim=-1)


def soft_bin_expectation(logits):
    probs = F.softmax(logits, dim=-1)
    grid = torch.arange(logits.shape[-1], device=logits.device, dtype=logits.dtype)
    return (probs * grid.view(1, -1)).sum(dim=-1)


def soft_bin_expectation_loss(logits, target_cont):
    pred_cont = soft_bin_expectation(logits)
    denom = max(float(logits.shape[-1] - 1), 1.0)
    return F.smooth_l1_loss(pred_cont / denom, target_cont.to(pred_cont) / denom, reduction="mean")


def ordinal_probability_distance_loss(logits, target_cont, power=1.0):
    probs = F.softmax(logits, dim=-1)
    grid = torch.arange(logits.shape[-1], device=logits.device, dtype=logits.dtype)
    dist = (grid.view(1, -1) - target_cont.to(logits).view(-1, 1)).abs()
    if float(power) != 1.0:
        dist = dist.pow(float(power))
    denom = max(float(logits.shape[-1] - 1), 1.0)
    return (probs * dist).sum(dim=-1).mean() / denom


def ordinal_probability_distance_each(logits, target_cont, power=1.0):
    probs = F.softmax(logits, dim=-1)
    grid = torch.arange(logits.shape[-1], device=logits.device, dtype=logits.dtype)
    dist = (grid.view(1, -1) - target_cont.to(logits).view(-1, 1)).abs()
    if float(power) != 1.0:
        dist = dist.pow(float(power))
    denom = max(float(logits.shape[-1] - 1), 1.0)
    return (probs * dist).sum(dim=-1) / denom


def bin_error_metrics(pred_bin, true_bin):
    err_abs = (pred_bin.long() - true_bin.long()).abs().float()
    axis_names = ("r", "v", "theta")
    metrics = {}
    for axis, name in enumerate(axis_names):
        axis_err = err_abs[:, axis]
        metrics[f"{name}_bin_mae"] = axis_err.mean()
        metrics[f"{name}_bin_rmse"] = axis_err.square().mean().sqrt()
        metrics[f"{name}_tail_gt1"] = (axis_err > 1).float().mean()
        metrics[f"{name}_tail_gt2"] = (axis_err > 2).float().mean()
        metrics[f"{name}_tail_gt3"] = (axis_err > 3).float().mean()
    return metrics


TAIL_METRIC_KEYS = [
    f"{name}_{suffix}"
    for name in ("r", "v", "theta")
    for suffix in ("bin_mae", "bin_rmse", "tail_gt1", "tail_gt2", "tail_gt3")
]


def paramnet_loss(out, Y_phys, Y_bin, Y_delta_norm, K_true, args):
    B, P = out["p_logit"].shape
    device = out["p_logit"].device
    pairs_list = build_matching_pairs(out, Y_phys, Y_bin, Y_delta_norm, args)

    obj_target = torch.zeros_like(out["p_logit"])
    matched_b, matched_p, matched_t = [], [], []
    for b, pairs in enumerate(pairs_list):
        for p_idx, t_idx in pairs:
            obj_target[b, p_idx] = 1.0
            matched_b.append(b)
            matched_p.append(p_idx)
            matched_t.append(t_idx)

    k_weights = torch.tensor(args.k_param_weights, dtype=out["p_logit"].dtype, device=device)
    K_clamped = K_true.clamp(0, len(args.k_param_weights) - 1)
    sample_w = k_weights[K_clamped]

    obj_weight = torch.where(
        obj_target > 0.5,
        torch.ones_like(obj_target),
        torch.full_like(obj_target, float(args.no_object_weight)),
    )
    loss_obj_each = F.binary_cross_entropy_with_logits(out["p_logit"], obj_target, reduction="none")
    loss_obj = weighted_mean((loss_obj_each * obj_weight).mean(dim=1), sample_w)

    zero = out["p_logit"].sum() * 0.0
    loss_r = zero
    loss_v = zero
    loss_theta = zero
    loss_delta = zero
    loss_softbin = zero
    loss_orddist = zero
    bin_acc = torch.zeros(3, device=device)
    bin_w1 = torch.zeros(3, device=device)
    delta_mae = torch.zeros(3, device=device)
    tail_metrics = {k: torch.zeros((), device=device) for k in TAIL_METRIC_KEYS}

    if len(matched_b) > 0:
        mb = torch.tensor(matched_b, dtype=torch.long, device=device)
        mp = torch.tensor(matched_p, dtype=torch.long, device=device)
        mt = torch.tensor(matched_t, dtype=torch.long, device=device)
        true_bin = Y_bin[mb, mt].long()
        true_delta = Y_delta_norm[mb, mt]
        w_m = sample_w[mb]

        loss_r_each = neighbor_soft_cross_entropy(
            out["r_logits"][mb, mp],
            true_bin[:, 0],
            getattr(args, "r_neighbor_ce", 0.0),
        )
        loss_v_each = neighbor_soft_cross_entropy(
            out["v_logits"][mb, mp],
            true_bin[:, 1],
            getattr(args, "v_neighbor_ce", 0.0),
        )
        loss_theta_each = neighbor_soft_cross_entropy(
            out["theta_logits"][mb, mp],
            true_bin[:, 2],
            getattr(args, "theta_neighbor_ce", 0.0),
        )
        loss_delta_each = F.smooth_l1_loss(out["delta_norm"][mb, mp], true_delta, reduction="none").mean(dim=1)
        true_cont = true_bin.float() + true_delta.float()

        loss_softbin_r_each = F.smooth_l1_loss(
            soft_bin_expectation(out["r_logits"][mb, mp]) / max(args.Nbin_r - 1, 1),
            true_cont[:, 0] / max(args.Nbin_r - 1, 1),
            reduction="none",
        )
        loss_softbin_v_each = F.smooth_l1_loss(
            soft_bin_expectation(out["v_logits"][mb, mp]) / max(args.Nbin_v - 1, 1),
            true_cont[:, 1] / max(args.Nbin_v - 1, 1),
            reduction="none",
        )
        loss_softbin_theta_each = F.smooth_l1_loss(
            soft_bin_expectation(out["theta_logits"][mb, mp]) / max(args.Nbin_theta - 1, 1),
            true_cont[:, 2] / max(args.Nbin_theta - 1, 1),
            reduction="none",
        )
        loss_orddist_r_each = ordinal_probability_distance_each(out["r_logits"][mb, mp], true_cont[:, 0])
        loss_orddist_v_each = ordinal_probability_distance_each(out["v_logits"][mb, mp], true_cont[:, 1])
        loss_orddist_theta_each = ordinal_probability_distance_each(out["theta_logits"][mb, mp], true_cont[:, 2])

        loss_r = weighted_mean(loss_r_each, w_m)
        loss_v = weighted_mean(loss_v_each, w_m)
        loss_theta = weighted_mean(loss_theta_each, w_m)
        loss_delta = weighted_mean(loss_delta_each, w_m)
        softbin_weight_sum = args.lambda_r_softbin + args.lambda_v_softbin + args.lambda_theta_softbin
        loss_softbin = (
            args.lambda_r_softbin * weighted_mean(loss_softbin_r_each, w_m)
            + args.lambda_v_softbin * weighted_mean(loss_softbin_v_each, w_m)
            + args.lambda_theta_softbin * weighted_mean(loss_softbin_theta_each, w_m)
        ) / max(float(softbin_weight_sum), 1e-12)
        orddist_weight_sum = args.lambda_r_orddist + args.lambda_v_orddist + args.lambda_theta_orddist
        loss_orddist = (
            args.lambda_r_orddist * weighted_mean(loss_orddist_r_each, w_m)
            + args.lambda_v_orddist * weighted_mean(loss_orddist_v_each, w_m)
            + args.lambda_theta_orddist * weighted_mean(loss_orddist_theta_each, w_m)
        ) / max(float(orddist_weight_sum), 1e-12)

        with torch.no_grad():
            pred_bin = out["pred_bins"][mb, mp].long()
            err = pred_bin - true_bin
            bin_acc = (err == 0).float().mean(dim=0)
            bin_w1 = (torch.abs(err) <= 1).float().mean(dim=0)
            delta_mae = torch.abs(out["delta_norm"][mb, mp] - true_delta).mean(dim=0)
            tail_metrics = bin_error_metrics(pred_bin, true_bin)

    cls_weight_sum = args.lambda_r_cls + args.lambda_v_cls + args.lambda_theta_cls
    loss_cls = (
        args.lambda_r_cls * loss_r
        + args.lambda_v_cls * loss_v
        + args.lambda_theta_cls * loss_theta
    ) / cls_weight_sum

    expected_count = torch.sigmoid(out["p_logit"]).sum(dim=1)
    loss_card = F.smooth_l1_loss(expected_count, K_true.float())

    loss = (
        args.lambda_obj * loss_obj
        + args.lambda_cls * loss_cls
        + args.lambda_delta * loss_delta
        + args.lambda_card * loss_card
        + args.lambda_softbin * loss_softbin
        + args.lambda_orddist * loss_orddist
    )

    with torch.no_grad():
        p_prob = torch.sigmoid(out["p_logit"])
        pred_count_threshold = (p_prob >= args.det_threshold).sum(dim=1).long()
        pred_count_topk = out["K_pred"].clamp(0, args.Kmax)
        count_acc_threshold = (pred_count_threshold == K_true).float().mean()
        count_acc_countnet = (pred_count_topk == K_true).float().mean()
        obj_pos = p_prob[obj_target > 0.5].mean() if (obj_target > 0.5).any() else zero.detach()
        obj_neg = p_prob[obj_target < 0.5].mean() if (obj_target < 0.5).any() else zero.detach()

    return loss, {
        "loss_obj": loss_obj.detach(),
        "loss_cls": loss_cls.detach(),
        "loss_r": loss_r.detach(),
        "loss_v": loss_v.detach(),
        "loss_theta": loss_theta.detach(),
        "loss_delta": loss_delta.detach(),
        "loss_softbin": loss_softbin.detach(),
        "loss_orddist": loss_orddist.detach(),
        "loss_card": loss_card.detach(),
        "r_acc": bin_acc[0].detach(),
        "v_acc": bin_acc[1].detach(),
        "theta_acc": bin_acc[2].detach(),
        "r_w1": bin_w1[0].detach(),
        "v_w1": bin_w1[1].detach(),
        "theta_w1": bin_w1[2].detach(),
        "delta_r_mae": delta_mae[0].detach(),
        "delta_v_mae": delta_mae[1].detach(),
        "delta_theta_mae": delta_mae[2].detach(),
        "count_acc_threshold": count_acc_threshold.detach(),
        "count_acc_countnet": count_acc_countnet.detach(),
        "obj_pos": obj_pos.detach(),
        "obj_neg": obj_neg.detach(),
        **{k: v.detach() for k, v in tail_metrics.items()},
    }


# ======================================================================
# 4. Metrics and checkpointing
# ======================================================================
def tensor_to_float(x):
    if x is None:
        return float("nan")
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def init_sums():
    return {
        "loss": 0.0,
        "obj": 0.0,
        "cls": 0.0,
        "delta": 0.0,
        "softbin": 0.0,
        "orddist": 0.0,
        "card": 0.0,
        "r_acc": 0.0,
        "v_acc": 0.0,
        "theta_acc": 0.0,
        "r_w1": 0.0,
        "v_w1": 0.0,
        "theta_w1": 0.0,
        "delta_r_mae": 0.0,
        "delta_v_mae": 0.0,
        "delta_theta_mae": 0.0,
        "count_acc_threshold": 0.0,
        "count_acc_countnet": 0.0,
        "obj_pos": 0.0,
        "obj_neg": 0.0,
        **{k: 0.0 for k in TAIL_METRIC_KEYS},
    }


def update_sums(sums, loss, stats, n):
    sums["loss"] += tensor_to_float(loss) * n
    sums["obj"] += tensor_to_float(stats["loss_obj"]) * n
    sums["cls"] += tensor_to_float(stats["loss_cls"]) * n
    sums["delta"] += tensor_to_float(stats["loss_delta"]) * n
    sums["softbin"] += tensor_to_float(stats["loss_softbin"]) * n
    sums["orddist"] += tensor_to_float(stats["loss_orddist"]) * n
    sums["card"] += tensor_to_float(stats["loss_card"]) * n
    for k in [
        "r_acc",
        "v_acc",
        "theta_acc",
        "r_w1",
        "v_w1",
        "theta_w1",
        "delta_r_mae",
        "delta_v_mae",
        "delta_theta_mae",
        "count_acc_threshold",
        "count_acc_countnet",
        "obj_pos",
        "obj_neg",
        *TAIL_METRIC_KEYS,
    ]:
        sums[k] += tensor_to_float(stats[k]) * n


def finalize_sums(sums, total):
    return {k: v / max(total, 1) for k, v in sums.items()}


def print_metrics(title, m):
    print(
        f"{title} | loss={m['loss']:.5f} obj={m['obj']:.4f} cls={m['cls']:.4f} "
        f"delta={m['delta']:.4f} softbin={m['softbin']:.4f} ord={m['orddist']:.4f} card={m['card']:.4f} | "
        f"acc=({m['r_acc']:.3f},{m['v_acc']:.3f},{m['theta_acc']:.3f}) "
        f"w1=({m['r_w1']:.3f},{m['v_w1']:.3f},{m['theta_w1']:.3f}) | "
        f"tail>1=({m['r_tail_gt1']:.3f},{m['v_tail_gt1']:.3f},{m['theta_tail_gt1']:.3f}) | "
        f"cnt_countnet={m['count_acc_countnet']:.3f} cnt_obj={m['count_acc_threshold']:.3f}"
    )


def build_scheduler(optimizer, args, steps_per_epoch):
    total_steps = max(int(args.epochs) * int(steps_per_epoch), 1)
    warmup_steps = min(max(int(args.lr_warmup_batches), 0), total_steps)
    min_lr_ratio = min(max(float(args.min_lr) / float(args.lr), 0.0), 1.0)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_history_csv(history, path):
    if not history:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_metric, args):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_metric": best_metric,
            "args": {k: getattr(args, k) for k in dir(args) if not k.startswith("_") and not callable(getattr(args, k))},
        },
        path,
    )


def args_to_config(args):
    config = {}
    for k in dir(args):
        if k.startswith("_"):
            continue
        v = getattr(args, k)
        if callable(v):
            continue
        if isinstance(v, (int, float, str, bool, type(None))):
            config[k] = v
        elif isinstance(v, (list, tuple)):
            config[k] = list(v)
    return config


def log_swanlab(log_dict, step=None):
    if swanlab is None:
        return
    clean = {
        k: v
        for k, v in log_dict.items()
        if not isinstance(v, (float, np.floating)) or math.isfinite(float(v))
    }
    if step is None:
        swanlab.log(clean)
        return
    try:
        swanlab.log(clean, step=step)
    except TypeError:
        clean.setdefault("batch", step)
        swanlab.log(clean)


def metrics_to_log(prefix, metrics):
    return {f"{prefix}/{k}": float(v) for k, v in metrics.items()}


# ======================================================================
# 5. Build model
# ======================================================================
def count_kwargs_from_args(args):
    return {
        "Kmax": args.Kmax,
        "spec_size": args.spec_size,
        "token_grid": args.token_grid,
        "use_high": args.use_high,
        "use_low": args.use_low,
        "base_ch": args.count_base_ch,
        "embed_dim": args.count_embed_dim,
        "num_heads": args.count_num_heads,
        "cross_attn_layers": args.count_cross_attn_layers,
        "fusion_attn_layers": args.count_fusion_attn_layers,
        "mlp_ratio": args.count_mlp_ratio,
        "attn_dropout": args.count_attn_dropout,
        "token_dropout": args.count_token_dropout,
        "hidden_dim": args.count_hidden_dim,
        "dropout": args.count_dropout,
    }


def build_model(args, device):
    model = VelocityEnhancedParamNet(
        Kmax=args.Kmax,
        Nbin_r=args.Nbin_r,
        Nbin_v=args.Nbin_v,
        Nbin_theta=args.Nbin_theta,
        r_min=args.r_min,
        r_max=args.r_max,
        v_min=args.v_min,
        v_max=args.v_max,
        theta_min=args.theta_min,
        theta_max=args.theta_max,
        count_kwargs=count_kwargs_from_args(args),
        count_model_path=args.count_model_path,
        freeze_count_backbone=args.freeze_count_backbone,
        spec_size=args.spec_size,
        decoder_dim=args.decoder_dim,
        decoder_heads=args.decoder_heads,
        decoder_layers=args.decoder_layers,
        decoder_ffn_dim=args.decoder_ffn_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        spectrum_base_ch=args.spectrum_base_ch,
        velocity_base_ch=args.velocity_base_ch,
        velocity_dropout=args.velocity_dropout,
    ).to(device)

    if args.count_ckpt:
        print("Loading CountNet checkpoint:", args.count_ckpt)
        missing, unexpected = model.load_count_checkpoint(args.count_ckpt, map_location=device, strict=True)
        print("CountNet ckpt loaded. missing:", missing, "unexpected:", unexpected)

    init_param_ckpt = getattr(args, "init_param_ckpt", None)
    if init_param_ckpt:
        print("Loading ParamNet init checkpoint:", init_param_ckpt)
        ckpt = torch.load(init_param_ckpt, map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=True)
        print("ParamNet init loaded. missing:", missing, "unexpected:", unexpected)

    return model


@torch.no_grad()
def shape_check(model, batch, device, args):
    model.eval()
    X_h, X_l, K_true, _, _, _ = prepare_batch(batch, device, args, train=False)
    count_prior = one_hot_count(K_true, args.Kmax).to(device)
    out = model(X_h, X_l, count_prior=count_prior)

    print("\n========== ParamNet Shape Check ==========")
    print("X_h:", tuple(X_h.shape))
    print("X_l:", tuple(X_l.shape))
    for k in ["p_logit", "r_logits", "v_logits", "theta_logits", "delta_norm", "pred_bins", "pred_phys", "count_logits"]:
        print(f"{k}:", tuple(out[k].shape))
    print("==========================================\n")


# ======================================================================
# 6. Train/eval loops
# ======================================================================
def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, args, epoch, global_step):
    model.train()
    if args.freeze_count_backbone:
        model.count_net.eval()

    sums = init_sums()
    total = 0
    optimizer.zero_grad(set_to_none=True)

    for it, batch in enumerate(loader, 1):
        X_h, X_l, K_true, Y_phys, Y_bin, Y_delta_norm = prepare_batch(batch, device, args, train=True)
        count_prior, count_out, teacher_prob = build_training_count_prior(model, X_h, X_l, K_true, args, epoch)

        with torch.cuda.amp.autocast(enabled=bool(args.use_amp and device.type == "cuda")):
            out = model(X_h, X_l, count_prior=count_prior, count_out=count_out)
            loss, stats = paramnet_loss(out, Y_phys, Y_bin, Y_delta_norm, K_true, args)

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.grad_clip,
            )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

        n = X_h.shape[0]
        total += n
        global_step += 1
        update_sums(sums, loss.detach(), stats, n)

        if args.use_swanlab and global_step % args.swanlab_log_every_batches == 0:
            log_swanlab(
                {
                    "train/loss": tensor_to_float(loss),
                    "train/loss_obj": tensor_to_float(stats["loss_obj"]),
                    "train/loss_cls": tensor_to_float(stats["loss_cls"]),
                    "train/loss_delta": tensor_to_float(stats["loss_delta"]),
                    "train/loss_softbin": tensor_to_float(stats["loss_softbin"]),
                    "train/loss_orddist": tensor_to_float(stats["loss_orddist"]),
                    "train/loss_card": tensor_to_float(stats["loss_card"]),
                    "train/r_acc": tensor_to_float(stats["r_acc"]),
                    "train/v_acc": tensor_to_float(stats["v_acc"]),
                    "train/theta_acc": tensor_to_float(stats["theta_acc"]),
                    "train/r_w1": tensor_to_float(stats["r_w1"]),
                    "train/v_w1": tensor_to_float(stats["v_w1"]),
                    "train/theta_w1": tensor_to_float(stats["theta_w1"]),
                    "train/delta_r_mae": tensor_to_float(stats["delta_r_mae"]),
                    "train/delta_v_mae": tensor_to_float(stats["delta_v_mae"]),
                    "train/delta_theta_mae": tensor_to_float(stats["delta_theta_mae"]),
                    "train/r_bin_rmse": tensor_to_float(stats["r_bin_rmse"]),
                    "train/v_bin_rmse": tensor_to_float(stats["v_bin_rmse"]),
                    "train/theta_bin_rmse": tensor_to_float(stats["theta_bin_rmse"]),
                    "train/r_tail_gt1": tensor_to_float(stats["r_tail_gt1"]),
                    "train/v_tail_gt1": tensor_to_float(stats["v_tail_gt1"]),
                    "train/theta_tail_gt1": tensor_to_float(stats["theta_tail_gt1"]),
                    "train/r_tail_gt3": tensor_to_float(stats["r_tail_gt3"]),
                    "train/v_tail_gt3": tensor_to_float(stats["v_tail_gt3"]),
                    "train/theta_tail_gt3": tensor_to_float(stats["theta_tail_gt3"]),
                    "train/count_acc_threshold": tensor_to_float(stats["count_acc_threshold"]),
                    "train/count_acc_countnet": tensor_to_float(stats["count_acc_countnet"]),
                    "train/obj_pos": tensor_to_float(stats["obj_pos"]),
                    "train/obj_neg": tensor_to_float(stats["obj_neg"]),
                    "train/teacher_prob": teacher_prob,
                    "epoch": epoch,
                    "lr": optimizer.param_groups[0]["lr"],
                },
                step=global_step,
            )

        if it % args.print_every == 0:
            running = finalize_sums(sums, total)
            print(
                f"Epoch {epoch} iter {it}/{len(loader)} | loss={running['loss']:.5f} "
                f"acc=({running['r_acc']:.3f},{running['v_acc']:.3f},{running['theta_acc']:.3f}) "
                f"teacher_prob={teacher_prob:.2f}"
            )

    return finalize_sums(sums, total), global_step


@torch.no_grad()
def evaluate(model, loader, device, args, prior_mode="countnet"):
    if prior_mode not in ("countnet", "teacher"):
        raise ValueError(f"Unknown prior_mode: {prior_mode}")

    model.eval()
    sums = init_sums()
    total = 0

    for batch in loader:
        X_h, X_l, K_true, Y_phys, Y_bin, Y_delta_norm = prepare_batch(batch, device, args, train=False)
        count_prior = None
        if prior_mode == "teacher":
            count_prior = one_hot_count(K_true, args.Kmax).to(device)

        with torch.cuda.amp.autocast(enabled=bool(args.use_amp and device.type == "cuda")):
            out = model(X_h, X_l, count_prior=count_prior)
            loss, stats = paramnet_loss(out, Y_phys, Y_bin, Y_delta_norm, K_true, args)

        n = X_h.shape[0]
        total += n
        update_sums(sums, loss.detach(), stats, n)

    return finalize_sums(sums, total)


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using device:", device)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    os.makedirs(args.save_dir, exist_ok=True)

    if args.use_swanlab:
        if swanlab is None:
            raise ImportError("use_swanlab=True but swanlab is not installed.")
        swanlab.init(
            project=args.swanlab_project,
            experiment_name=args.swanlab_experiment_name,
            config=args_to_config(args),
            mode="cloud",
        )

    cfg = {
        "r_min": args.r_min,
        "r_max": args.r_max,
        "v_min": args.v_min,
        "v_max": args.v_max,
        "theta_min": args.theta_min,
        "theta_max": args.theta_max,
        "Nbin_r": args.Nbin_r,
        "Nbin_v": args.Nbin_v,
        "Nbin_theta": args.Nbin_theta,
    }

    if getattr(args, "use_sionna_h5", False):
        if not os.path.exists(args.sionna_h5):
            raise FileNotFoundError(
                f"Sionna HDF5 cache not found: {args.sionna_h5}\n"
                "Run: python data_tools/convert_sionna_npz_to_h5.py"
            )
        full_set = MultiTargetParamDataset(args.sionna_h5)
        base_train_set, val_set, train_indices, val_indices = make_random_split(
            full_set,
            args.train_ratio,
            args.split_seed,
        )
        print_split_distribution(full_set, train_indices, "train split")
        print_split_distribution(full_set, val_indices, "val split")
        args.P_h_ref = full_set.P_h_ref
        source_cfg = full_set.cfg
    elif getattr(args, "use_sionna_npz", False):
        full_set = SionnaParamMemmapDataset(
            npz_path=args.train_npz,
            cache_dir=args.cache_dir,
            cfg=cfg,
            convert_chunk_rows=args.convert_chunk_rows,
            target_p_h_ref=args.sionna_target_p_h_ref,
        )
        base_train_set, val_set, train_indices, val_indices = make_random_split(
            full_set,
            args.train_ratio,
            args.split_seed,
        )
        print_split_distribution(full_set, train_indices, "train split")
        print_split_distribution(full_set, val_indices, "val split")
        args.P_h_ref = full_set.P_h_ref
        source_cfg = full_set.cfg
    else:
        base_train_set = MultiTargetParamDataset(args.train_mat)
        val_set = MultiTargetParamDataset(args.test_mat)
        args.P_h_ref = base_train_set.P_h_ref
        source_cfg = base_train_set.cfg

    train_set = FixedSNRGridDataset(base_train_set, args.train_snr_list)

    for k, v in source_cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)

    print("\n========== Fixed SNR Grid ==========")
    print("clean training scenes:", len(base_train_set))
    print("SNR values (dB):", list(train_set.snr_values))
    print("noisy views per epoch:", len(train_set))
    print("online AWGN: enabled")
    print("====================================\n")

    train_loader = make_loader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        seed=args.seed,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = make_loader(
        val_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        seed=args.seed + 999,
        prefetch_factor=args.prefetch_factor,
    )

    model = build_model(args, device)
    count_parameters(model)
    first_batch = next(iter(train_loader))
    shape_check(model, first_batch, device, args)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(optimizer, args, len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.use_amp and device.type == "cuda"))

    best_score = -1.0
    best_w1_score = -1.0
    best_tail_score = -1.0
    best_loss = float("inf")
    history = []
    global_step = 0

    best_path = os.path.join(args.save_dir, "crossfusion_paramnet_best.pt")
    best_w1_path = os.path.join(args.save_dir, "crossfusion_paramnet_best_w1.pt")
    best_tail_path = os.path.join(args.save_dir, "crossfusion_paramnet_best_tail.pt")
    best_loss_path = os.path.join(args.save_dir, "crossfusion_paramnet_best_loss.pt")
    last_path = os.path.join(args.save_dir, "crossfusion_paramnet_last.pt")
    csv_path = os.path.join(args.save_dir, "crossfusion_paramnet_train_history.csv")

    for epoch in range(1, args.epochs + 1):
        args.current_epoch = epoch
        snr_min_now = get_train_snr_min(args)
        teacher_prob = get_teacher_prob(epoch, args)

        print(
            f"\n========== Epoch {epoch}/{args.epochs} | lr={optimizer.param_groups[0]['lr']:.3e} | "
            f"train_snr=[{snr_min_now:.1f},{args.train_snr_max:.1f}] | "
            f"teacher_prob={teacher_prob:.2f} =========="
        )

        train_m, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            args,
            epoch,
            global_step,
        )
        val_teacher_m = evaluate(model, val_loader, device, args, prior_mode="teacher")
        val_countnet_m = evaluate(model, val_loader, device, args, prior_mode="countnet")

        print_metrics(f"TRAIN epoch={epoch}", train_m)
        print_metrics(f"VAL teacher  epoch={epoch}", val_teacher_m)
        print_metrics(f"VAL countnet epoch={epoch}", val_countnet_m)

        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "teacher_prob": teacher_prob}
        for k, v in train_m.items():
            row[f"train_{k}"] = v
        for k, v in val_teacher_m.items():
            row[f"val_teacher_{k}"] = v
        for k, v in val_countnet_m.items():
            row[f"val_countnet_{k}"] = v
        history.append(row)
        save_history_csv(history, csv_path)

        score = (val_countnet_m["r_acc"] + val_countnet_m["v_acc"] + val_countnet_m["theta_acc"]) / 3.0
        w1_score = (val_countnet_m["r_w1"] + val_countnet_m["v_w1"] + val_countnet_m["theta_w1"]) / 3.0
        tail_gt1 = (
            val_countnet_m["r_tail_gt1"]
            + val_countnet_m["v_tail_gt1"]
            + val_countnet_m["theta_tail_gt1"]
        ) / 3.0
        tail_gt3 = (
            val_countnet_m["r_tail_gt3"]
            + val_countnet_m["v_tail_gt3"]
            + val_countnet_m["theta_tail_gt3"]
        ) / 3.0
        tail_score = w1_score - args.tail_score_gt1_weight * tail_gt1 - args.tail_score_gt3_weight * tail_gt3

        save_checkpoint(last_path, model, optimizer, scheduler, epoch, score, args)

        if val_countnet_m["loss"] < best_loss:
            best_loss = val_countnet_m["loss"]
            save_checkpoint(best_loss_path, model, optimizer, scheduler, epoch, best_loss, args)
            print(f"Saved best loss model: epoch={epoch}, loss={best_loss:.5f}")

        if score > best_score:
            best_score = score
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_score, args)
            print(f"Saved best bin-acc model: epoch={epoch}, mean_bin_acc={best_score:.5f}")

        if w1_score > best_w1_score:
            best_w1_score = w1_score
            save_checkpoint(best_w1_path, model, optimizer, scheduler, epoch, best_w1_score, args)
            print(f"Saved best w1 model: epoch={epoch}, mean_w1={best_w1_score:.5f}")

        if tail_score > best_tail_score:
            best_tail_score = tail_score
            save_checkpoint(best_tail_path, model, optimizer, scheduler, epoch, best_tail_score, args)
            print(
                f"Saved best tail model: epoch={epoch}, score={best_tail_score:.5f}, "
                f"mean_w1={w1_score:.5f}, tail_gt1={tail_gt1:.5f}, tail_gt3={tail_gt3:.5f}"
            )

        if args.use_swanlab:
            train_score = (train_m["r_acc"] + train_m["v_acc"] + train_m["theta_acc"]) / 3.0
            teacher_score = (val_teacher_m["r_acc"] + val_teacher_m["v_acc"] + val_teacher_m["theta_acc"]) / 3.0
            train_w1_score = (train_m["r_w1"] + train_m["v_w1"] + train_m["theta_w1"]) / 3.0
            teacher_w1_score = (val_teacher_m["r_w1"] + val_teacher_m["v_w1"] + val_teacher_m["theta_w1"]) / 3.0
            countnet_w1_score = (val_countnet_m["r_w1"] + val_countnet_m["v_w1"] + val_countnet_m["theta_w1"]) / 3.0
            countnet_tail_gt1 = (
                val_countnet_m["r_tail_gt1"]
                + val_countnet_m["v_tail_gt1"]
                + val_countnet_m["theta_tail_gt1"]
            ) / 3.0
            countnet_tail_gt3 = (
                val_countnet_m["r_tail_gt3"]
                + val_countnet_m["v_tail_gt3"]
                + val_countnet_m["theta_tail_gt3"]
            ) / 3.0

            log_dict = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "teacher_prob": teacher_prob,
                "score/train_bin_acc_mean": train_score,
                "score/val_teacher_bin_acc_mean": teacher_score,
                "score/val_countnet_bin_acc_mean": score,
                "score/train_w1_mean": train_w1_score,
                "score/val_teacher_w1_mean": teacher_w1_score,
                "score/val_countnet_w1_mean": countnet_w1_score,
                "score/val_countnet_tail_gt1_mean": countnet_tail_gt1,
                "score/val_countnet_tail_gt3_mean": countnet_tail_gt3,
                "score/val_countnet_tail_score": tail_score,
            }
            log_dict.update(metrics_to_log("train_epoch", train_m))
            log_dict.update(metrics_to_log("val_teacher", val_teacher_m))
            log_dict.update(metrics_to_log("val_countnet", val_countnet_m))
            log_swanlab(log_dict, step=global_step)

    save_history_csv(history, csv_path)

    print("\n========== ParamNet Training Finished ==========")
    print(f"Best mean bin acc: {best_score:.5f}")
    print(f"Best mean w1     : {best_w1_score:.5f}")
    print(f"Best tail score  : {best_tail_score:.5f}")
    print(f"Best loss        : {best_loss:.5f}")
    print("History saved to :", csv_path)
    print("Best ckpt        :", best_path)
    print("Best w1 ckpt     :", best_w1_path)
    print("Best tail ckpt   :", best_tail_path)
    print("Best loss ckpt   :", best_loss_path)
    print("Last ckpt        :", last_path)

    if args.use_swanlab:
        swanlab.finish()


# ======================================================================
# 7. Args
# ======================================================================
if __name__ == "__main__":
    class Args:
        _this_dir = os.path.dirname(os.path.abspath(__file__))
        _scheme_b_dir = os.path.dirname(_this_dir)
        if not os.path.exists(os.path.join(_scheme_b_dir, "widths")):
            _scheme_b_dir = _this_dir

        train_mat = ""
        test_mat = ""
        save_dir = os.path.join(RELEASE_ROOT, "outputs", "paramnet")

        count_model_path = os.path.join(RELEASE_ROOT, "models", "countnet.py")
        count_ckpt = os.path.join(RELEASE_ROOT, "weights", "countnet_sionna_best.pt")
        init_param_ckpt = None

        # Final override: train ParamNet v6 on the new clean Sionna dataset.
        # Use the HDF5 cache generated by convert_sionna_npz_to_paramnet_h5.py.
        # This avoids huge .npy memmap working-set growth on Windows.
        use_sionna_h5 = True
        use_sionna_npz = False
        train_npz = os.environ.get("SIONNA_TRAIN_NPZ", "")
        sionna_h5 = os.path.join(
            RELEASE_ROOT,
            "data",
            "beiyou_train_clean_50000_paramnet.h5",
        )
        cache_dir = os.path.join(
            RELEASE_ROOT,
            "cache",
            "paramnet",
        )
        convert_chunk_rows = 4
        sionna_target_p_h_ref = 1.0
        train_ratio = 0.8
        split_seed = 2026
        save_dir = os.path.join(RELEASE_ROOT, "outputs", "paramnet")
        count_model_path = os.path.join(RELEASE_ROOT, "models", "countnet.py")
        count_ckpt = os.path.join(RELEASE_ROOT, "weights", "countnet_sionna_best.pt")

        Kmax = 5
        Nbin_r = 32
        Nbin_v = 32
        Nbin_theta = 32
        r_min = 50.0
        r_max = 300.0
        v_min = -30.0
        v_max = 30.0
        theta_min = -math.pi / 3
        theta_max = math.pi / 3

        use_high = True
        use_low = True
        spec_size = (32, 32, 32)
        token_grid = (8, 8, 8)
        count_base_ch = 32
        count_embed_dim = 128
        count_num_heads = 4
        count_cross_attn_layers = 1
        count_fusion_attn_layers = 0
        count_mlp_ratio = 2.0
        count_attn_dropout = 0.10
        count_token_dropout = 0.10
        count_hidden_dim = 256
        count_dropout = 0.25

        freeze_count_backbone = True
        decoder_dim = 192
        decoder_heads = 6
        decoder_layers = 3
        decoder_ffn_dim = 768
        hidden_dim = 384
        dropout = 0.10
        spectrum_base_ch = 32
        velocity_base_ch = 24
        velocity_dropout = 0.14

        # Each epoch contains 7 x 50,000 views. 86 epochs gives almost the
        # same total updates as the earlier 5-SNR x 120-epoch v4 design.
        epochs = 86
        batch_size = 48
        test_batch_size = 48
        lr = 9e-5
        min_lr = 8.0e-6
        lr_warmup_batches = 900
        weight_decay = 2e-3
        grad_clip = 5.0
        use_amp = True

        # Preserve the v3 count-prior schedule in sample-exposure terms.
        teacher_forcing_epochs = 14
        mix_decay_epochs = 17
        min_teacher_prob = 0.45
        pred_prior_type = "soft"  # soft / hard

        lambda_obj = 0.65
        lambda_cls = 3.5
        # Coarse-bin detector: keep delta output for compatibility, but do
        # not train it. Sub-bin estimation is delegated to the refiner.
        lambda_delta = 0.0
        lambda_card = 0.005
        lambda_softbin = 2
        lambda_orddist = 1.5
        lambda_r_cls = 2.0
        lambda_v_cls = 3.6
        lambda_theta_cls = 2.0
        lambda_r_softbin = 2.0
        lambda_v_softbin = 3.0
        lambda_theta_softbin = 2.0
        lambda_r_orddist = 2.0
        lambda_v_orddist = 3.0
        lambda_theta_orddist = 2.0
        r_neighbor_ce = 0.03
        v_neighbor_ce = 0.06
        theta_neighbor_ce = 0.03
        no_object_weight = 0.4
        k_param_weights = [0.6, 1.0, 1.15, 1.35, 1.60, 1.90]

        match_lambda_obj = 0.5
        match_lambda_cls = 1.0
        match_lambda_delta = 0.0
        match_lambda_geo = 4.0
        match_lambda_arg = 2.0
        match_arg_clip_bins = 4.0
        match_lambda_r = 2.0
        match_lambda_v = 3.2
        match_lambda_theta = 2.0
        tail_score_gt1_weight = 0.35
        tail_score_gt3_weight = 1.20
        det_threshold = 0.5

        train_use_awgn = True
        train_snr_mode = "fixed_grid"
        train_snr_min = -10
        train_snr_max = 20
        use_snr_curriculum = False
        snr_curriculum_start_min = -10
        snr_curriculum_epochs = 1
        train_snr_list = [-10, -5, 0, 5, 10, 15, 20]
        val_snr_db = -10
        P_h_ref = None

        seed = 2027
        cpu = False
        num_workers = 12
        prefetch_factor = 6
        print_every = 100

        use_swanlab = True
        swanlab_project = "countnet-isac"
        swanlab_experiment_name = "paramnet_crossfusion_sionna_v6_coarsebin"
        swanlab_log_every_batches = 20

    main(Args())

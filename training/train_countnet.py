import os
import sys
import math
import copy

import csv
import random
import zipfile
from pathlib import Path
import numpy as np
from numpy.lib import format as npformat
import h5py
import swanlab

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

RELEASE_ROOT = Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

from models.countnet import DualBandCrossFusionCountNet, count_parameters


# ======================================================================
# 1. Dataset
# ======================================================================
class MultiTargetCountDataset(Dataset):
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
            self.Kmax = int(np.max(self.K_list_all))

            if "P_h_clean" in f:
                self.P_h_all = np.array(f["P_h_clean"], dtype=np.float32).squeeze()
            else:
                self.P_h_all = np.ones(self.num_samples, dtype=np.float32)

            exist_all = self.K_list_all > 0
            valid_h = exist_all & np.isfinite(self.P_h_all) & (self.P_h_all > 0)

            self.P_h_ref = float(np.median(self.P_h_all[valid_h])) if np.any(valid_h) else 1.0
            self.P_h_ref = max(self.P_h_ref, 1e-30)

            print("\n========== Count Dataset Info ==========")
            print("file:", mat_path)
            print("num_samples:", self.num_samples)
            print("Kmax:", self.Kmax)
            print("K distribution:", {int(k): int(np.sum(self.K_list_all == k)) for k in range(self.Kmax + 1)})
            print("X_h_real shape:", f["X_h_real"].shape)
            print("X_l_real shape:", f["X_l_real"].shape)
            print(f"P_h_ref = {self.P_h_ref:.4e}")
            print("========================================\n")

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

        # MATLAB v7.3 common layout: [Nc, Ms, Nr, N]
        if len(s) == 4 and s[0] == Nc and s[1] == Ms and s[2] == Nr:
            return np.array(d[:, :, :, idx], dtype=np.float32)

        # h5py possible layout: [N, Nr, Ms, Nc]
        if (
            len(s) == 4
            and s[0] == self.num_samples
            and s[1] == Nr
            and s[2] == Ms
            and s[3] == Nc
        ):
            arr = np.array(d[idx, :, :, :], dtype=np.float32)
            return np.transpose(arr, (2, 1, 0))

        # compatible layout: [N, Nc, Ms, Nr]
        if (
            len(s) == 4
            and s[0] == self.num_samples
            and s[1] == Nc
            and s[2] == Ms
            and s[3] == Nr
        ):
            return np.array(d[idx, :, :, :], dtype=np.float32)

        raise ValueError(f"无法识别 {name} shape={s}, expected={expected_shape}")

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
            "P_h": torch.tensor(max(float(self.P_h_all[idx]), 1e-30), dtype=torch.float32),
            "idx": torch.tensor(idx, dtype=torch.long),
        }


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


def _stream_complex_member_to_ri_memmap(npz_path, member, out_path, chunk_rows=4):
    out_path = Path(out_path)

    with zipfile.ZipFile(npz_path, "r") as zf, zf.open(member, "r") as f:
        version = npformat.read_magic(f)
        shape, fortran_order, dtype = npformat._read_array_header(f, version)

        if fortran_order:
            raise ValueError(f"{member} is Fortran ordered; streaming converter expects C order.")
        if dtype != np.dtype(np.complex128):
            raise ValueError(f"{member} dtype={dtype}, expected complex128.")
        if len(shape) != 4:
            raise ValueError(f"{member} shape={shape}, expected [N, Nc, Ms, Nr].")

        n_samples = int(shape[0])
        sample_shape = tuple(int(v) for v in shape[1:])
        elems_per_row = int(np.prod(sample_shape))
        bytes_per_row = elems_per_row * dtype.itemsize

        print(f"Converting {member} -> {out_path}")
        print(f"  source shape={shape}, output shape={(n_samples, 2, *sample_shape)}, chunk_rows={chunk_rows}")

        # Do not use write-mode memmap here. On Windows, writing tens of GB via
        # a mmap can make the modified-page cache grow until RAM is exhausted.
        # A normal sequential .npy writer keeps memory bounded by one chunk.
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


def _save_npz_small_arrays(npz_path, cache_dir):
    with np.load(npz_path, allow_pickle=False) as z:
        np.save(cache_dir / "K_list.npy", np.asarray(z["K_list"], dtype=np.int64))

        if "P_h_clean" in z:
            p_h = np.asarray(z["P_h_clean"], dtype=np.float32)
        else:
            p_h = np.ones_like(np.asarray(z["K_list"], dtype=np.float32), dtype=np.float32)
        np.save(cache_dir / "P_h_clean.npy", p_h)

        if "P_l_clean" in z:
            p_l = np.asarray(z["P_l_clean"], dtype=np.float32)
        else:
            p_l = np.ones_like(p_h, dtype=np.float32)
        np.save(cache_dir / "P_l_clean.npy", p_l)


def ensure_sionna_npz_cache(npz_path, cache_dir, convert_chunk_rows=4):
    npz_path = Path(npz_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    marker = cache_dir / "CACHE_READY.txt"
    xh_path = cache_dir / "X_h_ri.npy"
    xl_path = cache_dir / "X_l_ri.npy"
    k_path = cache_dir / "K_list.npy"
    ph_path = cache_dir / "P_h_clean.npy"
    pl_path = cache_dir / "P_l_clean.npy"

    required = [xh_path, xl_path, k_path, ph_path, pl_path]
    if marker.exists() and all(p.exists() for p in required):
        print(f"Using existing Sionna memmap cache: {cache_dir}")
        return cache_dir

    if not npz_path.exists():
        raise FileNotFoundError(f"Sionna npz not found: {npz_path}")

    with zipfile.ZipFile(npz_path, "r") as zf:
        for member in ("X_h.npy", "X_l.npy", "K_list.npy"):
            if member not in zf.namelist():
                raise KeyError(f"{npz_path} missing required member: {member}")

        xh_shape, xh_fortran, xh_dtype = _read_npy_header_from_zip(zf, "X_h.npy")
        xl_shape, xl_fortran, xl_dtype = _read_npy_header_from_zip(zf, "X_l.npy")

    print("\n========== Sionna NPZ Cache Build ==========")
    print("npz:", npz_path)
    print("cache:", cache_dir)
    print("X_h:", xh_shape, xh_dtype, "fortran=", xh_fortran)
    print("X_l:", xl_shape, xl_dtype, "fortran=", xl_fortran)
    print("This one-time conversion writes float32 real/imag memmap arrays.")
    print("===========================================\n")

    _save_npz_small_arrays(npz_path, cache_dir)
    _stream_complex_member_to_ri_memmap(npz_path, "X_l.npy", xl_path, chunk_rows=max(int(convert_chunk_rows), 16))
    _stream_complex_member_to_ri_memmap(npz_path, "X_h.npy", xh_path, chunk_rows=int(convert_chunk_rows))

    marker.write_text(
        f"source={npz_path}\nX_h={xh_shape},{xh_dtype}\nX_l={xl_shape},{xl_dtype}\n",
        encoding="utf-8",
    )
    print(f"Sionna memmap cache ready: {cache_dir}")
    return cache_dir


class SionnaCountMemmapDataset(Dataset):
    def __init__(
        self,
        npz_path,
        cache_dir,
        convert_chunk_rows=4,
        target_p_h_ref=1.0,
    ):
        super().__init__()

        self.npz_path = str(npz_path)
        self.cache_dir = ensure_sionna_npz_cache(
            npz_path=npz_path,
            cache_dir=cache_dir,
            convert_chunk_rows=convert_chunk_rows,
        )

        self.xh_path = self.cache_dir / "X_h_ri.npy"
        self.xl_path = self.cache_dir / "X_l_ri.npy"
        self.k_path = self.cache_dir / "K_list.npy"
        self.ph_path = self.cache_dir / "P_h_clean.npy"

        self.X_h = None
        self.X_l = None
        self.K_list_all = None
        self.P_h_all = None
        self._ensure_arrays_open()

        self.num_samples = int(self.K_list_all.shape[0])
        self.Kmax = int(np.max(self.K_list_all))

        exist_all = np.asarray(self.K_list_all) > 0
        p_h_all = np.asarray(self.P_h_all, dtype=np.float32)
        valid_h = exist_all & np.isfinite(p_h_all) & (p_h_all > 0)
        raw_p_h_ref = float(np.median(p_h_all[valid_h])) if np.any(valid_h) else 1.0
        raw_p_h_ref = max(raw_p_h_ref, 1e-30)

        # Sionna tensors are around 1e-8 in amplitude, so the FFT preprocessor's
        # eps clamp would flatten useful log-power statistics. Rescale inputs to
        # the same order as the original MATLAB dataset; SNR is preserved because
        # P_h is scaled by the same power factor below.
        self.input_scale = math.sqrt(max(float(target_p_h_ref), 1e-30) / raw_p_h_ref)
        self.P_h_scale = self.input_scale ** 2
        self.P_h_ref = raw_p_h_ref * self.P_h_scale
        self.P_h_ref = max(self.P_h_ref, 1e-30)

        print("\n========== Sionna Count Dataset Info ==========")
        print("file:", self.npz_path)
        print("cache:", self.cache_dir)
        print("num_samples:", self.num_samples)
        print("Kmax:", self.Kmax)
        print("K distribution:", {int(k): int(np.sum(np.asarray(self.K_list_all) == k)) for k in range(self.Kmax + 1)})
        print("X_h memmap shape:", self.X_h.shape)
        print("X_l memmap shape:", self.X_l.shape)
        print(f"raw_P_h_ref = {raw_p_h_ref:.4e}")
        print(f"input_scale = {self.input_scale:.4e}")
        print(f"P_h_ref = {self.P_h_ref:.4e}")
        print("==============================================\n")

    def __len__(self):
        return self.num_samples

    def __getstate__(self):
        state = self.__dict__.copy()
        # Windows DataLoader uses spawn and pickles the Dataset. Do not pickle
        # memmap objects; workers reopen them lazily from file paths.
        state["X_h"] = None
        state["X_l"] = None
        state["K_list_all"] = None
        state["P_h_all"] = None
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

    def __getitem__(self, idx):
        self._ensure_arrays_open()

        X_h = np.asarray(self.X_h[idx], dtype=np.float32) * np.float32(self.input_scale)
        X_l = np.asarray(self.X_l[idx], dtype=np.float32) * np.float32(self.input_scale)

        return {
            "X_h": torch.from_numpy(X_h),
            "X_l": torch.from_numpy(X_l),
            "K": torch.tensor(int(self.K_list_all[idx]), dtype=torch.long),
            "P_h": torch.tensor(max(float(self.P_h_all[idx]) * self.P_h_scale, 1e-30), dtype=torch.float32),
            "idx": torch.tensor(idx, dtype=torch.long),
        }


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


# ======================================================================
# 2. AWGN
# ======================================================================
def add_awgn_to_batch(
    X_h,
    X_l,
    P_h,
    K_true,
    snr_db,
    P_h_ref,
):
    if snr_db is None:
        return X_h, X_l

    eps = 1e-30
    B = X_h.shape[0]
    device = X_h.device
    dtype = X_h.dtype

    P_h = torch.clamp(P_h.to(device=device, dtype=dtype).view(B), min=eps)
    exist_mask = K_true.to(device=device).view(B) > 0

    P_h_ref = torch.full((B,), float(P_h_ref), device=device, dtype=dtype).clamp_min(eps)
    P_h_for_noise = torch.where(exist_mask, P_h, P_h_ref)

    snr_db = torch.as_tensor(snr_db, device=device, dtype=dtype)

    if snr_db.ndim == 0:
        snr_db = snr_db.expand(B)
    else:
        snr_db = snr_db.view(B)

    snr_linear = 10.0 ** (snr_db / 10.0)

    # Same receiver noise floor for both bands. The sampled SNR controls the
    # high-frequency band; the low-frequency band naturally gets higher SNR
    # when its received power is larger.
    sigma2 = P_h_for_noise / snr_linear
    noise_std = torch.sqrt(sigma2 / 2.0)

    noise_h = noise_std.view(B, 1, 1, 1, 1) * torch.randn_like(X_h)
    noise_l = noise_std.view(B, 1, 1, 1, 1) * torch.randn_like(X_l)

    return X_h + noise_h, X_l + noise_l


def get_train_snr_min(args):
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
    P_h = batch["P_h"].to(device, non_blocking=True)

    if train and args.train_use_awgn:
        if args.train_snr_mode == "uniform":
            snr_min = get_train_snr_min(args)
            snr_db = torch.empty(X_h.shape[0], device=device).uniform_(
                snr_min,
                float(args.train_snr_max),
            )
        elif args.train_snr_mode == "choice":
            snr_db = random.choice(args.train_snr_list)
        else:
            raise ValueError(f"Unknown train_snr_mode: {args.train_snr_mode}")

        X_h, X_l = add_awgn_to_batch(
            X_h=X_h,
            X_l=X_l,
            P_h=P_h,
            K_true=K_true,
            snr_db=snr_db,
            P_h_ref=args.P_h_ref,
        )

    if (not train) and (args.val_snr_db is not None):
        X_h, X_l = add_awgn_to_batch(
            X_h=X_h,
            X_l=X_l,
            P_h=P_h,
            K_true=K_true,
            snr_db=args.val_snr_db,
            P_h_ref=args.P_h_ref,
        )

    return X_h, X_l, K_true


# ======================================================================
# 3. Loss
# ======================================================================
def soft_ce_loss(logits, target, weight=None, label_smoothing=0.0, reduction="mean"):
    num_classes = logits.shape[1]
    smoothing = min(max(float(label_smoothing), 0.0), 1.0 - 1.0 / num_classes)
    log_prob = F.log_softmax(logits, dim=1)

    if smoothing > 0:
        target_prob = torch.full_like(log_prob, smoothing / max(num_classes - 1, 1))
        target_prob.scatter_(1, target.view(-1, 1), 1.0 - smoothing)
        loss = -target_prob * log_prob

        if weight is not None:
            loss = loss * weight.view(1, -1)

        loss = loss.sum(dim=1)
    else:
        loss = F.cross_entropy(
            logits,
            target,
            weight=weight,
            reduction="none",
        )

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss

    raise ValueError(f"Unknown reduction: {reduction}")


def focal_ce_loss(logits, target, weight=None, gamma=1.5, label_smoothing=0.0):
    ce = soft_ce_loss(
        logits,
        target,
        weight=weight,
        label_smoothing=label_smoothing,
        reduction="none",
    )

    prob = torch.softmax(logits, dim=1)
    pt = prob.gather(1, target.view(-1, 1)).squeeze(1)
    pt = pt.clamp(1e-6, 1.0 - 1e-6)

    loss = ((1.0 - pt) ** gamma) * ce
    return loss.mean()


def get_classification_loss(logits, K_true, class_weights, args):
    if args.loss_type == "ce":
        return soft_ce_loss(
            logits,
            K_true,
            weight=class_weights,
            label_smoothing=args.label_smoothing,
        )

    if args.loss_type == "focal":
        return focal_ce_loss(
            logits,
            K_true,
            weight=class_weights,
            gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
        )

    raise ValueError(f"Unknown loss_type: {args.loss_type}")


def distance_aware_count_loss(logits, K_true, class_weights, args):
    classes = torch.arange(args.Kmax + 1, device=logits.device, dtype=logits.dtype)
    K_float = K_true.to(dtype=logits.dtype)
    distance = torch.abs(classes.view(1, -1) - K_float.view(-1, 1))

    tau = max(float(args.distance_tau), 1e-6)
    target_prob = torch.softmax(-distance / tau, dim=1)

    if class_weights is not None and args.distance_use_class_weight:
        target_prob = target_prob * class_weights.to(device=logits.device, dtype=logits.dtype).view(1, -1)
        target_prob = target_prob / target_prob.sum(dim=1, keepdim=True).clamp_min(1e-8)

    log_prob = F.log_softmax(logits, dim=1)
    prob = torch.softmax(logits, dim=1)

    kl_loss = F.kl_div(log_prob, target_prob, reduction="batchmean")
    expected_K = torch.sum(prob * classes.view(1, -1), dim=1)
    mean_loss = F.smooth_l1_loss(expected_K, K_float)
    var_loss = torch.sum(prob * (classes.view(1, -1) - expected_K.view(-1, 1)) ** 2, dim=1).mean()

    total_loss = (
        kl_loss
        + args.mean_loss_weight * mean_loss
        + args.var_loss_weight * var_loss
    )

    loss_dict = {
        "cls": kl_loss.detach(),
        "group": None,
        "mean": mean_loss.detach(),
        "var": var_loss.detach(),
        "ordinal": None,
        "reg": None,
        "margin": None,
        "under": None,
        "band_aux": None,
        "k4": None,
        "high": None,
        "entropy": None,
        "total": total_loss.detach(),
    }
    return total_loss, loss_dict


def hybrid_exact_count_loss(logits, K_true, class_weights, args):
    cls_loss = get_classification_loss(logits, K_true, class_weights, args)
    prob = torch.softmax(logits, dim=1)
    classes = torch.arange(args.Kmax + 1, device=logits.device, dtype=logits.dtype)
    K_float = K_true.to(dtype=logits.dtype)

    high_mask = K_true >= 3

    if torch.any(high_mask):
        high_loss = F.cross_entropy(logits[high_mask, 3:6], K_true[high_mask] - 3)
    else:
        high_loss = logits.new_zeros(())

    distance = torch.abs(classes.view(1, -1) - K_float.view(-1, 1))
    expected_abs_error = torch.sum(prob * distance, dim=1).mean()

    total_loss = (
        cls_loss
        + args.high_boundary_loss_weight * high_loss
        + args.expected_abs_loss_weight * expected_abs_error
    )

    loss_dict = {
        "cls": cls_loss.detach(),
        "group": None,
        "mean": expected_abs_error.detach(),
        "var": None,
        "ordinal": None,
        "reg": None,
        "margin": None,
        "under": None,
        "band_aux": None,
        "k4": None,
        "high": high_loss.detach(),
        "entropy": None,
        "total": total_loss.detach(),
    }
    return total_loss, loss_dict


def ordinal_bce_loss(ord_logits, K_true, args):
    thresholds = torch.arange(1, args.Kmax + 1, device=K_true.device).view(1, -1)
    ord_target = (K_true.view(-1, 1) >= thresholds).float()
    ordinal_loss = F.binary_cross_entropy_with_logits(
        ord_logits,
        ord_target,
        reduction="none",
    )

    ord_weight = torch.tensor(
        args.ord_threshold_weights,
        dtype=ordinal_loss.dtype,
        device=ordinal_loss.device,
    ).view(1, -1)
    return (ordinal_loss * ord_weight).mean()


def factorized_exact_count_loss(out, K_true, class_weights, args):
    logits = out["K_logits"]
    cls_loss = get_classification_loss(logits, K_true, class_weights, args)

    low_logit = torch.logsumexp(logits[:, :3], dim=1)
    high_logit = torch.logsumexp(logits[:, 3:6], dim=1)
    group_logits = torch.stack([low_logit, high_logit], dim=1)
    group_target = (K_true >= 3).long()
    group_loss = F.cross_entropy(group_logits, group_target)

    high_mask = K_true >= 3
    if torch.any(high_mask):
        high_target = K_true[high_mask] - 3
        high_weight = torch.tensor(
            args.high_class_weights,
            dtype=logits.dtype,
            device=logits.device,
        )
        high_loss = F.cross_entropy(
            out["K_high_logits"][high_mask],
            high_target,
            weight=high_weight,
        )
    else:
        high_loss = logits.new_zeros(())

    ordinal_loss = ordinal_bce_loss(out["K_ord_logits"], K_true, args)
    margin_loss = adjacent_margin_loss(logits, K_true, args)
    under_loss = undercount_margin_loss(logits, K_true, args)

    band_aux_loss = logits.new_zeros(())
    band_aux_count = 0

    if args.band_aux_loss_weight > 0:
        if "K_band_h_logits" in out:
            band_aux_loss = band_aux_loss + get_classification_loss(
                out["K_band_h_logits"],
                K_true,
                class_weights,
                args,
            )
            band_aux_count += 1

        if "K_band_l_logits" in out:
            band_aux_loss = band_aux_loss + get_classification_loss(
                out["K_band_l_logits"],
                K_true,
                class_weights,
                args,
            )
            band_aux_count += 1

        if band_aux_count > 0:
            band_aux_loss = band_aux_loss / float(band_aux_count)

    k4_loss = logits.new_zeros(())
    if args.k4_aux_loss_weight > 0 and "K4_logit" in out:
        k4_mask = K_true >= 3

        if torch.any(k4_mask):
            k4_target = (K_true[k4_mask] == 4).to(dtype=logits.dtype)
            pos_weight = torch.tensor(
                float(args.k4_pos_weight),
                dtype=logits.dtype,
                device=logits.device,
            )
            k4_loss = F.binary_cross_entropy_with_logits(
                out["K4_logit"][k4_mask],
                k4_target,
                pos_weight=pos_weight,
            )

    prob = torch.softmax(logits, dim=1)
    classes = torch.arange(args.Kmax + 1, device=logits.device, dtype=logits.dtype)
    distance = torch.abs(classes.view(1, -1) - K_true.to(logits.dtype).view(-1, 1))
    expected_abs_error = torch.sum(prob * distance, dim=1).mean()

    total_loss = (
        args.ce_loss_weight * cls_loss
        + args.group_loss_weight * group_loss
        + args.high_count_loss_weight * high_loss
        + args.ordinal_loss_weight * ordinal_loss
        + args.margin_loss_weight * margin_loss
        + args.undercount_margin_loss_weight * under_loss
        + args.band_aux_loss_weight * band_aux_loss
        + args.k4_aux_loss_weight * k4_loss
        + args.expected_abs_loss_weight * expected_abs_error
    )

    loss_dict = {
        "cls": cls_loss.detach(),
        "group": group_loss.detach(),
        "mean": expected_abs_error.detach(),
        "var": None,
        "ordinal": ordinal_loss.detach(),
        "reg": None,
        "margin": margin_loss.detach(),
        "under": under_loss.detach(),
        "band_aux": band_aux_loss.detach(),
        "k4": k4_loss.detach(),
        "high": high_loss.detach(),
        "entropy": None,
        "total": total_loss.detach(),
    }
    return total_loss, loss_dict


def adjacent_margin_loss(logits, K_true, args):
    margin = float(args.adjacent_margin)
    losses = []

    for k in range(3, args.Kmax + 1):
        mask = K_true == k

        if not torch.any(mask):
            continue

        true_logit = logits[mask, k]

        if k - 1 >= 0:
            losses.append(F.relu(margin - (true_logit - logits[mask, k - 1])))

        if k + 1 <= args.Kmax:
            losses.append(F.relu(margin - (true_logit - logits[mask, k + 1])))

    if len(losses) == 0:
        return logits.new_zeros(())

    return torch.cat(losses).mean()


def undercount_margin_loss(logits, K_true, args):
    margin = float(args.undercount_margin)
    class_weights = torch.tensor(
        args.undercount_margin_class_weights,
        dtype=logits.dtype,
        device=logits.device,
    )
    losses = []

    for k in range(1, args.Kmax + 1):
        mask = K_true == k

        if not torch.any(mask):
            continue

        true_logit = logits[mask, k]
        lower_logit = logits[mask, k - 1]
        weight = class_weights[k]
        losses.append(weight * F.relu(margin - (true_logit - lower_logit)))

    if len(losses) == 0:
        return logits.new_zeros(())

    return torch.cat(losses).mean()


def get_count_loss(out, K_true, class_weights, args):
    logits = out["K_logits"]

    if args.count_loss_type == "factorized_exact":
        return factorized_exact_count_loss(out, K_true, class_weights, args)

    if args.count_loss_type == "distance_aware":
        return distance_aware_count_loss(logits, K_true, class_weights, args)

    if args.count_loss_type == "hybrid_exact":
        return hybrid_exact_count_loss(logits, K_true, class_weights, args)

    if args.count_loss_type != "legacy":
        raise ValueError(f"Unknown count_loss_type: {args.count_loss_type}")

    cls_loss = get_classification_loss(logits, K_true, class_weights, args)

    total_loss = args.ce_loss_weight * cls_loss
    loss_dict = {
        "cls": cls_loss.detach(),
        "group": None,
        "ordinal": None,
        "reg": None,
        "margin": None,
        "under": None,
        "band_aux": None,
        "k4": None,
        "high": None,
        "entropy": None,
    }

    if args.ordinal_loss_weight > 0:
        ord_logits = out["K_ord_logits"]
        ordinal_loss = ordinal_bce_loss(ord_logits, K_true, args)
        total_loss = total_loss + args.ordinal_loss_weight * ordinal_loss
        loss_dict["ordinal"] = ordinal_loss.detach()

    if args.count_reg_loss_weight > 0:
        count_pred = torch.sigmoid(out["K_count_logit"])
        count_target = K_true.float() / float(args.Kmax)
        reg_loss = F.smooth_l1_loss(count_pred, count_target)
        total_loss = total_loss + args.count_reg_loss_weight * reg_loss
        loss_dict["reg"] = reg_loss.detach()

    if args.margin_loss_weight > 0:
        margin_loss = adjacent_margin_loss(logits, K_true, args)
        total_loss = total_loss + args.margin_loss_weight * margin_loss
        loss_dict["margin"] = margin_loss.detach()

    if args.undercount_margin_loss_weight > 0:
        under_loss = undercount_margin_loss(logits, K_true, args)
        total_loss = total_loss + args.undercount_margin_loss_weight * under_loss
        loss_dict["under"] = under_loss.detach()

    if args.high_count_loss_weight > 0 and "K_high_logits" in out:
        high_mask = K_true >= 3

        if torch.any(high_mask):
            high_target = K_true[high_mask] - 3
            high_loss = F.cross_entropy(out["K_high_logits"][high_mask], high_target)
        else:
            high_loss = logits.new_zeros(())

        total_loss = total_loss + args.high_count_loss_weight * high_loss
        loss_dict["high"] = high_loss.detach()

    if args.confidence_penalty_weight > 0:
        prob = torch.softmax(logits, dim=1)
        entropy = -(prob * torch.log(prob.clamp_min(1e-8))).sum(dim=1).mean()
        total_loss = total_loss - args.confidence_penalty_weight * entropy
        loss_dict["entropy"] = entropy.detach()

    loss_dict["total"] = total_loss.detach()
    return total_loss, loss_dict


def ordinal_logits_to_class_log_prob(ord_logits):
    q = torch.sigmoid(ord_logits)
    q = torch.cummin(q, dim=1).values

    p0 = 1.0 - q[:, :1]
    pmid = q[:, :-1] - q[:, 1:]
    plast = q[:, -1:]
    class_prob = torch.cat([p0, pmid, plast], dim=1)
    class_prob = class_prob.clamp_min(1e-6)
    class_prob = class_prob / class_prob.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return torch.log(class_prob)


def get_refined_logits(out, args):
    logits = out["K_logits"]

    if (
        args.ordinal_refine_logit_weight > 0
        and "K_ord_logits" in out
        and out["K_ord_logits"].shape[1] + 1 == logits.shape[1]
    ):
        logits = logits + args.ordinal_refine_logit_weight * ordinal_logits_to_class_log_prob(
            out["K_ord_logits"]
        )

    refined_logits = logits.clone()
    prob = torch.softmax(logits, dim=1)
    high_prob = prob[:, 3:6].sum(dim=1) if logits.shape[1] >= 6 else prob.new_zeros(prob.shape[0])
    high_gate = (torch.argmax(logits, dim=1) >= 3) | (high_prob >= args.high_refine_min_high_prob)

    if (
        args.high_refine_logit_weight > 0
        and "K_high_logits" in out
        and logits.shape[1] >= 6
        and torch.any(high_gate)
    ):
        refined_logits[high_gate, 3:6] = (
            refined_logits[high_gate, 3:6]
            + args.high_refine_logit_weight * out["K_high_logits"][high_gate]
        )

    if (
        args.k4_refine_logit_weight > 0
        and "K4_logit" in out
        and logits.shape[1] >= 5
        and torch.any(high_gate)
    ):
        refined_logits[high_gate, 4] = (
            refined_logits[high_gate, 4]
            + args.k4_refine_logit_weight * out["K4_logit"][high_gate]
        )

    return refined_logits


# ======================================================================
# 4. Metrics
# ======================================================================
def make_confusion(Kmax, device=None):
    if device is None:
        return np.zeros((Kmax + 1, Kmax + 1), dtype=np.int64)

    return torch.zeros((Kmax + 1, Kmax + 1), dtype=torch.long, device=device)


def update_confusion(conf, K_true, K_pred):
    if torch.is_tensor(conf):
        Kmax = conf.shape[0] - 1
        K_true = K_true.clamp(0, Kmax).long()
        K_pred = K_pred.clamp(0, Kmax).long()
        idx = K_true * (Kmax + 1) + K_pred
        counts = torch.bincount(idx, minlength=(Kmax + 1) ** 2)
        conf += counts.view(Kmax + 1, Kmax + 1)
        return

    K_true = K_true.detach().cpu().numpy().astype(np.int64)
    K_pred = K_pred.detach().cpu().numpy().astype(np.int64)

    Kmax = conf.shape[0] - 1

    for t, p in zip(K_true, K_pred):
        t = int(np.clip(t, 0, Kmax))
        p = int(np.clip(p, 0, Kmax))
        conf[t, p] += 1


def confusion_to_metrics(conf):
    if torch.is_tensor(conf):
        conf = conf.detach().cpu().numpy()

    Kmax = conf.shape[0] - 1

    total = conf.sum()
    correct = np.trace(conf)

    acc = correct / max(total, 1)

    off1 = 0
    for i in range(Kmax + 1):
        for j in range(Kmax + 1):
            if abs(i - j) <= 1:
                off1 += conf[i, j]

    off1_acc = off1 / max(total, 1)

    perK_acc = []
    perK_num = []

    for k in range(Kmax + 1):
        n = conf[k].sum()
        perK_num.append(int(n))

        if n > 0:
            perK_acc.append(conf[k, k] / n)
        else:
            perK_acc.append(np.nan)

    valid_acc = [x for x in perK_acc if np.isfinite(x)]
    macro_acc = float(np.mean(valid_acc)) if len(valid_acc) > 0 else 0.0

    valid_positive_acc = [max(float(x), 1e-12) for x in perK_acc if np.isfinite(x) and x > 0]
    harmonic_acc = (
        len(valid_positive_acc) / float(np.sum([1.0 / x for x in valid_positive_acc]))
        if len(valid_positive_acc) > 0
        else 0.0
    )

    high_start = min(3, Kmax)
    high_total = conf[high_start:, :].sum()
    high_correct = np.trace(conf[high_start:, high_start:])
    high_count_acc = high_correct / max(high_total, 1)

    def row_rate(src, dst):
        if src > Kmax or dst > Kmax:
            return np.nan

        row_total = conf[src].sum()

        if row_total <= 0:
            return np.nan

        return conf[src, dst] / row_total

    mae = 0.0
    for i in range(Kmax + 1):
        for j in range(Kmax + 1):
            mae += abs(i - j) * conf[i, j]

    mae = mae / max(total, 1)

    return {
        "acc": float(acc),
        "macro_acc": float(macro_acc),
        "harmonic_acc": float(harmonic_acc),
        "high_count_acc": float(high_count_acc),
        "K3_to_K4": float(row_rate(3, 4)) if np.isfinite(row_rate(3, 4)) else np.nan,
        "K4_to_K3": float(row_rate(4, 3)) if np.isfinite(row_rate(4, 3)) else np.nan,
        "K4_to_K5": float(row_rate(4, 5)) if np.isfinite(row_rate(4, 5)) else np.nan,
        "K5_to_K4": float(row_rate(5, 4)) if np.isfinite(row_rate(5, 4)) else np.nan,
        "off1_acc": float(off1_acc),
        "mae": float(mae),
        "perK_acc": perK_acc,
        "perK_num": perK_num,
        "conf": conf,
    }


def print_metrics(title, metrics, avg_loss, args):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    print(
        f"loss={avg_loss:.5f} | "
        f"Acc={metrics['acc']:.4f} | "
        f"MacroAcc={metrics['macro_acc']:.4f} | "
        f"HighAcc={metrics['high_count_acc']:.4f} | "
        f"K±1={metrics['off1_acc']:.4f} | "
        f"MAE_K={metrics['mae']:.4f}"
    )

    if "band_h_acc" in metrics and "band_l_acc" in metrics:
        print(
            f"BandAux | "
            f"H_acc={metrics['band_h_acc']:.4f}, "
            f"H_macro={metrics['band_h_macro_acc']:.4f}, "
            f"H_high={metrics['band_h_high_count_acc']:.4f} | "
            f"L_acc={metrics['band_l_acc']:.4f}, "
            f"L_macro={metrics['band_l_macro_acc']:.4f}, "
            f"L_high={metrics['band_l_high_count_acc']:.4f}"
        )

    print("\nPer-K accuracy:")
    print(f"{'K':>2s} | {'samples':>7s} | {'Kacc':>8s}")
    print("-" * 28)

    for k in range(args.Kmax + 1):
        n = metrics["perK_num"][k]
        a = metrics["perK_acc"][k]

        if np.isfinite(a):
            print(f"{k:2d} | {n:7d} | {a:8.4f}")
        else:
            print(f"{k:2d} | {n:7d} | {'nan':>8s}")

    print("\nConfusion matrix: rows=true K, cols=pred K")
    print(metrics["conf"])
    print("=" * 80 + "\n")


# ======================================================================
# 5. Train / Eval
# ======================================================================
def tensor_to_float(value):
    if value is None:
        return 0.0
    if torch.is_tensor(value):
        return float(value.item())
    return float(value)


def autocast_context(device, enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)

    return torch.cuda.amp.autocast(enabled=enabled)


def make_grad_scaler(enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)

    return torch.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, ema_model, device, args, class_weights, epoch, global_step):
    model.train()

    amp_enabled = bool(args.use_amp and device.type == "cuda")
    total_loss = torch.zeros((), device=device)
    total_samples = 0
    conf = make_confusion(args.Kmax, device=device)

    for batch_idx, batch in enumerate(loader):
        X_h, X_l, K_true = prepare_batch(
            batch=batch,
            device=device,
            args=args,
            train=True,
        )

        optimizer.zero_grad(set_to_none=True)

        with autocast_context(device, amp_enabled):
            out = model(X_h, X_l)
            loss, loss_dict = get_count_loss(
                out=out,
                K_true=K_true,
                class_weights=class_weights,
                args=args,
            )
        logits = get_refined_logits(out, args)

        scale_before = scaler.get_scale()
        scaler.scale(loss).backward()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()
        optimizer_stepped = (not amp_enabled) or (scale_after >= scale_before)

        if optimizer_stepped and ema_model is not None:
            update_ema_model(ema_model, model, args.ema_decay)

        if optimizer_stepped and scheduler is not None:
            scheduler.step()

        loss_detached = loss.detach()
        batch_size = int(K_true.size(0))

        total_loss += loss_detached * batch_size
        total_samples += batch_size

        with torch.no_grad():
            pred_logits = get_refined_logits(out, args)
            K_pred = torch.argmax(pred_logits, dim=1)
            update_confusion(conf, K_true, K_pred)

        global_step += 1
        should_log = args.use_swanlab and global_step % max(int(args.swanlab_log_every_batches), 1) == 0
        should_print = (batch_idx + 1) % args.print_every == 0
        metrics = None

        if should_log:
            metrics = confusion_to_metrics(conf)
            log_dict = {
                "batch": global_step,
                "epoch": epoch,

                "train/loss": tensor_to_float(loss_detached),
                "train/loss_cls": tensor_to_float(loss_dict["cls"]),
                "train/loss_group": tensor_to_float(loss_dict.get("group")),
                "train/loss_mean": tensor_to_float(loss_dict.get("mean")),
                "train/loss_var": tensor_to_float(loss_dict.get("var")),
                "train/loss_ordinal": tensor_to_float(loss_dict["ordinal"]),
                "train/loss_reg": tensor_to_float(loss_dict["reg"]),
                "train/loss_margin": tensor_to_float(loss_dict["margin"]),
                "train/loss_under": tensor_to_float(loss_dict.get("under")),
                "train/loss_band_aux": tensor_to_float(loss_dict.get("band_aux")),
                "train/loss_k4": tensor_to_float(loss_dict.get("k4")),
                "train/loss_high": tensor_to_float(loss_dict["high"]),
                "train/entropy": tensor_to_float(loss_dict["entropy"]),
                "train/loss_running": tensor_to_float(total_loss / max(total_samples, 1)),
                "train/acc": metrics["acc"],
                "train/macro_acc": metrics["macro_acc"],
                "train/high_count_acc": metrics["high_count_acc"],
                "train/harmonic_acc": metrics["harmonic_acc"],
                "train/off1_acc": metrics["off1_acc"],
                "train/mae_K": metrics["mae"],

                "lr": get_current_lr(optimizer),
            }

            for k in range(args.Kmax + 1):
                log_dict[f"train/K{k}_acc"] = metrics["perK_acc"][k]

            log_swanlab(log_dict, step=global_step)

        if should_print:
            if metrics is None:
                metrics = confusion_to_metrics(conf)

            print(
                f"batch {batch_idx+1:04d}/{len(loader):04d} | "
                f"loss={tensor_to_float(loss_detached):.5f} | "
                f"lr={get_current_lr(optimizer):.3e} | "
                f"acc={metrics['acc']:.4f} | "
                f"macro={metrics['macro_acc']:.4f} | "
                f"K±1={metrics['off1_acc']:.4f}"
            )

    avg_loss = tensor_to_float(total_loss / max(total_samples, 1))
    metrics = confusion_to_metrics(conf)

    return avg_loss, metrics, global_step


@torch.no_grad()
def evaluate(model, loader, device, args, class_weights):
    model.eval()

    amp_enabled = bool(args.use_amp and device.type == "cuda")
    total_loss = torch.zeros((), device=device)
    total_samples = 0
    conf = make_confusion(args.Kmax, device=device)
    conf_band_h = make_confusion(args.Kmax, device=device)
    conf_band_l = make_confusion(args.Kmax, device=device)

    for batch in loader:
        X_h, X_l, K_true = prepare_batch(
            batch=batch,
            device=device,
            args=args,
            train=False,
        )

        with autocast_context(device, amp_enabled):
            out = model(X_h, X_l)
            loss, _ = get_count_loss(
                out=out,
                K_true=K_true,
                class_weights=class_weights,
                args=args,
            )
        logits = get_refined_logits(out, args)

        batch_size = int(K_true.size(0))
        total_loss += loss.detach() * batch_size
        total_samples += batch_size

        K_pred = torch.argmax(logits, dim=1)
        update_confusion(conf, K_true, K_pred)

        if "K_band_h_logits" in out:
            K_pred_h = torch.argmax(out["K_band_h_logits"], dim=1)
            update_confusion(conf_band_h, K_true, K_pred_h)

        if "K_band_l_logits" in out:
            K_pred_l = torch.argmax(out["K_band_l_logits"], dim=1)
            update_confusion(conf_band_l, K_true, K_pred_l)

    avg_loss = tensor_to_float(total_loss / max(total_samples, 1))
    metrics = confusion_to_metrics(conf)
    band_h_metrics = confusion_to_metrics(conf_band_h)
    band_l_metrics = confusion_to_metrics(conf_band_l)

    metrics["band_h_acc"] = band_h_metrics["acc"]
    metrics["band_h_macro_acc"] = band_h_metrics["macro_acc"]
    metrics["band_h_high_count_acc"] = band_h_metrics["high_count_acc"]
    metrics["band_l_acc"] = band_l_metrics["acc"]
    metrics["band_l_macro_acc"] = band_l_metrics["macro_acc"]
    metrics["band_l_high_count_acc"] = band_l_metrics["high_count_acc"]

    return avg_loss, metrics


# ======================================================================
# 6. Utils
# ======================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(dataset, batch_size, shuffle, num_workers, drop_last, seed, prefetch_factor):
    g = torch.Generator()
    g.manual_seed(seed)

    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=g,
    )

    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(dataset, **kwargs)


def get_current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def build_batch_warmup_cosine_scheduler(optimizer, args, steps_per_epoch):
    total_steps = max(int(args.epochs) * int(steps_per_epoch), 1)
    warmup_steps = max(int(args.lr_warmup_batches), 0)
    warmup_steps = min(warmup_steps, total_steps)

    if args.lr <= 0:
        raise ValueError("args.lr should be > 0 when using lr scheduler")

    min_lr_ratio = float(args.min_lr) / float(args.lr)
    min_lr_ratio = min(max(min_lr_ratio, 0.0), 1.0)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        if total_steps <= warmup_steps:
            return 1.0

        progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_history_csv(history, path):
    keys = list(history[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()

        for row in history:
            writer.writerow(row)

def args_to_swanlab_config(args):
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
    log_dict = {
        k: v
        for k, v in log_dict.items()
        if not isinstance(v, (float, np.floating)) or math.isfinite(float(v))
    }

    if step is None:
        swanlab.log(log_dict)
        return

    try:
        swanlab.log(log_dict, step=step)
    except TypeError:
        log_dict = dict(log_dict)
        log_dict.setdefault("batch", step)
        swanlab.log(log_dict)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_metric, args):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "args": {k: getattr(args, k) for k in dir(args) if not k.startswith("_") and not callable(getattr(args, k))},
    }

    torch.save(ckpt, path)


def build_class_weights(args, device):
    if args.use_class_weight:
        w = torch.tensor(args.class_weights, dtype=torch.float32, device=device)

        if w.numel() != args.Kmax + 1:
            raise ValueError(f"class_weights length should be {args.Kmax + 1}")

        print("Use class weights:", w.detach().cpu().numpy())
        return w

    return None


def build_model(args, device):
    model = DualBandCrossFusionCountNet(
        Kmax=args.Kmax,
        spec_size=args.spec_size,
        token_grid=args.token_grid,
        use_high=args.use_high,
        use_low=args.use_low,
        base_ch=args.base_ch,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        cross_attn_layers=args.cross_attn_layers,
        fusion_attn_layers=args.fusion_attn_layers,
        mlp_ratio=args.mlp_ratio,
        attn_dropout=args.attn_dropout,
        token_dropout=args.token_dropout,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    return model


def build_ema_model(model):
    ema_model = copy.deepcopy(model).eval()

    for p in ema_model.parameters():
        p.requires_grad_(False)

    return ema_model


@torch.no_grad()
def update_ema_model(ema_model, model, decay):
    ema_state = ema_model.state_dict()
    model_state = model.state_dict()

    for name, ema_value in ema_state.items():
        model_value = model_state[name]

        if torch.is_floating_point(ema_value):
            ema_value.mul_(decay).add_(model_value.detach(), alpha=1.0 - decay)
        else:
            ema_value.copy_(model_value)


def set_cross_attention_trainable(model, trainable):
    if not hasattr(model, "cross_blocks"):
        return

    for p in model.cross_blocks.parameters():
        p.requires_grad_(trainable)


@torch.no_grad()
def shape_check(model, batch, device):
    model.eval()

    X_h = batch["X_h"].to(device)
    X_l = batch["X_l"].to(device)

    out = model(X_h, X_l)

    print("\n========== Shape Check ==========")
    print("X_h:", tuple(X_h.shape))
    print("X_l:", tuple(X_l.shape))
    print("K_logits:", tuple(out["K_logits"].shape))
    print("K_band_h_logits:", tuple(out["K_band_h_logits"].shape))
    print("K_band_l_logits:", tuple(out["K_band_l_logits"].shape))
    print("K_ord_logits:", tuple(out["K_ord_logits"].shape))
    print("K_count_logit:", tuple(out["K_count_logit"].shape))
    print("K_high_logits:", tuple(out["K_high_logits"].shape))
    print("K4_logit:", tuple(out["K4_logit"].shape))
    print("=================================\n")


# ======================================================================
# 7. Main
# ======================================================================
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
        swanlab.init(
            project=args.swanlab_project,
            experiment_name=args.swanlab_experiment_name,
            config=args_to_swanlab_config(args),
            mode="cloud",
        )

    full_set = SionnaCountMemmapDataset(
        npz_path=args.train_npz,
        cache_dir=args.cache_dir,
        convert_chunk_rows=args.convert_chunk_rows,
        target_p_h_ref=args.sionna_target_p_h_ref,
    )
    train_set, val_set, train_indices, val_indices = make_random_split(
        dataset=full_set,
        train_ratio=args.train_ratio,
        seed=args.split_seed,
    )

    print_split_distribution(full_set, train_indices, "Train split")
    print_split_distribution(full_set, val_indices, "Val split")

    args.P_h_ref = full_set.P_h_ref

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
    ema_model = build_ema_model(model) if args.use_ema else None

    first_batch = next(iter(train_loader))
    shape_check(model, first_batch, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = make_grad_scaler(enabled=bool(args.use_amp and device.type == "cuda"))
    scheduler = build_batch_warmup_cosine_scheduler(
        optimizer=optimizer,
        args=args,
        steps_per_epoch=len(train_loader),
    )

    print(
        f"LR scheduler: batch warmup + cosine | "
        f"warmup_batches={args.lr_warmup_batches}, "
        f"total_batches={args.epochs * len(train_loader)}, "
        f"base_lr={args.lr:.3e}, min_lr={args.min_lr:.3e}"
    )

    class_weights = build_class_weights(args, device)

    best_macro = -1.0
    best_acc = -1.0
    best_high = -1.0

    best_macro_path = os.path.join(args.save_dir, "crossfusion_countnet_best_macro.pt")
    best_acc_path = os.path.join(args.save_dir, "crossfusion_countnet_best_acc.pt")
    best_high_path = os.path.join(args.save_dir, "crossfusion_countnet_best_high.pt")
    last_path = os.path.join(args.save_dir, "crossfusion_countnet_last.pt")
    csv_path = os.path.join(args.save_dir, "crossfusion_count_train_history.csv")

    history = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        args.current_epoch = epoch
        cross_trainable = epoch > int(args.freeze_cross_attn_epochs)
        set_cross_attention_trainable(model, cross_trainable)
        snr_min_now = get_train_snr_min(args)

        print(
            f"\n========== Epoch {epoch}/{args.epochs} | "
            f"lr={get_current_lr(optimizer):.3e} | "
            f"train_snr=[{snr_min_now:.1f},{args.train_snr_max:.1f}] | "
            f"cross_attn_trainable={cross_trainable} =========="
        )

        train_loss, train_metrics, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            ema_model=ema_model,
            device=device,
            args=args,
            class_weights=class_weights,
            epoch=epoch,
            global_step=global_step,
        )

        eval_model = ema_model if ema_model is not None else model

        val_loss, val_metrics = evaluate(
            model=eval_model,
            loader=val_loader,
            device=device,
            args=args,
            class_weights=class_weights,
        )

        print_metrics(
            title=f"TRAIN | epoch={epoch}",
            metrics=train_metrics,
            avg_loss=train_loss,
            args=args,
        )

        print_metrics(
            title=f"VAL | epoch={epoch}",
            metrics=val_metrics,
            avg_loss=val_loss,
            args=args,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_metrics["acc"],
            "train_macro_acc": train_metrics["macro_acc"],
            "train_harmonic_acc": train_metrics["harmonic_acc"],
            "train_high_count_acc": train_metrics["high_count_acc"],
            "train_off1_acc": train_metrics["off1_acc"],
            "train_mae": train_metrics["mae"],
            "lr": get_current_lr(optimizer),

            "val_loss": val_loss,
            "val_acc": val_metrics["acc"],
            "val_macro_acc": val_metrics["macro_acc"],
            "val_harmonic_acc": val_metrics["harmonic_acc"],
            "val_high_count_acc": val_metrics["high_count_acc"],
            "val_band_h_acc": val_metrics.get("band_h_acc", np.nan),
            "val_band_h_macro_acc": val_metrics.get("band_h_macro_acc", np.nan),
            "val_band_h_high_count_acc": val_metrics.get("band_h_high_count_acc", np.nan),
            "val_band_l_acc": val_metrics.get("band_l_acc", np.nan),
            "val_band_l_macro_acc": val_metrics.get("band_l_macro_acc", np.nan),
            "val_band_l_high_count_acc": val_metrics.get("band_l_high_count_acc", np.nan),
            "val_K3_to_K4": val_metrics["K3_to_K4"],
            "val_K4_to_K3": val_metrics["K4_to_K3"],
            "val_K4_to_K5": val_metrics["K4_to_K5"],
            "val_K5_to_K4": val_metrics["K5_to_K4"],
            "val_off1_acc": val_metrics["off1_acc"],
            "val_mae": val_metrics["mae"],
        })

        for k in range(args.Kmax + 1):
            history[-1][f"train_K{k}_acc"] = train_metrics["perK_acc"][k]
            history[-1][f"val_K{k}_acc"] = val_metrics["perK_acc"][k]

        save_history_csv(history, csv_path)

        if args.use_swanlab:
            log_dict = {
                "batch": global_step,
                "epoch": epoch,

                "val/loss": val_loss,
                "val/acc": val_metrics["acc"],
                "val/macro_acc": val_metrics["macro_acc"],
                "val/harmonic_acc": val_metrics["harmonic_acc"],
                "val/high_count_acc": val_metrics["high_count_acc"],
                "val_band_h/acc": val_metrics.get("band_h_acc", np.nan),
                "val_band_h/macro_acc": val_metrics.get("band_h_macro_acc", np.nan),
                "val_band_h/high_count_acc": val_metrics.get("band_h_high_count_acc", np.nan),
                "val_band_l/acc": val_metrics.get("band_l_acc", np.nan),
                "val_band_l/macro_acc": val_metrics.get("band_l_macro_acc", np.nan),
                "val_band_l/high_count_acc": val_metrics.get("band_l_high_count_acc", np.nan),
                "val/K3_to_K4": val_metrics["K3_to_K4"],
                "val/K4_to_K3": val_metrics["K4_to_K3"],
                "val/K4_to_K5": val_metrics["K4_to_K5"],
                "val/K5_to_K4": val_metrics["K5_to_K4"],
                "val/off1_acc": val_metrics["off1_acc"],
                "val/mae_K": val_metrics["mae"],

                "lr": get_current_lr(optimizer),
            }

            for k in range(args.Kmax + 1):
                log_dict[f"val/K{k}_acc"] = val_metrics["perK_acc"][k]

            log_swanlab(log_dict, step=global_step)

        save_checkpoint(
            last_path,
            model=eval_model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_metric=val_metrics["macro_acc"],
            args=args,
        )

        if val_metrics["macro_acc"] > best_macro:
            best_macro = val_metrics["macro_acc"]

            save_checkpoint(
                best_macro_path,
                model=eval_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_macro,
                args=args,
            )

            print(f"Saved best macro model: epoch={epoch}, macro_acc={best_macro:.5f}")

        if val_metrics["high_count_acc"] > best_high:
            best_high = val_metrics["high_count_acc"]

            save_checkpoint(
                best_high_path,
                model=eval_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_high,
                args=args,
            )

            print(f"Saved best high-count model: epoch={epoch}, high_acc={best_high:.5f}")

        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]

            save_checkpoint(
                best_acc_path,
                model=eval_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_acc,
                args=args,
            )

            print(f"Saved best acc model: epoch={epoch}, acc={best_acc:.5f}")

    save_history_csv(history, csv_path)

    print("\n========== Training Finished ==========")
    print(f"Best macro acc: {best_macro:.5f}")
    print(f"Best acc      : {best_acc:.5f}")
    print(f"Best high acc : {best_high:.5f}")
    print("History saved to:", csv_path)
    print("Best macro ckpt:", best_macro_path)
    print("Best acc ckpt  :", best_acc_path)
    print("Best high ckpt :", best_high_path)

    if args.use_swanlab:
        swanlab.finish()


# ======================================================================
# 8. Args
# ======================================================================
if __name__ == "__main__":
    class Args:
        # ============================================================
        # Dataset
        # ============================================================
        train_mat = ""
        test_mat = ""

        save_dir = os.path.join(RELEASE_ROOT, "outputs", "countnet")

        # Sionna NPZ dataset override. The old mat paths above are kept only
        # because this file was copied from the best CountNet script.
        train_npz = os.environ.get("SIONNA_TRAIN_NPZ", "")
        cache_dir = os.path.join(RELEASE_ROOT, "cache", "countnet")
        convert_chunk_rows = 4
        sionna_target_p_h_ref = 1.0
        train_ratio = 0.8
        split_seed = 2026
        save_dir = os.path.join(RELEASE_ROOT, "outputs", "countnet")

        save_dir = os.path.join(RELEASE_ROOT, "outputs", "countnet")

        # Final override for the new clean Sionna dataset. Keep this block
        # after the copied legacy paths above so it is the effective config.
        train_npz = os.environ.get("SIONNA_TRAIN_NPZ", "")
        cache_dir = os.path.join(RELEASE_ROOT, "cache", "countnet")
        save_dir = os.path.join(RELEASE_ROOT, "outputs", "countnet")

        # Keep archived best training outputs inside this folder.
        save_dir = os.path.join(RELEASE_ROOT, "outputs", "countnet")

        # ============================================================
        # Task
        # ============================================================
        Kmax = 5

        # ============================================================
        # Model
        # ============================================================
        use_high = True
        use_low = True

        spec_size = (32, 32, 32)
        token_grid = (8, 8, 8)

        base_ch = 32
        embed_dim = 128
        num_heads = 4
        cross_attn_layers = 1
        fusion_attn_layers = 0
        mlp_ratio = 2.0
        attn_dropout = 0.10
        token_dropout = 0.10
        hidden_dim = 256
        dropout = 0.25

        # ============================================================
        # Training
        # ============================================================
        epochs = 100
        batch_size = 32
        test_batch_size = 32

        lr = 8e-5
        min_lr = 1e-6
        lr_warmup_batches = 500
        weight_decay = 1e-3
        grad_clip = 5.0
        use_amp = True
        use_ema = True
        ema_decay = 0.999

        # ce / focal
        count_loss_type = "factorized_exact"
        distance_tau = 0.45
        distance_use_class_weight = True
        mean_loss_weight = 0.20
        var_loss_weight = 0.02
        group_loss_weight = 0.0
        high_boundary_loss_weight = 0.60
        expected_abs_loss_weight = 0.0

        loss_type = "ce"
        focal_gamma = 1.5
        label_smoothing = 0.02
        confidence_penalty_weight = 0.0
        ce_loss_weight = 1.0
        ordinal_loss_weight = 0.0
        count_reg_loss_weight = 0.0
        margin_loss_weight = 0.0
        adjacent_margin = 0.45
        undercount_margin_loss_weight = 0.0
        undercount_margin = 0.55
        undercount_margin_class_weights = [0.0, 0.0, 0.0, 0.25, 1.20, 1.50]
        band_aux_loss_weight = 0.25
        k4_aux_loss_weight = 0.0
        k4_pos_weight = 2.0
        high_count_loss_weight = 0.60
        high_refine_logit_weight = 0.45
        k4_refine_logit_weight = 0.0
        ordinal_refine_logit_weight = 0.0
        high_refine_min_high_prob = 0.40
        ord_threshold_weights = [0.25, 0.50, 1.00, 1.75, 2.50]
        high_class_weights = [1.0, 1.25, 1.15]

        # 建议先打开，高 K 稍微加权
        use_class_weight = True

        # K=0,1,2,3,4,5
        class_weights = [0.70, 0.80, 1.00, 1.20, 1.45, 1.30]

        # ============================================================
        # AWGN
        # ============================================================
        # 第一轮建议 False，先看 clean count 能不能学好
        train_use_awgn = True

        # uniform: sample one continuous SNR per sample from [train_snr_min, train_snr_max]
        # choice: sample one SNR per batch from train_snr_list
        train_snr_mode = "uniform"
        train_snr_min = -10
        train_snr_max = 20
        use_snr_curriculum = True
        snr_curriculum_start_min = 15
        snr_curriculum_epochs = 30

        # train_snr_mode="choice" 时才使用
        train_snr_list = [-10, -5, 0, 5, 10, 15, 20]

        # 验证时 None 表示 clean
        val_snr_db = 0

        # ============================================================
        # Misc
        # ============================================================
        seed = 2026
        cpu = False
        num_workers = 8
        prefetch_factor = 4
        print_every = 100
        swanlab_log_every_batches = 20
        freeze_cross_attn_epochs = 5

        P_h_ref = None

        # ============================================================
        # SwanLab
        # ============================================================
        use_swanlab = True
        swanlab_project = "countnet-isac"
        swanlab_experiment_name = "countnet_crossfusion_sionna_beiyou_train_clean_50000_scaled"

    args = Args()
    main(args)

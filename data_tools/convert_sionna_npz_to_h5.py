import argparse
import math
import zipfile
from pathlib import Path

import h5py
import numpy as np
from numpy.lib import format as npformat


def read_npy_header_from_zip(zf, member):
    with zf.open(member, "r") as f:
        version = npformat.read_magic(f)
        shape, fortran_order, dtype = npformat._read_array_header(f, version)
    return shape, fortran_order, dtype


def read_exact(stream, nbytes):
    chunks = []
    remaining = int(nbytes)
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(f"Unexpected EOF while reading {nbytes} bytes from npz member")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def make_bins_from_phys(y_phys, cfg):
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


def make_delta_phys_from_phys(y_phys, y_bin, cfg):
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


def build_labels(npz_path, cfg):
    with np.load(npz_path, allow_pickle=False) as z:
        k_list = np.asarray(z["K_list"], dtype=np.int64).reshape(-1)
        if "Y_rva" in z:
            y_rva = np.asarray(z["Y_rva"], dtype=np.float32)[:, :, :3]
        elif "Y" in z:
            y_rva = np.asarray(z["Y"], dtype=np.float32)[:, :, :3]
        else:
            raise KeyError("NPZ must contain Y_rva or Y")

        target_mask = np.zeros(y_rva.shape[:2], dtype=np.float32)
        if "target_mask" in z:
            target_mask[:] = np.asarray(z["target_mask"], dtype=np.float32)
        else:
            for i, k in enumerate(k_list):
                target_mask[i, : int(k)] = 1.0

        p_h = np.asarray(z["P_h_clean"], dtype=np.float32).reshape(-1) if "P_h_clean" in z else np.ones_like(k_list, dtype=np.float32)
        p_l = np.asarray(z["P_l_clean"], dtype=np.float32).reshape(-1) if "P_l_clean" in z else np.ones_like(p_h, dtype=np.float32)

    y_phys = np.concatenate([target_mask[..., None], y_rva], axis=2).astype(np.float32)
    y_phys[target_mask <= 0.5, 1:4] = 0.0
    y_bin = make_bins_from_phys(y_phys, cfg)
    y_delta_phys = make_delta_phys_from_phys(y_phys, y_bin, cfg)
    return k_list, p_h, p_l, y_phys, y_bin, y_delta_phys


def write_complex_member_to_h5(npz_path, member, h5, real_name, imag_name, chunk_rows, input_scale=1.0):
    with zipfile.ZipFile(npz_path, "r") as zf, zf.open(member, "r") as f:
        version = npformat.read_magic(f)
        shape, fortran_order, dtype = npformat._read_array_header(f, version)
        if fortran_order:
            raise ValueError(f"{member} is Fortran ordered; expected C order.")
        if dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
            raise ValueError(f"{member} dtype={dtype}, expected complex64 or complex128.")
        if len(shape) != 4:
            raise ValueError(f"{member} shape={shape}, expected [N,Nc,Ms,Nr].")

        n_samples = int(shape[0])
        sample_shape = tuple(int(v) for v in shape[1:])
        elems_per_row = int(np.prod(sample_shape))
        bytes_per_row = elems_per_row * dtype.itemsize
        chunks = (max(int(chunk_rows), 1), *sample_shape)

        d_real = h5.create_dataset(real_name, shape=shape, dtype="float32", chunks=chunks)
        d_imag = h5.create_dataset(imag_name, shape=shape, dtype="float32", chunks=chunks)

        print(f"Writing {member} -> {real_name}/{imag_name}")
        print(f"  source shape={shape}, dtype={dtype}, chunk_rows={chunk_rows}")
        for start in range(0, n_samples, int(chunk_rows)):
            rows = min(int(chunk_rows), n_samples - start)
            buf = read_exact(f, rows * bytes_per_row)
            arr = np.frombuffer(buf, dtype=dtype).reshape((rows, *sample_shape))
            d_real[start : start + rows] = (arr.real * input_scale).astype(np.float32, copy=False)
            d_imag[start : start + rows] = (arr.imag * input_scale).astype(np.float32, copy=False)
            if start == 0 or (start // int(chunk_rows)) % 250 == 0 or start + rows == n_samples:
                print(f"  wrote rows {start + rows}/{n_samples}")


def convert(args):
    cfg = {
        "r_min": args.r_min,
        "r_max": args.r_max,
        "v_min": args.v_min,
        "v_max": args.v_max,
        "theta_min": args.theta_min,
        "theta_max": args.theta_max,
        "Nbin_r": args.nbin_r,
        "Nbin_v": args.nbin_v,
        "Nbin_theta": args.nbin_theta,
    }
    npz_path = Path(args.npz)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    with zipfile.ZipFile(npz_path, "r") as zf:
        for member in ("X_h.npy", "X_l.npy", "K_list.npy"):
            if member not in zf.namelist():
                raise KeyError(f"{npz_path} missing required member: {member}")
        print("X_h header:", read_npy_header_from_zip(zf, "X_h.npy"))
        print("X_l header:", read_npy_header_from_zip(zf, "X_l.npy"))

    print("Building labels...")
    k_list, p_h, p_l, y_phys, y_bin, y_delta_phys = build_labels(npz_path, cfg)
    exist = k_list > 0
    valid_h = exist & np.isfinite(p_h) & (p_h > 0)
    raw_p_h_ref = float(np.median(p_h[valid_h])) if np.any(valid_h) else 1.0
    raw_p_h_ref = max(raw_p_h_ref, 1e-30)
    input_scale = math.sqrt(max(float(args.target_p_h_ref), 1e-30) / raw_p_h_ref)
    power_scale = input_scale ** 2
    print(f"raw_P_h_ref = {raw_p_h_ref:.4e}")
    print(f"target_P_h_ref = {args.target_p_h_ref:.4e}")
    print(f"input_scale = {input_scale:.4e}")

    print("Writing HDF5:", tmp_path)
    with h5py.File(tmp_path, "w") as h5:
        h5.create_dataset("K_list", data=k_list.astype(np.int64))
        h5.attrs["raw_P_h_ref"] = raw_p_h_ref
        h5.attrs["target_P_h_ref"] = float(args.target_p_h_ref)
        h5.attrs["input_scale"] = input_scale
        h5.attrs["power_scale"] = power_scale
        h5.create_dataset("P_h_clean", data=(p_h * power_scale).astype(np.float32))
        h5.create_dataset("P_l_clean", data=(p_l * power_scale).astype(np.float32))
        h5.create_dataset("Y_phys", data=y_phys.astype(np.float32))
        h5.create_dataset("Y_bin0_debug", data=y_bin.astype(np.int64))
        h5.create_dataset("Y_res_debug", data=y_delta_phys.astype(np.float32))

        cfg_group = h5.create_group("cfg")
        for key, value in cfg.items():
            cfg_group.create_dataset(key, data=np.array(value))

        write_complex_member_to_h5(
            npz_path,
            "X_l.npy",
            h5,
            "X_l_real",
            "X_l_imag",
            max(int(args.chunk_rows), 16),
            input_scale=input_scale,
        )
        write_complex_member_to_h5(
            npz_path,
            "X_h.npy",
            h5,
            "X_h_real",
            "X_h_imag",
            int(args.chunk_rows),
            input_scale=input_scale,
        )

    if out_path.exists():
        out_path.unlink()
    tmp_path.rename(out_path)
    print("Done:", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, help="Source Sionna NPZ test set.")
    parser.add_argument("--out", required=True, help="Output HDF5 path.")
    parser.add_argument("--chunk-rows", type=int, default=4)
    parser.add_argument("--target-p-h-ref", type=float, default=1.0)
    parser.add_argument("--r-min", type=float, default=50.0)
    parser.add_argument("--r-max", type=float, default=300.0)
    parser.add_argument("--v-min", type=float, default=-30.0)
    parser.add_argument("--v-max", type=float, default=30.0)
    parser.add_argument("--theta-min", type=float, default=-math.pi / 3)
    parser.add_argument("--theta-max", type=float, default=math.pi / 3)
    parser.add_argument("--nbin-r", type=int, default=32)
    parser.add_argument("--nbin-v", type=int, default=32)
    parser.add_argument("--nbin-theta", type=int, default=32)
    convert(parser.parse_args())


if __name__ == "__main__":
    main()

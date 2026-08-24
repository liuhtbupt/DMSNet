import csv
import itertools
import math
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


HERE = Path(__file__).resolve().parent
RELEASE_ROOT = HERE.parent
DEFAULT_CACHE_DIR = RELEASE_ROOT / "data"
DEFAULT_SNRS = (-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0)


class JointEvalDataset(Dataset):
    def __init__(self, h5_path, high_shape=(64, 112, 32), low_shape=(64, 14, 4)):
        self.h5_path = str(h5_path)
        self.high_shape = tuple(high_shape)
        self.low_shape = tuple(low_shape)
        self._file = None
        with h5py.File(self.h5_path, "r") as handle:
            self.K_list_all = np.asarray(handle["K_list"], dtype=np.int64).squeeze()
            self.num_samples = int(self.K_list_all.shape[0])
            self.Y_phys_all = self._fix_label_layout(
                np.asarray(handle["Y_phys"], dtype=np.float32), 4, "Y_phys"
            )
            self.Kmax = int(self.Y_phys_all.shape[1])
            self.P_h_all = (
                np.asarray(handle["P_h_clean"], dtype=np.float32).squeeze()
                if "P_h_clean" in handle
                else np.ones(self.num_samples, dtype=np.float32)
            )
            self.cfg = self._read_cfg(handle)
            exists = self.K_list_all > 0
            valid_power = (
                exists & np.isfinite(self.P_h_all) & (self.P_h_all > 0)
            )
            self.P_h_ref = (
                float(np.median(self.P_h_all[valid_power]))
                if np.any(valid_power)
                else 1.0
            )
            self.P_h_ref = max(self.P_h_ref, 1e-30)
            self.input_scale = float(handle.attrs.get("input_scale", 1.0))
            self.raw_P_h_ref = float(handle.attrs.get("raw_P_h_ref", np.nan))
            self.target_P_h_ref = float(
                handle.attrs.get("target_P_h_ref", self.P_h_ref)
            )
            self.high_storage_shape = tuple(handle["X_h_real"].shape)
            self.low_storage_shape = tuple(handle["X_l_real"].shape)

        print("\n========== Joint Evaluation Dataset ==========")
        print("file:", self.h5_path)
        print("num_samples:", self.num_samples)
        print(
            "K distribution:",
            {
                int(k): int(np.sum(self.K_list_all == k))
                for k in range(self.Kmax + 1)
            },
        )
        print("Y_phys:", self.Y_phys_all.shape)
        print("X_h_real:", self.high_storage_shape)
        print("X_l_real:", self.low_storage_shape)
        print("cfg:", self.cfg)
        print(f"input_scale = {self.input_scale:.4e}")
        print(f"raw_P_h_ref = {self.raw_P_h_ref:.4e}")
        print(f"P_h_ref = {self.P_h_ref:.4e}")
        print("==============================================\n")

    @staticmethod
    def _read_cfg(handle):
        cfg = {
            "r_min": 50.0,
            "r_max": 300.0,
            "v_min": -30.0,
            "v_max": 30.0,
            "theta_min": -math.pi / 3,
            "theta_max": math.pi / 3,
            "Nbin_r": 32,
            "Nbin_v": 32,
            "Nbin_theta": 32,
        }
        if "cfg" in handle:
            for key in cfg:
                if key in handle["cfg"]:
                    value = np.asarray(handle["cfg"][key]).squeeze()
                    cfg[key] = int(value) if key.startswith("Nbin_") else float(value)
        return cfg

    def _fix_label_layout(self, values, field_dim, name):
        if values.ndim != 3:
            raise ValueError(f"{name} must be 3D, got {values.shape}")
        n = self.num_samples
        if values.shape[0] == n and values.shape[2] == field_dim:
            return values.copy()
        if values.shape[0] == field_dim and values.shape[2] == n:
            return np.transpose(values, (2, 1, 0)).copy()
        if values.shape[0] == n and values.shape[1] == field_dim:
            return np.transpose(values, (0, 2, 1)).copy()
        if values.shape[1] == field_dim and values.shape[2] == n:
            return np.transpose(values, (2, 0, 1)).copy()
        raise ValueError(
            f"Unexpected {name} layout: {values.shape}, field_dim={field_dim}, N={n}"
        )

    def __len__(self):
        return self.num_samples

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def _ensure_open(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")

    def _read_sample(self, name, index, expected_shape):
        values = self._file[name]
        shape = values.shape
        nc, ms, nr = expected_shape
        if len(shape) == 4 and shape[:3] == (nc, ms, nr):
            return np.asarray(values[:, :, :, index], dtype=np.float32)
        if len(shape) == 4 and shape == (self.num_samples, nr, ms, nc):
            return np.transpose(
                np.asarray(values[index], dtype=np.float32), (2, 1, 0)
            )
        if len(shape) == 4 and shape == (self.num_samples, nc, ms, nr):
            return np.asarray(values[index], dtype=np.float32)
        raise ValueError(
            f"Cannot recognize {name} shape={shape}; expected sample {expected_shape}"
        )

    def __getitem__(self, index):
        self._ensure_open()
        xh = np.stack(
            [
                self._read_sample("X_h_real", index, self.high_shape),
                self._read_sample("X_h_imag", index, self.high_shape),
            ],
            axis=0,
        ).astype(np.float32)
        xl = np.stack(
            [
                self._read_sample("X_l_real", index, self.low_shape),
                self._read_sample("X_l_imag", index, self.low_shape),
            ],
            axis=0,
        ).astype(np.float32)
        return {
            "X_h": torch.from_numpy(xh),
            "X_l": torch.from_numpy(xl),
            "K": torch.tensor(int(self.K_list_all[index]), dtype=torch.long),
            "Y_phys": torch.from_numpy(self.Y_phys_all[index]),
            "P_h": torch.tensor(
                max(float(self.P_h_all[index]), 1e-30), dtype=torch.float32
            ),
            "idx": torch.tensor(int(index), dtype=torch.long),
        }


def make_loader(dataset, batch_size, num_workers, device):
    kwargs = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": int(num_workers),
        "pin_memory": device.type == "cuda",
    }
    if int(num_workers) > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**kwargs)


def bin_widths(cfg):
    return np.asarray(
        [
            (cfg["r_max"] - cfg["r_min"]) / cfg["Nbin_r"],
            (cfg["v_max"] - cfg["v_min"]) / cfg["Nbin_v"],
            (cfg["theta_max"] - cfg["theta_min"]) / cfg["Nbin_theta"],
        ],
        dtype=np.float64,
    )


def make_noise_generator(device, seed, snr_db, batch_index):
    snr_key = int(round((float(snr_db) + 100.0) * 1000.0))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + 1_000_003 * snr_key + int(batch_index))
    return generator


def add_awgn_to_batch(
    xh, xl, p_h, k_true, snr_db, p_h_ref, *, generator=None
):
    if snr_db is None:
        return xh, xl
    batch = xh.shape[0]
    power = p_h.to(device=xh.device, dtype=xh.dtype).view(batch).clamp_min(1e-30)
    reference = torch.full_like(power, float(p_h_ref))
    power = torch.where(k_true.to(xh.device).view(batch) > 0, power, reference)
    snr = torch.as_tensor(snr_db, device=xh.device, dtype=xh.dtype)
    if snr.ndim == 0:
        snr = snr.expand(batch)
    sigma = torch.sqrt(power / (10.0 ** (snr.view(batch) / 10.0)) / 2.0)
    shape = (batch, 1, 1, 1, 1)
    noise_h = torch.randn(
        xh.shape, device=xh.device, dtype=xh.dtype, generator=generator
    )
    noise_l = torch.randn(
        xl.shape, device=xl.device, dtype=xl.dtype, generator=generator
    )
    return xh + sigma.view(shape) * noise_h, xl + sigma.view(shape) * noise_l


def update_confusion(confusion, truth, prediction):
    np.add.at(
        confusion,
        (
            np.asarray(truth, dtype=np.int64),
            np.asarray(prediction, dtype=np.int64),
        ),
        1,
    )


def detection_metrics(confusion):
    confusion = np.asarray(confusion, dtype=np.int64)
    total = int(confusion.sum())
    metrics = {
        "count_accuracy": float(np.trace(confusion) / max(total, 1)),
        "count_mae": float(
            sum(
                abs(i - j) * int(confusion[i, j])
                for i in range(confusion.shape[0])
                for j in range(confusion.shape[1])
            )
            / max(total, 1)
        ),
    }
    f1_values = []
    for cls in range(confusion.shape[0]):
        tp = int(confusion[cls, cls])
        fp = int(confusion[:, cls].sum()) - tp
        fn = int(confusion[cls, :].sum()) - tp
        denominator = 2 * tp + fp + fn
        f1_values.append(0.0 if denominator == 0 else 2.0 * tp / denominator)
        class_total = int(confusion[cls].sum())
        metrics[f"K{cls}_acc"] = (
            float(confusion[cls, cls] / class_total) if class_total else float("nan")
        )
        metrics[f"K{cls}_num"] = class_total
    metrics["macro_f1"] = float(np.mean(f1_values))
    return metrics


def best_conditional_matches(prediction, truth, widths):
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    pred_count, true_count = len(prediction), len(truth)
    if min(pred_count, true_count) == 0:
        return []
    widths = np.asarray(widths, dtype=np.float64).reshape(1, 1, 3)
    cost = np.linalg.norm(
        (prediction[:, None, :] - truth[None, :, :]) / widths, axis=2
    )
    best_pairs = []
    best_cost = float("inf")
    if pred_count <= true_count:
        for true_perm in itertools.permutations(range(true_count), pred_count):
            value = sum(cost[pred_idx, true_perm[pred_idx]] for pred_idx in range(pred_count))
            if value < best_cost:
                best_cost = value
                best_pairs = [
                    (pred_idx, true_perm[pred_idx])
                    for pred_idx in range(pred_count)
                ]
    else:
        for pred_subset in itertools.combinations(range(pred_count), true_count):
            for true_perm in itertools.permutations(range(true_count), true_count):
                value = sum(
                    cost[pred_subset[pos], true_perm[pos]]
                    for pos in range(true_count)
                )
                if value < best_cost:
                    best_cost = value
                    best_pairs = [
                        (pred_subset[pos], true_perm[pos])
                        for pos in range(true_count)
                    ]
    return best_pairs


def percentile(values, q):
    values = np.asarray(values, dtype=np.float64)
    return float("nan") if values.size == 0 else float(np.percentile(values, q))


def percentile_summary(values, prefix):
    return {
        f"{prefix}_p50": percentile(values, 50),
        f"{prefix}_p90": percentile(values, 90),
        f"{prefix}_p95": percentile(values, 95),
    }


def timed_forward(model, xh, xl, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = model(xh, xl)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return output, time.perf_counter() - started


def write_csv_atomic(path, rows, fieldnames=None):
    path = Path(path)
    if not rows and not fieldnames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    columns = list(fieldnames or rows[0].keys())
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def print_progress(label, batch_number, total_batches, interval):
    if (
        batch_number == 1
        or batch_number == total_batches
        or batch_number % max(int(interval), 1) == 0
    ):
        print(f"[{label}] batch {batch_number}/{total_batches}", flush=True)

"""Evaluate target enumeration and complete-network runtime on the 10k test set."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

RELEASE_ROOT = Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

from evaluation.common import (
    DEFAULT_CACHE_DIR,
    DEFAULT_SNRS,
    JointEvalDataset,
    add_awgn_to_batch,
    detection_metrics,
    make_loader,
    make_noise_generator,
    print_progress,
    timed_forward,
    update_confusion,
    write_csv_atomic,
)
from models import build_joint_model


DEFAULT_TEST_H5 = DEFAULT_CACHE_DIR / "beiyou_test_10000_clean_joint.h5"
DEFAULT_OUT_DIR = RELEASE_ROOT / "outputs" / "count_runtime"


@torch.no_grad()
def evaluate_count_runtime(
    model,
    loader,
    device,
    snr_db,
    p_h_ref,
    noise_seed,
    log_interval,
    max_batches=None,
):
    confusion = np.zeros((6, 6), dtype=np.int64)
    runtime_seconds = 0.0
    sample_count = 0
    total_batches = len(loader)
    if max_batches is not None:
        total_batches = min(total_batches, int(max_batches))
    warmed_up = False

    for batch_index, batch in enumerate(loader):
        if batch_index >= total_batches:
            break
        xh = batch["X_h"].to(device, non_blocking=True)
        xl = batch["X_l"].to(device, non_blocking=True)
        k_true = batch["K"].to(device, non_blocking=True)
        p_h = batch["P_h"].to(device, non_blocking=True)
        generator = make_noise_generator(
            device, noise_seed, snr_db, batch_index
        )
        xh, xl = add_awgn_to_batch(
            xh,
            xl,
            p_h,
            k_true,
            snr_db,
            p_h_ref,
            generator=generator,
        )

        if not warmed_up:
            _ = model(xh, xl)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            warmed_up = True

        output, elapsed = timed_forward(model, xh, xl, device)
        runtime_seconds += elapsed
        sample_count += int(xh.shape[0])
        update_confusion(
            confusion,
            k_true.detach().cpu().numpy(),
            output["K_pred"].detach().cpu().numpy(),
        )
        print_progress(
            f"count/runtime SNR={snr_db:g}",
            batch_index + 1,
            total_batches,
            log_interval,
        )

    metrics = detection_metrics(confusion)
    metrics.update(
        snr_db=float(snr_db),
        samples=int(sample_count),
        runtime_ms_per_scene=(
            1000.0 * runtime_seconds / max(sample_count, 1)
        ),
        timed_runtime_seconds=float(runtime_seconds),
    )
    return metrics, confusion


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-h5", default=str(DEFAULT_TEST_H5))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--snrs", nargs="+", type=float, default=list(DEFAULT_SNRS)
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--noise-seed", type=int, default=20260630)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    test_h5 = Path(args.test_h5)
    if not test_h5.is_file():
        raise FileNotFoundError(
            f"Converted 10k test HDF5 not found: {test_h5}\n"
            "Run data_tools/convert_sionna_npz_to_h5.py first."
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dataset = JointEvalDataset(test_h5)
    loader = make_loader(
        dataset, args.batch_size, args.num_workers, device
    )
    model = build_joint_model(device=device).eval()

    metric_rows = []
    confusion_rows = []
    metric_path = out_dir / "count_runtime_snr_metrics.csv"
    confusion_path = out_dir / "count_confusion_by_snr.csv"
    for snr_db in args.snrs:
        print(f"\n========== Count/runtime SNR={snr_db:g} dB ==========")
        metrics, confusion = evaluate_count_runtime(
            model=model,
            loader=loader,
            device=device,
            snr_db=float(snr_db),
            p_h_ref=dataset.P_h_ref,
            noise_seed=args.noise_seed,
            log_interval=args.log_interval,
            max_batches=args.max_batches,
        )
        metric_rows.append(metrics)
        for true_k in range(confusion.shape[0]):
            for pred_k in range(confusion.shape[1]):
                confusion_rows.append(
                    {
                        "snr_db": float(snr_db),
                        "K_true": true_k,
                        "K_pred": pred_k,
                        "samples": int(confusion[true_k, pred_k]),
                    }
                )
        write_csv_atomic(metric_path, metric_rows)
        write_csv_atomic(confusion_path, confusion_rows)
        print(
            f"count_acc={metrics['count_accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"count_mae={metrics['count_mae']:.4f} "
            f"runtime={metrics['runtime_ms_per_scene']:.2f} ms/scene",
            flush=True,
        )
        print("Saved partial metrics:", metric_path)

    print("\nSaved count/runtime metrics:", metric_path)
    print("Saved confusion matrices:", confusion_path)


if __name__ == "__main__":
    main()

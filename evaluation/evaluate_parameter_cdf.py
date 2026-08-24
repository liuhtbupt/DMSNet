"""Export matched-target parameter errors and P50/P90/P95 on the K=2 test set."""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

RELEASE_ROOT = Path(__file__).resolve().parents[1]
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))

from evaluation.common import (
    DEFAULT_CACHE_DIR,
    JointEvalDataset,
    add_awgn_to_batch,
    best_conditional_matches,
    bin_widths,
    make_loader,
    make_noise_generator,
    percentile_summary,
    print_progress,
    write_csv_atomic,
)
from models import build_joint_model


DEFAULT_TEST_H5 = DEFAULT_CACHE_DIR / "beiyou_test_2000_clean_K2_joint.h5"
DEFAULT_OUT_DIR = RELEASE_ROOT / "outputs" / "parameter_cdf_k2"


@torch.no_grad()
def evaluate_parameter_cdf(
    model,
    loader,
    device,
    snr_db,
    p_h_ref,
    widths,
    noise_seed,
    log_interval,
    max_batches=None,
):
    error_rows = []
    scene_count = 0
    exact_count_scenes = 0
    true_targets = 0
    predicted_targets = 0
    matched_targets = 0
    total_batches = len(loader)
    if max_batches is not None:
        total_batches = min(total_batches, int(max_batches))

    for batch_index, batch in enumerate(loader):
        if batch_index >= total_batches:
            break
        xh = batch["X_h"].to(device, non_blocking=True)
        xl = batch["X_l"].to(device, non_blocking=True)
        k_true = batch["K"].to(device, non_blocking=True)
        p_h = batch["P_h"].to(device, non_blocking=True)
        y_phys = batch["Y_phys"].cpu().numpy()
        sample_indices = batch["idx"].cpu().numpy()
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
        output = model(xh, xl)
        k_prediction = output["K_pred"].detach().cpu().numpy()
        prediction = output["pred_phys"].detach().cpu().numpy()
        prediction_mask = output["pred_mask"].detach().cpu().numpy()
        k_truth = k_true.detach().cpu().numpy()

        for local_index in range(xh.shape[0]):
            true_k = int(k_truth[local_index])
            pred_k = int(k_prediction[local_index])
            if true_k != 2:
                raise ValueError(
                    "K=2 CDF evaluator received a non-K2 sample: "
                    f"sample={int(sample_indices[local_index])}, K={true_k}"
                )
            pred = prediction[local_index, prediction_mask[local_index]]
            truth = y_phys[local_index, :true_k, 1:4]
            pairs = best_conditional_matches(pred, truth, widths)
            scene_count += 1
            exact_count_scenes += int(pred_k == true_k)
            true_targets += true_k
            predicted_targets += pred_k
            matched_targets += len(pairs)

            for match_rank, (pred_index, true_index) in enumerate(pairs):
                residual = pred[pred_index] - truth[true_index]
                abs_error = np.abs(residual)
                error_rows.append(
                    {
                        "snr_db": float(snr_db),
                        "sample_idx": int(sample_indices[local_index]),
                        "match_rank": int(match_rank),
                        "K_true": true_k,
                        "K_pred": pred_k,
                        "pred_r_m": float(pred[pred_index, 0]),
                        "pred_v_mps": float(pred[pred_index, 1]),
                        "pred_theta_rad": float(pred[pred_index, 2]),
                        "true_r_m": float(truth[true_index, 0]),
                        "true_v_mps": float(truth[true_index, 1]),
                        "true_theta_rad": float(truth[true_index, 2]),
                        "abs_err_r_m": float(abs_error[0]),
                        "abs_err_v_mps": float(abs_error[1]),
                        "abs_err_theta_rad": float(abs_error[2]),
                        "abs_err_theta_deg": float(
                            abs_error[2] * 180.0 / math.pi
                        ),
                        "normalized_3d_error": float(
                            np.linalg.norm(residual / widths)
                        ),
                    }
                )
        print_progress(
            f"K2 CDF SNR={snr_db:g}",
            batch_index + 1,
            total_batches,
            log_interval,
        )

    range_errors = [row["abs_err_r_m"] for row in error_rows]
    velocity_errors = [row["abs_err_v_mps"] for row in error_rows]
    angle_rad_errors = [row["abs_err_theta_rad"] for row in error_rows]
    angle_deg_errors = [row["abs_err_theta_deg"] for row in error_rows]
    summary = {
        "snr_db": float(snr_db),
        "scenes": int(scene_count),
        "true_targets": int(true_targets),
        "predicted_targets": int(predicted_targets),
        "matched_targets": int(matched_targets),
        "matched_target_coverage": float(
            matched_targets / max(true_targets, 1)
        ),
        "exact_count_rate_on_k2": float(
            exact_count_scenes / max(scene_count, 1)
        ),
    }
    summary.update(percentile_summary(range_errors, "range_m"))
    summary.update(percentile_summary(velocity_errors, "velocity_mps"))
    summary.update(percentile_summary(angle_rad_errors, "angle_rad"))
    summary.update(percentile_summary(angle_deg_errors, "angle_deg"))
    return summary, error_rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-h5", default=str(DEFAULT_TEST_H5))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--snrs", nargs="+", type=float, default=[0.0])
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
            f"Converted K=2 test HDF5 not found: {test_h5}\n"
            "Run data_tools/convert_sionna_npz_to_h5.py first."
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dataset = JointEvalDataset(test_h5)
    unique_k = np.unique(dataset.K_list_all)
    if not np.array_equal(unique_k, np.asarray([2], dtype=np.int64)):
        raise ValueError(
            f"K=2 CDF dataset must contain only K=2, found {unique_k.tolist()}"
        )
    loader = make_loader(
        dataset, args.batch_size, args.num_workers, device
    )
    widths = bin_widths(dataset.cfg)
    model = build_joint_model(device=device).eval()

    summaries = []
    all_errors = []
    summary_path = out_dir / "parameter_cdf_k2_percentiles.csv"
    error_path = out_dir / "parameter_cdf_k2_error_samples.csv"
    for snr_db in args.snrs:
        print(f"\n========== K=2 parameter CDF SNR={snr_db:g} dB ==========")
        summary, errors = evaluate_parameter_cdf(
            model=model,
            loader=loader,
            device=device,
            snr_db=float(snr_db),
            p_h_ref=dataset.P_h_ref,
            widths=widths,
            noise_seed=args.noise_seed,
            log_interval=args.log_interval,
            max_batches=args.max_batches,
        )
        summaries.append(summary)
        all_errors.extend(errors)
        write_csv_atomic(summary_path, summaries)
        write_csv_atomic(error_path, all_errors)
        print(
            "Range P50/P90/P95="
            f"{summary['range_m_p50']:.6f}/"
            f"{summary['range_m_p90']:.6f}/"
            f"{summary['range_m_p95']:.6f} m"
        )
        print(
            "Velocity P50/P90/P95="
            f"{summary['velocity_mps_p50']:.6f}/"
            f"{summary['velocity_mps_p90']:.6f}/"
            f"{summary['velocity_mps_p95']:.6f} m/s"
        )
        print(
            "Angle P50/P90/P95="
            f"{summary['angle_deg_p50']:.6f}/"
            f"{summary['angle_deg_p90']:.6f}/"
            f"{summary['angle_deg_p95']:.6f} deg"
        )
        print(
            f"matched coverage={summary['matched_target_coverage']:.4f}, "
            f"K2 exact-count rate={summary['exact_count_rate_on_k2']:.4f}"
        )
        print("Saved partial percentile summary:", summary_path)
        print("Saved partial error samples:", error_path)

    print("\nSaved percentile summary:", summary_path)
    print("Saved raw CDF error samples:", error_path)


if __name__ == "__main__":
    main()

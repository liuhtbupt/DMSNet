import os
import sys

RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RELEASE_ROOT not in sys.path:
    sys.path.insert(0, RELEASE_ROOT)

from training import train_refiner_stage_a as trainer
from training.train_refiner_stage_b import Args as StageBArgs


class Args(StageBArgs):
    stage_b_dir = os.path.join(
        RELEASE_ROOT,
        "outputs",
        "refiner_stage_b",
    )
    init_refiner_ckpt = os.path.join(stage_b_dir, "patchrefiner_heatmap_best_loss.pt")
    save_dir = os.path.join(
        RELEASE_ROOT,
        "outputs",
        "refiner_stage_c",
    )

    # Stage C changes only the scene complexity: K=1..5 instead of K=1.
    # K=0 samples contain no parameter target and are excluded.
    stage_all_train_samples = 6000
    stage_all_val_samples = 1000
    stage_all_balance_k = True

    epochs = 50
    lr = 2.0e-5
    min_lr = 2.0e-6
    lr_warmup_batches = 300
    weight_decay = 1.0e-3

    stage_a_k1_clean_epochs = 0
    stage_b_k1_snr_epochs = 0

    # B has already learned the full AWGN range. Keep the same distribution
    # so any degradation in C is attributable mainly to multi-target mixing.
    stage_c_train_snr_min = -10
    stage_c_train_snr_max = 20
    stage_c_snr_curriculum_start_min = -10
    stage_c_snr_curriculum_epochs = epochs
    stage_c_val_snr_db = 0
    val_noise_seed = 240613

    save_densefft_visuals = False
    verify_densefft_crop = False

    seed = 2051
    swanlab_experiment_name = "patchrefiner_sionna_v1_stageC_multitarget_awgn"


if __name__ == "__main__":
    trainer.main(Args())

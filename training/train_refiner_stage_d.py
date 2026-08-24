import os
import sys

RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RELEASE_ROOT not in sys.path:
    sys.path.insert(0, RELEASE_ROOT)

from training import train_refiner_stage_a as trainer
from training.train_refiner_stage_c import Args as StageCArgs


class Args(StageCArgs):
    all_k_stage_label = "D_predcrop"

    stage_c_dir = os.path.join(
        RELEASE_ROOT,
        "outputs",
        "refiner_stage_c",
    )
    init_refiner_ckpt = os.path.join(
        stage_c_dir,
        "patchrefiner_heatmap_best_loss.pt",
    )
    save_dir = os.path.join(
        RELEASE_ROOT,
        "outputs",
        "refiner_stage_d",
    )

    # Stage D isolates coarse-bin errors:
    # true K is still supplied, while patch centers come from coarse ParamNet.
    coarse_count_mode = "teacher"
    crop_center_mode = "mix"
    crop_teacher_prob_start = 0.50
    crop_teacher_prob_end = 0.0
    crop_teacher_decay_epochs = 10

    # D0 runs automatically before model construction and training.
    run_pred_crop_audit = True
    pred_crop_audit_only = False
    pred_crop_audit_seed = 240615
    pred_crop_audit_candidate_spans = (0.75, 1.0, 1.5, 2.0, 2.5, 3.0)
    pred_crop_auto_quantile = 0.90
    pred_crop_auto_margin = 0.15
    pred_crop_auto_round_step = 0.25
    pred_crop_auto_min_span = (1.5, 1.5, 1.5)
    pred_crop_auto_max_span = (3.0, 3.0, 3.0)
    auto_configure_pred_crop_span = False

    # D0 shows that widening from 1.5 to 3.0 bins only raises joint coverage
    # from about 83.6% to 88.0%. Keep the spatial crop dense and expand only
    # the decoder range to the full patch extent.
    patch_half_span_bins = (1.5, 1.5, 1.5)
    axis_limits = (1.5, 1.5, 1.5)
    direct_max_delta = (1.5, 1.5, 1.5)
    loss_inside_patch_only = True

    # Continue from Stage C with a conservative learning rate.
    epochs = 40
    lr = 1.0e-5
    min_lr = 1.0e-6
    lr_warmup_batches = 200
    weight_decay = 1.0e-3

    stage_a_k1_clean_epochs = 0
    stage_b_k1_snr_epochs = 0
    stage_c_train_snr_min = -10
    stage_c_train_snr_max = 20
    stage_c_snr_curriculum_start_min = -10
    stage_c_snr_curriculum_epochs = epochs
    stage_c_val_snr_db = 0
    val_snr_db = 0

    # D0 is the pre-training diagnostic; old teacher-crop visualizations are
    # not representative of the Stage D input distribution.
    save_densefft_visuals = False
    verify_densefft_crop = False

    history_filename = "patchrefiner_stageD_history.csv"
    seed = 2071
    val_noise_seed = 240615
    swanlab_experiment_name = "patchrefiner_sionna_v1_stageD_predcrop"


if __name__ == "__main__":
    trainer.main(Args())

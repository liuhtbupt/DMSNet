import math
import os
import sys

RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RELEASE_ROOT not in sys.path:
    sys.path.insert(0, RELEASE_ROOT)

from training import train_refiner_stage_a as trainer


class Args:
    _this_dir = os.path.dirname(os.path.abspath(__file__))

    train_mat = ""
    test_mat = ""

    stage_a_dir = os.path.join(
        RELEASE_ROOT,
        "outputs",
        "refiner_stage_a",
    )
    init_refiner_ckpt = os.path.join(stage_a_dir, "patchrefiner_heatmap_best_loss.pt")
    save_dir = os.path.join(
        RELEASE_ROOT,
        "outputs",
        "refiner_stage_b",
    )

    count_model_path = os.path.join(RELEASE_ROOT, "models", "countnet.py")
    count_ckpt = os.path.join(RELEASE_ROOT, "weights", "countnet_sionna_best.pt")
    coarse_model_path = os.path.join(RELEASE_ROOT, "models", "paramnet.py")
    coarse_ckpt = os.path.join(RELEASE_ROOT, "weights", "paramnet_sionna_best.pt")

    use_sionna_h5 = True
    sionna_h5 = os.path.join(
        RELEASE_ROOT,
        "data",
        "beiyou_train_clean_50000_paramnet.h5",
    )
    train_ratio = 0.8
    split_seed = 2026
    count_model_path = os.path.join(RELEASE_ROOT, "models", "countnet.py")
    count_ckpt = os.path.join(RELEASE_ROOT, "weights", "countnet_sionna_best.pt")
    coarse_model_path = os.path.join(RELEASE_ROOT, "models", "paramnet.py")
    coarse_ckpt = os.path.join(RELEASE_ROOT, "weights", "paramnet_sionna_best.pt")

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

    refiner_feat_size = (256, 256, 256)
    patch_size = (25, 17, 15)
    patch_half_span_bins = (1.5, 1.5, 1.5)
    densefft_fft_mode = "current"
    axis_bins = (257, 193, 161)
    axis_limits = (0.75, 0.75, 0.75)
    refiner_channels = 96
    refiner_hidden_dim = 224
    refiner_dropout = 0.12
    softargmax_temp = 0.055
    patch_denoiser_arch = "stem"
    patch_denoise_hidden = 64
    patch_denoise_base_ch = 32
    patch_denoise_residual_scale = 0.35
    use_physical_axis_branch = False
    use_full_patch_direct_branch = True
    use_range_phase_direct_branch = False
    direct_max_delta = (0.75, 0.50, 0.50)

    axis_sigma_r = 0.025
    axis_sigma_v = 0.030
    axis_sigma_theta = 0.030
    delta_dim_weights = [2.2, 2.0, 1.2]
    direct_dim_weights = [3.0, 1.6, 1.0]
    lambda_heatmap = 0.45
    lambda_delta = 1.45
    lambda_phys_l1 = 0.55
    lambda_phys_l2 = 0.20
    lambda_branch_heatmap = 0.0
    lambda_branch_delta = 0.0
    lambda_direct_delta = 1.15
    lambda_range_phase_delta = 0.0

    use_clean_noisy_pair = False
    lambda_clean_pair_loss = 0.0
    lambda_clean_noisy_consistency = 0.0
    lambda_denoise_feature_consistency = 0.0
    patch_denoiser_ckpt = ""
    patch_denoiser_mode = "none"
    patch_denoiser_lr_scale = 0.10

    # Stage B starts from the converged Stage A model, so use a smaller LR.
    epochs = 60
    batch_size = 1
    test_batch_size = 1
    lr = 3.0e-5
    min_lr = 3.0e-6
    lr_warmup_batches = 300
    weight_decay = 1.0e-3
    grad_clip = 5.0
    use_amp = True

    match_lambda_obj = 0.5
    match_lambda_cls = 1.0
    match_lambda_delta = 0.5
    match_lambda_geo = 4.0
    match_lambda_arg = 2.0
    match_arg_clip_bins = 4.0
    match_lambda_r = 2.2
    match_lambda_v = 3.2
    match_lambda_theta = 2.2

    # Train on random AWGN. The lower bound expands from 20 dB to -10 dB.
    train_use_awgn = True
    train_snr_mode = "uniform"
    train_snr_min = -10
    train_snr_max = 20
    use_snr_curriculum = True
    snr_curriculum_start_min = 20
    snr_curriculum_epochs = epochs
    train_snr_list = [-10, -8, -6, -4, -2, 0, 5, 10, 15, 20]
    val_snr_db = 0
    val_noise_seed = 240611
    P_h_ref = None

    stage_k1_train_samples = 4096
    stage_k1_val_samples = 1024
    stage_a_k1_clean_epochs = 0
    stage_b_k1_snr_epochs = epochs
    stage_b_train_snr_min = -10
    stage_b_train_snr_max = 20
    stage_b_snr_curriculum_start_min = 20
    # Keep validation conditions fixed so epochs remain directly comparable.
    stage_b_val_snr_start = 0
    stage_b_val_snr_end = 0

    stage_c_train_snr_min = -12
    stage_c_train_snr_max = 0
    stage_c_snr_curriculum_start_min = -3
    stage_c_snr_curriculum_epochs = 60
    stage_c_val_snr_db = -10

    seed = 2041
    cpu = False
    num_workers = 8
    prefetch_factor = 4
    print_every = 100

    # Stage A already verified the physical crop. Avoid repeated 1024^3 startup work.
    save_densefft_visuals = False
    verify_densefft_crop = False

    use_swanlab = True
    swanlab_project = "countnet-isac"
    swanlab_experiment_name = "patchrefiner_sionna_v1_stageB_k1_awgn"
    swanlab_log_every_batches = 20


if __name__ == "__main__":
    trainer.main(Args())

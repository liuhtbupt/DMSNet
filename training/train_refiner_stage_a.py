import os
import csv
import math
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import Subset
import h5py
import numpy as np

try:
    import swanlab
except ImportError:
    swanlab = None

RELEASE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RELEASE_ROOT not in sys.path:
    sys.path.insert(0, RELEASE_ROOT)

from models.refiner import (
    AxisHeatmapPatchRefinerNet,
    count_parameters,
)
from training.train_paramnet import (
    MultiTargetParamDataset,
    make_loader,
    prepare_batch as prepare_batch_scene_snr,
    set_seed,
    build_matching_pairs,
    one_hot_count,
    save_checkpoint,
    save_history_csv,
    args_to_config,
    log_swanlab,
    build_scheduler,
    build_model as build_paramnet_model,
    tensor_to_float,
)


def build_coarse_model(args, device):
    model = build_paramnet_model(args, device)
    ckpt_path = str(getattr(args, "coarse_ckpt", "") or "").strip()
    if ckpt_path:
        print("Loading coarse ParamNet checkpoint:", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=True)
        print("Coarse ParamNet loaded. missing:", missing, "unexpected:", unexpected)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def add_fixed_reference_awgn(X_h, X_l, snr_db, power_ref):
    if snr_db is None:
        return X_h, X_l

    B = X_h.shape[0]
    snr_db = torch.as_tensor(snr_db, device=X_h.device, dtype=X_h.dtype)
    if snr_db.ndim == 0:
        snr_db = snr_db.expand(B)
    else:
        snr_db = snr_db.view(B)

    power_ref = max(float(power_ref), 1e-30)
    sigma2 = torch.full_like(snr_db, power_ref) / (10.0 ** (snr_db / 10.0))
    noise_std = torch.sqrt(sigma2 / 2.0).view(B, 1, 1, 1, 1)
    return X_h + noise_std * torch.randn_like(X_h), X_l + noise_std * torch.randn_like(X_l)


def get_train_snr_min(args):
    if not args.use_snr_curriculum:
        return float(args.train_snr_min)
    epoch = max(int(getattr(args, "current_epoch", 1)), 1)
    total_epochs = max(int(args.snr_curriculum_epochs), 1)
    progress = min(max((epoch - 1) / max(total_epochs - 1, 1), 0.0), 1.0)
    return float(args.snr_curriculum_start_min) + progress * (
        float(args.train_snr_min) - float(args.snr_curriculum_start_min)
    )


def prepare_batch(batch, device, args, train=True):
    noise_mode = str(getattr(args, "noise_power_mode", "scene_power")).lower()
    if noise_mode == "scene_power":
        return prepare_batch_scene_snr(batch, device, args, train=train)
    if noise_mode != "fixed_reference":
        raise ValueError(f"Unknown noise_power_mode: {noise_mode}")

    X_h = batch["X_h"].to(device, non_blocking=True)
    X_l = batch["X_l"].to(device, non_blocking=True)
    K_true = batch["K"].to(device, non_blocking=True)
    Y_phys = batch["Y_phys"].to(device, non_blocking=True)
    Y_bin = batch["Y_bin"].to(device, non_blocking=True)
    Y_delta_norm = batch["Y_delta_norm"].to(device, non_blocking=True)

    snr_db = None
    if train and args.train_use_awgn:
        if args.train_snr_mode == "uniform":
            snr_db = torch.empty(X_h.shape[0], device=device).uniform_(
                get_train_snr_min(args),
                float(args.train_snr_max),
            )
        elif args.train_snr_mode == "choice":
            choices = torch.tensor(args.train_snr_list, device=device, dtype=X_h.dtype)
            choice_idx = torch.randint(0, len(choices), (X_h.shape[0],), device=device)
            snr_db = choices[choice_idx]
        else:
            raise ValueError(f"Unknown train_snr_mode: {args.train_snr_mode}")
    elif (not train) and (args.val_snr_db is not None):
        snr_db = float(args.val_snr_db)

    X_h, X_l = add_fixed_reference_awgn(
        X_h,
        X_l,
        snr_db,
        power_ref=getattr(args, "noise_power_ref", 1.0),
    )
    return X_h, X_l, K_true, Y_phys, Y_bin, Y_delta_norm


@torch.no_grad()
def run_coarse_teacher(coarse_model, X_h, X_l, K_true, args):
    count_prior = one_hot_count(K_true, args.Kmax).to(device=X_h.device, dtype=X_h.dtype)
    return coarse_model(X_h, X_l, count_prior=count_prior)


def build_teacher_crop_bins(coarse_out, Y_bin, pairs_list):
    crop_bins = coarse_out["pred_bins"].detach().clone()
    for b, pairs in enumerate(pairs_list):
        for p_idx, t_idx in pairs:
            crop_bins[b, p_idx] = Y_bin[b, t_idx].long()
    return crop_bins


def load_radar_params(mat_path):
    params = {}
    with h5py.File(mat_path, "r") as f:
        if "params" not in f:
            return params
        for key in f["params"].keys():
            params[key] = float(np.array(f["params"][key]).squeeze())
    return params


def bin_centers_to_phys(bins, args):
    widths = torch.tensor(
        [
            (args.r_max - args.r_min) / args.Nbin_r,
            (args.v_max - args.v_min) / args.Nbin_v,
            (args.theta_max - args.theta_min) / args.Nbin_theta,
        ],
        device=bins.device,
        dtype=torch.float32,
    )
    mins = torch.tensor([args.r_min, args.v_min, args.theta_min], device=bins.device, dtype=torch.float32)
    return mins.view(1, 1, 3) + (bins.float() + 0.5) * widths.view(1, 1, 3)


def build_teacher_crop_centers_phys(coarse_out, Y_bin, pairs_list, args):
    if "pred_phys" in coarse_out:
        centers = coarse_out["pred_phys"].detach().clone()
    else:
        centers = bin_centers_to_phys(coarse_out["pred_bins"].detach(), args)
    true_bin_centers = bin_centers_to_phys(Y_bin, args)
    for b, pairs in enumerate(pairs_list):
        for p_idx, t_idx in pairs:
            centers[b, p_idx] = true_bin_centers[b, t_idx]
    return centers


@torch.no_grad()
def run_coarse_configured(coarse_model, X_h, X_l, K_true, args):
    count_mode = str(getattr(args, "coarse_count_mode", "teacher")).lower()
    if count_mode == "teacher":
        return run_coarse_teacher(coarse_model, X_h, X_l, K_true, args)
    if count_mode == "pred":
        return coarse_model(X_h, X_l, count_prior=None)
    raise ValueError(f"Unknown coarse_count_mode: {count_mode}")


def get_runtime_teacher_crop_prob(args):
    crop_mode = str(getattr(args, "crop_center_mode", "teacher")).lower()
    if crop_mode == "teacher":
        return 1.0
    if crop_mode == "pred":
        return 0.0
    if crop_mode != "mix":
        raise ValueError(f"Unknown crop_center_mode: {crop_mode}")

    epoch = max(int(getattr(args, "current_epoch", 1)), 1)
    start = float(getattr(args, "crop_teacher_prob_start", 1.0))
    end = float(getattr(args, "crop_teacher_prob_end", 0.0))
    total = max(int(getattr(args, "crop_teacher_decay_epochs", 1)), 1)
    progress = min(max((epoch - 1) / max(total - 1, 1), 0.0), 1.0)
    return start + progress * (end - start)


def build_runtime_crop_centers_phys(coarse_out, Y_bin, pairs_list, args, train):
    predicted = bin_centers_to_phys(coarse_out["pred_bins"].detach(), args)
    teacher = bin_centers_to_phys(Y_bin, args)
    teacher_prob = get_runtime_teacher_crop_prob(args)
    if teacher_prob <= 0.0:
        return predicted, teacher_prob

    centers = predicted.clone()
    for b, pairs in enumerate(pairs_list):
        for p_idx, t_idx in pairs:
            use_teacher = teacher_prob >= 1.0
            if train and 0.0 < teacher_prob < 1.0:
                use_teacher = torch.rand((), device=centers.device).item() < teacher_prob
            if use_teacher:
                centers[b, p_idx] = teacher[b, t_idx]
    return centers, teacher_prob


def _as_numpy_image(x):
    x = x.detach().float().cpu()
    lo = torch.quantile(x.flatten(), 0.01)
    hi = torch.quantile(x.flatten(), 0.99)
    x = (x - lo) / (hi - lo).clamp_min(1e-6)
    return x.clamp(0.0, 1.0).numpy()


@torch.no_grad()
def _prepare_visual_tensors(batch, sample_idx, device, args, noisy=False, snr_db=20.0):
    old_train_use_awgn = args.train_use_awgn
    old_val_snr_db = args.val_snr_db
    args.train_use_awgn = bool(noisy)
    args.val_snr_db = float(snr_db) if noisy else None
    tensors = prepare_batch(batch, device, args, train=False)
    args.train_use_awgn = old_train_use_awgn
    args.val_snr_db = old_val_snr_db
    return tuple(t[sample_idx : sample_idx + 1] if torch.is_tensor(t) else t for t in tensors)


@torch.no_grad()
def _find_visual_sample(loader, device, args, min_k):
    old_train_use_awgn = args.train_use_awgn
    old_val_snr_db = args.val_snr_db
    args.train_use_awgn = False
    args.val_snr_db = None
    try:
        for batch in loader:
            _, _, K_true, _, _, _ = prepare_batch(batch, device, args, train=False)
            hits = torch.nonzero(K_true >= int(min_k), as_tuple=False).flatten()
            if hits.numel() > 0:
                return batch, int(hits[0].item())
    finally:
        args.train_use_awgn = old_train_use_awgn
        args.val_snr_db = old_val_snr_db
    return None, None


@torch.no_grad()
def _save_one_densefft_visual(refiner, coarse_model, batch, sample_idx, device, args, out_dir, tag, noisy=False, snr_db=20.0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X_h, X_l, K_true, Y_phys, Y_bin, Y_delta_norm = _prepare_visual_tensors(
        batch, sample_idx, device, args, noisy=noisy, snr_db=snr_db
    )
    coarse_out = run_coarse_teacher(coarse_model, X_h, X_l, K_true, args)
    pairs_list = build_matching_pairs(coarse_out, Y_phys, Y_bin, Y_delta_norm, args)
    if len(pairs_list[0]) == 0:
        return
    crop_center_phys = build_teacher_crop_centers_phys(coarse_out, Y_bin, pairs_list, args)
    p_idx, t_idx = pairs_list[0][0]

    extractor = refiner.patch_extractor
    center_h = extractor._physical_centers_to_indices(crop_center_phys, "high")[0, p_idx].round().long()
    center_l = extractor._physical_centers_to_indices(crop_center_phys, "low")[0, p_idx].round().long()

    def slices_full(x, center):
        r0, v0, th0 = [int(v.item()) for v in center]
        return [
            ("R-V full", x[:, :, th0]),
            ("R-T full", x[:, v0, :]),
            ("V-T full", x[r0, :, :]),
        ]

    def slices_patch(x):
        pr, pv, pt = [s // 2 for s in x.shape]
        return [
            ("R-V patch", x[:, :, pt]),
            ("R-T patch", x[:, pv, :]),
            ("V-T patch", x[pr, :, :]),
        ]

    def build_band_panels(band_name, X, center):
        z = extractor._fft3(X)
        logmag = torch.log1p(z.abs())
        full = logmag[0]
        patch = extractor._crop(logmag[:, None], crop_center_phys, band_name)[0, p_idx, 0]
        full_slices = [(f"{band_name} {name}", img.detach().cpu()) for name, img in slices_full(full, center)]
        patch_slices = [(f"{band_name} {name}", img.detach().cpu()) for name, img in slices_patch(patch)]
        patch_cpu = patch.detach().cpu()
        del z, logmag, full, patch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return full_slices + patch_slices, patch_cpu

    panels = []
    high_panels, patch_h = build_band_panels("high", X_h, center_h)
    low_panels, patch_l = build_band_panels("low", X_l, center_l)
    panels.extend(high_panels)
    panels.extend(low_panels)

    widths = torch.tensor(
        [
            (float(args.r_max) - float(args.r_min)) / float(args.Nbin_r),
            (float(args.v_max) - float(args.v_min)) / float(args.Nbin_v),
            (float(args.theta_max) - float(args.theta_min)) / float(args.Nbin_theta),
        ],
        device=device,
        dtype=Y_phys.dtype,
    )
    target_delta = (Y_phys[0, t_idx, 1:4] - crop_center_phys[0, p_idx]) / widths
    expected_pixel = target_delta * refiner.axis_scales.to(target_delta)
    patch_center = torch.tensor(
        [(s - 1) / 2.0 for s in patch_h.shape],
        device=device,
        dtype=expected_pixel.dtype,
    )
    true_patch_pos = (patch_center + expected_pixel).detach().cpu()

    fig, axes = plt.subplots(4, 3, figsize=(13.5, 13.5), constrained_layout=True)
    fig.suptitle(
        f"{tag} | K={int(K_true[0].item())} | noisy={noisy} | snr={snr_db if noisy else 'clean'} | "
        f"bin={Y_bin[0, t_idx].detach().cpu().tolist()} | delta={Y_delta_norm[0, t_idx].detach().cpu().tolist()}",
        fontsize=10,
    )
    for ax, (title, img) in zip(axes.flatten(), panels):
        ax.imshow(_as_numpy_image(img).T, origin="lower", aspect="auto", cmap="magma")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.flatten()[len(panels) :]:
        ax.axis("off")

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{tag}.png"), dpi=180)
    plt.close(fig)

    def plane_true_xy(name):
        if "R-V" in name:
            return float(true_patch_pos[0]), float(true_patch_pos[1])
        if "R-T" in name:
            return float(true_patch_pos[0]), float(true_patch_pos[2])
        if "V-T" in name:
            return float(true_patch_pos[1]), float(true_patch_pos[2])
        return None

    fig_patch, axes_patch = plt.subplots(2, 3, figsize=(13.5, 8.0), constrained_layout=True)
    fig_patch.suptitle(
        f"{tag} patch-only log-magnitude slices | "
        f"target pixel offset={expected_pixel.detach().cpu().tolist()}",
        fontsize=11,
    )
    patch_panels = []
    patch_panels.extend([(f"high {name}", img) for name, img in slices_patch(patch_h)])
    patch_panels.extend([(f"low {name}", img) for name, img in slices_patch(patch_l)])
    for ax, (title, img) in zip(axes_patch.flatten(), patch_panels):
        ax.imshow(_as_numpy_image(img).T, origin="lower", aspect="auto", cmap="magma")
        center_x = (img.shape[0] - 1) / 2.0
        center_y = (img.shape[1] - 1) / 2.0
        ax.axvline(center_x, color="cyan", linewidth=0.8, alpha=0.8, linestyle="--")
        ax.axhline(center_y, color="cyan", linewidth=0.8, alpha=0.8, linestyle="--")
        true_xy = plane_true_xy(title)
        if true_xy is not None:
            ax.axvline(true_xy[0], color="lime", linewidth=0.9, alpha=0.9, linestyle=":")
            ax.axhline(true_xy[1], color="lime", linewidth=0.9, alpha=0.9, linestyle=":")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig_patch.savefig(os.path.join(out_dir, f"{tag}_patch_only.png"), dpi=180)
    plt.close(fig_patch)

    def axis_profiles(x):
        pr, pv, pt = [s // 2 for s in x.shape]
        return {
            "range center": x[:, pv, pt],
            "range max": x.amax(dim=(1, 2)),
            "velocity center": x[pr, :, pt],
            "velocity max": x.amax(dim=(0, 2)),
            "theta center": x[pr, pv, :],
            "theta max": x.amax(dim=(0, 1)),
        }

    fig_prof, axes_prof = plt.subplots(3, 2, figsize=(12.0, 9.0), constrained_layout=True)
    fig_prof.suptitle(f"{tag} patch 1D profiles", fontsize=11)
    high_profiles = axis_profiles(patch_h)
    low_profiles = axis_profiles(patch_l)
    profile_names = ["range center", "range max", "velocity center", "velocity max", "theta center", "theta max"]
    for ax, name in zip(axes_prof.flatten(), profile_names):
        h = high_profiles[name].detach().float().cpu()
        l = low_profiles[name].detach().float().cpu()
        ax.plot(h.numpy(), label="high", linewidth=1.5)
        ax.plot(l.numpy(), label="low", linewidth=1.5)
        if name.startswith("range"):
            true_x = float(true_patch_pos[0])
        elif name.startswith("velocity"):
            true_x = float(true_patch_pos[1])
        else:
            true_x = float(true_patch_pos[2])
        center_x = (len(h) - 1) / 2.0
        high_peak_x = int(torch.argmax(h).item())
        low_peak_x = int(torch.argmax(l).item())
        ax.axvline(center_x, color="k", linewidth=0.8, linestyle="--", alpha=0.75, label="crop center")
        ax.axvline(true_x, color="lime", linewidth=1.0, linestyle=":", alpha=0.95, label="true target")
        ax.axvline(high_peak_x, color="tab:blue", linewidth=0.9, linestyle="-.", alpha=0.8, label="high peak")
        ax.axvline(low_peak_x, color="tab:orange", linewidth=0.9, linestyle="-.", alpha=0.8, label="low peak")
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig_prof.savefig(os.path.join(out_dir, f"{tag}_profiles.png"), dpi=180)
    plt.close(fig_prof)


@torch.no_grad()
def save_densefft_visualizations(refiner, coarse_model, val_loader_k1, val_loader_all, device, args):
    if not bool(getattr(args, "save_densefft_visuals", False)):
        return
    out_dir = os.path.join(args.save_dir, "densefft_visuals")
    print(f"Saving dense FFT visualizations to: {out_dir}")

    k1_batch, k1_idx = _find_visual_sample(val_loader_k1, device, args, min_k=1)
    multi_batch, multi_idx = _find_visual_sample(val_loader_all, device, args, min_k=2)

    if k1_batch is not None:
        _save_one_densefft_visual(refiner, coarse_model, k1_batch, k1_idx, device, args, out_dir, "k1_clean", noisy=False)
        _save_one_densefft_visual(
            refiner, coarse_model, k1_batch, k1_idx, device, args, out_dir, "k1_noisy_snr20", noisy=True, snr_db=20.0
        )
        _save_one_densefft_visual(
            refiner, coarse_model, k1_batch, k1_idx, device, args, out_dir, "k1_noisy_snr0", noisy=True, snr_db=0.0
        )
    if multi_batch is not None:
        _save_one_densefft_visual(refiner, coarse_model, multi_batch, multi_idx, device, args, out_dir, "multi_clean", noisy=False)
        _save_one_densefft_visual(
            refiner, coarse_model, multi_batch, multi_idx, device, args, out_dir, "multi_noisy_snr10", noisy=True, snr_db=10.0
        )


@torch.no_grad()
def verify_densefft_crop_theory(refiner, coarse_model, val_loader_k1, device, args):
    if not bool(getattr(args, "verify_densefft_crop", True)):
        return
    batch, sample_idx = _find_visual_sample(val_loader_k1, device, args, min_k=1)
    if batch is None:
        return
    X_h, X_l, K_true, Y_phys, Y_bin, Y_delta_norm = _prepare_visual_tensors(
        batch, sample_idx, device, args, noisy=False
    )
    coarse_out = run_coarse_teacher(coarse_model, X_h, X_l, K_true, args)
    pairs_list = build_matching_pairs(coarse_out, Y_phys, Y_bin, Y_delta_norm, args)
    if len(pairs_list[0]) == 0:
        return
    crop_center_phys = build_teacher_crop_centers_phys(coarse_out, Y_bin, pairs_list, args)
    p_idx, t_idx = pairs_list[0][0]
    extractor = refiner.patch_extractor
    widths = torch.tensor(
        [
            (args.r_max - args.r_min) / args.Nbin_r,
            (args.v_max - args.v_min) / args.Nbin_v,
            (args.theta_max - args.theta_min) / args.Nbin_theta,
        ],
        device=device,
        dtype=X_h.dtype,
    )
    target_delta = (Y_phys[0, t_idx, 1:4] - crop_center_phys[0, p_idx]) / widths
    expected_pixel = target_delta * refiner.axis_scales.to(target_delta)

    lines = [
        "========== Dense FFT Crop Theory Check ==========",
        f"K={int(K_true[0].item())}, target={t_idx}, pred_slot={p_idx}",
        f"true_bin={Y_bin[0, t_idx].detach().cpu().tolist()}",
        f"crop_center_phys={crop_center_phys[0, p_idx].detach().cpu().tolist()}",
        f"true_phys={Y_phys[0, t_idx, 1:4].detach().cpu().tolist()}",
        f"target_delta_bins={target_delta.detach().cpu().tolist()}",
        f"expected_patch_pixel_offset={expected_pixel.detach().cpu().tolist()}",
    ]

    for band, X in (("high", X_h), ("low", X_l)):
        logmag = torch.log1p(extractor._fft3(X).abs())
        patch = extractor._crop(logmag[:, None], crop_center_phys, band)[0, p_idx, 0]
        flat_idx = torch.argmax(patch)
        peak = torch.tensor(torch.unravel_index(flat_idx, patch.shape), device=device, dtype=X_h.dtype)
        center = torch.tensor([s // 2 for s in patch.shape], device=device, dtype=X_h.dtype)
        peak_offset = peak - center
        lines.append(f"[{band}] patch_shape={tuple(int(s) for s in patch.shape)}")
        lines.append(f"[{band}] peak_offset_pixels={peak_offset.detach().cpu().tolist()}")
        lines.append(f"[{band}] peak_minus_expected_pixels={(peak_offset - expected_pixel).detach().cpu().tolist()}")
        del logmag, patch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    lines.append("=================================================")
    text = "\n".join(lines)
    print(text)
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "densefft_crop_theory_check.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


def gaussian_axis_loss(logits, axis, target, sigma):
    axis = axis.to(device=logits.device, dtype=logits.dtype).view(1, -1)
    target = target.to(device=logits.device, dtype=logits.dtype).view(-1, 1)
    target = torch.maximum(torch.minimum(target, axis.max()), axis.min())
    label = torch.exp(-0.5 * ((axis - target) / float(sigma)).square())
    label = label / label.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return -(label * F.log_softmax(logits, dim=-1)).sum(dim=-1)


def heatmap_refiner_loss(ref_out, crop_center_phys, coarse_out, Y_phys, Y_bin, Y_delta_norm, args):
    pairs_list = build_matching_pairs(coarse_out, Y_phys, Y_bin, Y_delta_norm, args)
    matched_b, matched_p, matched_t = [], [], []
    for b, pairs in enumerate(pairs_list):
        for p_idx, t_idx in pairs:
            matched_b.append(b)
            matched_p.append(p_idx)
            matched_t.append(t_idx)

    zero = ref_out["delta_from_crop"].sum() * 0.0
    if len(matched_b) == 0:
        empty = {
            "loss_heatmap": zero.detach(),
            "loss_delta": zero.detach(),
            "loss_phys_l1": zero.detach(),
            "loss_phys_l2": zero.detach(),
            "loss_phys_branch": zero.detach(),
            "loss_direct": zero.detach(),
            "loss_range_phase": zero.detach(),
            "r_acc": zero.detach(),
            "v_acc": zero.detach(),
            "theta_acc": zero.detach(),
            "r_w1": zero.detach(),
            "v_w1": zero.detach(),
            "theta_w1": zero.detach(),
            "mae_r": zero.detach(),
            "mae_v": zero.detach(),
            "mae_theta": zero.detach(),
            "rmse_r": zero.detach(),
            "rmse_v": zero.detach(),
            "rmse_theta": zero.detach(),
            "delta_mae_r": zero.detach(),
            "delta_mae_v": zero.detach(),
            "delta_mae_theta": zero.detach(),
            "pred_delta_mean_r": zero.detach(),
            "pred_delta_mean_v": zero.detach(),
            "pred_delta_mean_theta": zero.detach(),
            "pred_delta_std_r": zero.detach(),
            "pred_delta_std_v": zero.detach(),
            "pred_delta_std_theta": zero.detach(),
            "target_delta_mean_r": zero.detach(),
            "target_delta_mean_v": zero.detach(),
            "target_delta_mean_theta": zero.detach(),
            "target_delta_std_r": zero.detach(),
            "target_delta_std_v": zero.detach(),
            "target_delta_std_theta": zero.detach(),
            "delta_corr_r": zero.detach(),
            "delta_corr_v": zero.detach(),
            "delta_corr_theta": zero.detach(),
            "r_phys_delta_mae": zero.detach(),
            "r_phys_delta_std": zero.detach(),
            "r_phys_delta_corr": zero.detach(),
            "v_phys_delta_mae": zero.detach(),
            "v_phys_delta_std": zero.detach(),
            "v_phys_delta_corr": zero.detach(),
            "direct_delta_mae_r": zero.detach(),
            "direct_delta_mae_v": zero.detach(),
            "direct_delta_mae_theta": zero.detach(),
            "direct_delta_corr_r": zero.detach(),
            "direct_delta_corr_v": zero.detach(),
            "direct_delta_corr_theta": zero.detach(),
            "direct_gate_r": zero.detach(),
            "direct_gate_v": zero.detach(),
            "direct_gate_theta": zero.detach(),
            "range_phase_delta_mae": zero.detach(),
            "range_phase_delta_corr": zero.detach(),
            "range_phase_gate": zero.detach(),
            "range_physical_gain": zero.detach(),
            "velocity_physical_gain": zero.detach(),
            "inside_r": zero.detach(),
            "inside_v": zero.detach(),
            "inside_theta": zero.detach(),
            "inside_all": zero.detach(),
            "coarse_exact_r": zero.detach(),
            "coarse_exact_v": zero.detach(),
            "coarse_exact_theta": zero.detach(),
            "coarse_w1_r": zero.detach(),
            "coarse_w1_v": zero.detach(),
            "coarse_w1_theta": zero.detach(),
            "conditional_mae_r": zero.detach(),
            "conditional_mae_v": zero.detach(),
            "conditional_mae_theta": zero.detach(),
            "conditional_rmse_r": zero.detach(),
            "conditional_rmse_v": zero.detach(),
            "conditional_rmse_theta": zero.detach(),
            "num_obj": 0,
        }
        return zero, empty

    device = crop_center_phys.device
    mb = torch.tensor(matched_b, dtype=torch.long, device=device)
    mp = torch.tensor(matched_p, dtype=torch.long, device=device)
    mt = torch.tensor(matched_t, dtype=torch.long, device=device)

    pred_delta = ref_out["delta_from_crop"][mb, mp]
    pred_phys = ref_out["pred_phys"][mb, mp]
    true_phys = Y_phys[mb, mt, 1:4]
    true_bin = Y_bin[mb, mt].long()
    crop_phys = crop_center_phys[mb, mp].to(true_phys)
    widths = torch.tensor(
        [
            (args.r_max - args.r_min) / args.Nbin_r,
            (args.v_max - args.v_min) / args.Nbin_v,
            (args.theta_max - args.theta_min) / args.Nbin_theta,
        ],
        dtype=pred_phys.dtype,
        device=device,
    )
    target_delta = (true_phys - crop_phys) / widths.view(1, 3)
    limits = torch.tensor(args.axis_limits, dtype=target_delta.dtype, device=device).view(1, 3)
    inside_for_loss = (target_delta.abs() <= limits).all(dim=1)
    if not bool(getattr(args, "loss_inside_patch_only", False)):
        inside_for_loss = torch.ones_like(inside_for_loss)

    r_logits = ref_out["r_logits"][mb, mp]
    v_logits = ref_out["v_logits"][mb, mp]
    theta_logits = ref_out["theta_logits"][mb, mp]

    dim_w = torch.tensor(args.delta_dim_weights, dtype=pred_delta.dtype, device=device)
    direct_delta = (
        ref_out["direct_delta_from_crop"][mb, mp]
        if "direct_delta_from_crop" in ref_out
        else None
    )
    range_phase_delta = (
        ref_out["range_phase_delta"][mb, mp]
        if "range_phase_delta" in ref_out
        else None
    )

    if inside_for_loss.any():
        target_delta_loss = target_delta[inside_for_loss]
        pred_delta_loss = pred_delta[inside_for_loss]
        pred_phys_loss = pred_phys[inside_for_loss]
        true_phys_loss = true_phys[inside_for_loss]
        r_logits_loss = r_logits[inside_for_loss]
        v_logits_loss = v_logits[inside_for_loss]
        theta_logits_loss = theta_logits[inside_for_loss]

        loss_r_hm = gaussian_axis_loss(r_logits_loss, ref_out["r_axis"] if "r_axis" in ref_out else args.r_axis, target_delta_loss[:, 0], args.axis_sigma_r)
        loss_v_hm = gaussian_axis_loss(v_logits_loss, ref_out["v_axis"] if "v_axis" in ref_out else args.v_axis, target_delta_loss[:, 1], args.axis_sigma_v)
        loss_theta_hm = gaussian_axis_loss(theta_logits_loss, ref_out["theta_axis"] if "theta_axis" in ref_out else args.theta_axis, target_delta_loss[:, 2], args.axis_sigma_theta)
        loss_heatmap = (
            dim_w[0] * loss_r_hm.mean()
            + dim_w[1] * loss_v_hm.mean()
            + dim_w[2] * loss_theta_hm.mean()
        ) / dim_w.sum()
        loss_delta = (
            F.smooth_l1_loss(
                pred_delta_loss,
                target_delta_loss,
                reduction="none",
            )
            * dim_w.view(1, 3)
        ).mean()
    else:
        target_delta_loss = target_delta[:0]
        pred_delta_loss = pred_delta[:0]
        pred_phys_loss = pred_phys[:0]
        true_phys_loss = true_phys[:0]
        loss_heatmap = pred_delta.sum() * 0.0
        loss_delta = pred_delta.sum() * 0.0

    if args.lambda_direct_delta > 0.0 and direct_delta is not None and inside_for_loss.any():
        direct_w = torch.tensor(args.direct_dim_weights, dtype=pred_delta.dtype, device=device)
        loss_direct = (
            F.smooth_l1_loss(
                direct_delta[inside_for_loss],
                target_delta_loss,
                reduction="none",
            )
            * direct_w.view(1, 3)
        ).mean()
    else:
        loss_direct = pred_delta.sum() * 0.0

    if args.lambda_range_phase_delta > 0.0 and range_phase_delta is not None and inside_for_loss.any():
        loss_range_phase = F.smooth_l1_loss(
            range_phase_delta[inside_for_loss],
            target_delta_loss[:, 0],
            reduction="mean",
        )
    else:
        loss_range_phase = pred_delta.sum() * 0.0

    if args.lambda_branch_heatmap > 0.0 and args.lambda_branch_delta > 0.0 and "r_phys_logits" in ref_out and "v_phys_logits" in ref_out and inside_for_loss.any():
        r_phys_logits = ref_out["r_phys_logits"][mb, mp][inside_for_loss]
        v_phys_logits = ref_out["v_phys_logits"][mb, mp][inside_for_loss]
        r_phys_delta_loss = ref_out["r_phys_delta"][mb, mp][inside_for_loss]
        v_phys_delta_loss = ref_out["v_phys_delta"][mb, mp][inside_for_loss]

        loss_r_phys_hm = gaussian_axis_loss(
            r_phys_logits,
            ref_out["r_axis"] if "r_axis" in ref_out else args.r_axis,
            target_delta_loss[:, 0],
            args.axis_sigma_r,
        ).mean()
        loss_v_phys_hm = gaussian_axis_loss(
            v_phys_logits,
            ref_out["v_axis"] if "v_axis" in ref_out else args.v_axis,
            target_delta_loss[:, 1],
            args.axis_sigma_v,
        ).mean()
        loss_r_phys_delta = F.smooth_l1_loss(r_phys_delta_loss, target_delta_loss[:, 0], reduction="mean")
        loss_v_phys_delta = F.smooth_l1_loss(v_phys_delta_loss, target_delta_loss[:, 1], reduction="mean")
        loss_phys_branch = (
            args.lambda_branch_heatmap * 0.5 * (loss_r_phys_hm + loss_v_phys_hm)
            + args.lambda_branch_delta * 0.5 * (loss_r_phys_delta + loss_v_phys_delta)
        )
    else:
        loss_phys_branch = pred_delta.sum() * 0.0

    if inside_for_loss.any():
        norm_err = (pred_phys_loss - true_phys_loss) / widths.view(1, 3)
        loss_phys_l1 = (norm_err.abs() * dim_w.view(1, 3)).mean()
        loss_phys_l2 = (norm_err.square() * dim_w.view(1, 3)).mean()
    else:
        loss_phys_l1 = pred_delta.sum() * 0.0
        loss_phys_l2 = pred_delta.sum() * 0.0
    loss = (
        args.lambda_heatmap * loss_heatmap
        + args.lambda_delta * loss_delta
        + args.lambda_phys_l1 * loss_phys_l1
        + args.lambda_phys_l2 * loss_phys_l2
        + loss_phys_branch
        + args.lambda_direct_delta * loss_direct
        + args.lambda_range_phase_delta * loss_range_phase
    )

    with torch.no_grad():
        pred_bin = ref_out["pred_bins"][mb, mp].long()
        err_bin = pred_bin - true_bin
        bin_acc = (err_bin == 0).float().mean(dim=0)
        bin_w1 = (err_bin.abs() <= 1).float().mean(dim=0)
        phys_err = pred_phys - true_phys
        phys_mae = phys_err.abs().mean(dim=0)
        phys_rmse = phys_err.square().mean(dim=0).sqrt()
        delta_mae = (pred_delta - target_delta).abs().mean(dim=0)
        pred_delta_mean = pred_delta.mean(dim=0)
        target_delta_mean = target_delta.mean(dim=0)
        pred_delta_std = pred_delta.std(dim=0, unbiased=False)
        target_delta_std = target_delta.std(dim=0, unbiased=False)
        inside_axis = target_delta.abs() <= limits
        inside_rate = inside_axis.float().mean(dim=0)
        inside_all_mask = inside_axis.all(dim=1)
        inside_all = inside_all_mask.float().mean()
        if inside_all_mask.any():
            conditional_phys_err = phys_err[inside_all_mask]
            conditional_mae = conditional_phys_err.abs().mean(dim=0)
            conditional_rmse = conditional_phys_err.square().mean(dim=0).sqrt()
        else:
            conditional_phys_err = torch.empty(
                (0, 3),
                device=device,
                dtype=phys_err.dtype,
            )
            conditional_mae = torch.zeros(3, device=device, dtype=phys_err.dtype)
            conditional_rmse = torch.zeros(3, device=device, dtype=phys_err.dtype)
        coarse_bin = coarse_out["pred_bins"][mb, mp].long()
        coarse_err = coarse_bin - true_bin
        coarse_exact = (coarse_err == 0).float().mean(dim=0)
        coarse_w1 = (coarse_err.abs() <= 1).float().mean(dim=0)

        pred_centered = pred_delta - pred_delta_mean.view(1, 3)
        target_centered = target_delta - target_delta_mean.view(1, 3)
        delta_corr = (pred_centered * target_centered).mean(dim=0) / (
            pred_delta_std * target_delta_std
        ).clamp_min(1e-6)

        if "r_phys_delta" in ref_out:
            r_phys_delta = ref_out["r_phys_delta"][mb, mp]
            r_phys_delta_mae = (r_phys_delta - target_delta[:, 0]).abs().mean()
            r_phys_delta_std = r_phys_delta.std(unbiased=False)
            r_phys_delta_corr = ((r_phys_delta - r_phys_delta.mean()) * target_centered[:, 0]).mean() / (
                r_phys_delta_std * target_delta_std[0]
            ).clamp_min(1e-6)
        else:
            r_phys_delta_mae = zero.detach()
            r_phys_delta_std = zero.detach()
            r_phys_delta_corr = zero.detach()

        if "v_phys_delta" in ref_out:
            v_phys_delta = ref_out["v_phys_delta"][mb, mp]
            v_phys_delta_mae = (v_phys_delta - target_delta[:, 1]).abs().mean()
            v_phys_delta_std = v_phys_delta.std(unbiased=False)
            v_phys_delta_corr = ((v_phys_delta - v_phys_delta.mean()) * target_centered[:, 1]).mean() / (
                v_phys_delta_std * target_delta_std[1]
            ).clamp_min(1e-6)
        else:
            v_phys_delta_mae = zero.detach()
            v_phys_delta_std = zero.detach()
            v_phys_delta_corr = zero.detach()

        if direct_delta is not None:
            direct_mae = (direct_delta - target_delta).abs().mean(dim=0)
            direct_std = direct_delta.std(dim=0, unbiased=False)
            direct_centered = direct_delta - direct_delta.mean(dim=0, keepdim=True)
            direct_corr = (direct_centered * target_centered).mean(dim=0) / (
                direct_std * target_delta_std
            ).clamp_min(1e-6)
            direct_gates = ref_out.get("direct_delta_gates", torch.zeros(3, device=device, dtype=pred_delta.dtype)).to(device)
        else:
            direct_mae = torch.zeros(3, device=device, dtype=pred_delta.dtype)
            direct_corr = torch.zeros(3, device=device, dtype=pred_delta.dtype)
            direct_gates = torch.zeros(3, device=device, dtype=pred_delta.dtype)

        if range_phase_delta is not None:
            range_phase_delta_mae = (range_phase_delta - target_delta[:, 0]).abs().mean()
            range_phase_std = range_phase_delta.std(unbiased=False)
            range_phase_delta_corr = ((range_phase_delta - range_phase_delta.mean()) * target_centered[:, 0]).mean() / (
                range_phase_std * target_delta_std[0]
            ).clamp_min(1e-6)
            range_phase_gate = ref_out.get("range_phase_gate", zero.detach()).to(device)
        else:
            range_phase_delta_mae = zero.detach()
            range_phase_delta_corr = zero.detach()
            range_phase_gate = zero.detach()

    return loss, {
        "loss_heatmap": loss_heatmap.detach(),
        "loss_delta": loss_delta.detach(),
        "loss_phys_l1": loss_phys_l1.detach(),
        "loss_phys_l2": loss_phys_l2.detach(),
        "loss_phys_branch": loss_phys_branch.detach(),
        "loss_direct": loss_direct.detach(),
        "loss_range_phase": loss_range_phase.detach(),
        "r_acc": bin_acc[0].detach(),
        "v_acc": bin_acc[1].detach(),
        "theta_acc": bin_acc[2].detach(),
        "r_w1": bin_w1[0].detach(),
        "v_w1": bin_w1[1].detach(),
        "theta_w1": bin_w1[2].detach(),
        "mae_r": phys_mae[0].detach(),
        "mae_v": phys_mae[1].detach(),
        "mae_theta": phys_mae[2].detach(),
        "rmse_r": phys_rmse[0].detach(),
        "rmse_v": phys_rmse[1].detach(),
        "rmse_theta": phys_rmse[2].detach(),
        "delta_mae_r": delta_mae[0].detach(),
        "delta_mae_v": delta_mae[1].detach(),
        "delta_mae_theta": delta_mae[2].detach(),
        "pred_delta_mean_r": pred_delta_mean[0].detach(),
        "pred_delta_mean_v": pred_delta_mean[1].detach(),
        "pred_delta_mean_theta": pred_delta_mean[2].detach(),
        "pred_delta_std_r": pred_delta_std[0].detach(),
        "pred_delta_std_v": pred_delta_std[1].detach(),
        "pred_delta_std_theta": pred_delta_std[2].detach(),
        "target_delta_mean_r": target_delta_mean[0].detach(),
        "target_delta_mean_v": target_delta_mean[1].detach(),
        "target_delta_mean_theta": target_delta_mean[2].detach(),
        "target_delta_std_r": target_delta_std[0].detach(),
        "target_delta_std_v": target_delta_std[1].detach(),
        "target_delta_std_theta": target_delta_std[2].detach(),
        "delta_corr_r": delta_corr[0].detach(),
        "delta_corr_v": delta_corr[1].detach(),
        "delta_corr_theta": delta_corr[2].detach(),
        "r_phys_delta_mae": r_phys_delta_mae.detach(),
        "r_phys_delta_std": r_phys_delta_std.detach(),
        "r_phys_delta_corr": r_phys_delta_corr.detach(),
        "v_phys_delta_mae": v_phys_delta_mae.detach(),
        "v_phys_delta_std": v_phys_delta_std.detach(),
        "v_phys_delta_corr": v_phys_delta_corr.detach(),
        "direct_delta_mae_r": direct_mae[0].detach(),
        "direct_delta_mae_v": direct_mae[1].detach(),
        "direct_delta_mae_theta": direct_mae[2].detach(),
        "direct_delta_corr_r": direct_corr[0].detach(),
        "direct_delta_corr_v": direct_corr[1].detach(),
        "direct_delta_corr_theta": direct_corr[2].detach(),
        "direct_gate_r": direct_gates[0].detach(),
        "direct_gate_v": direct_gates[1].detach(),
        "direct_gate_theta": direct_gates[2].detach(),
        "range_phase_delta_mae": range_phase_delta_mae.detach(),
        "range_phase_delta_corr": range_phase_delta_corr.detach(),
        "range_phase_gate": range_phase_gate.detach(),
        "range_physical_gain": ref_out.get("range_physical_gain", zero.detach()).detach(),
        "velocity_physical_gain": ref_out.get("velocity_physical_gain", zero.detach()).detach(),
        "inside_r": inside_rate[0].detach(),
        "inside_v": inside_rate[1].detach(),
        "inside_theta": inside_rate[2].detach(),
        "inside_all": inside_all.detach(),
        "coarse_exact_r": coarse_exact[0].detach(),
        "coarse_exact_v": coarse_exact[1].detach(),
        "coarse_exact_theta": coarse_exact[2].detach(),
        "coarse_w1_r": coarse_w1[0].detach(),
        "coarse_w1_v": coarse_w1[1].detach(),
        "coarse_w1_theta": coarse_w1[2].detach(),
        "conditional_mae_r": conditional_mae[0].detach(),
        "conditional_mae_v": conditional_mae[1].detach(),
        "conditional_mae_theta": conditional_mae[2].detach(),
        "conditional_rmse_r": conditional_rmse[0].detach(),
        "conditional_rmse_v": conditional_rmse[1].detach(),
        "conditional_rmse_theta": conditional_rmse[2].detach(),
        "_pred_delta_values": pred_delta.detach().float().cpu(),
        "_target_delta_values": target_delta.detach().float().cpu(),
        "_phys_err_values": phys_err.detach().float().cpu(),
        "_inside_phys_err_values": conditional_phys_err.detach().float().cpu(),
        "_direct_delta_values": (
            direct_delta.detach().float().cpu()
            if direct_delta is not None
            else torch.empty((0, 3), dtype=torch.float32)
        ),
        "_r_phys_delta_values": (
            r_phys_delta.detach().float().cpu()
            if "r_phys_delta" in ref_out
            else torch.empty((0,), dtype=torch.float32)
        ),
        "_v_phys_delta_values": (
            v_phys_delta.detach().float().cpu()
            if "v_phys_delta" in ref_out
            else torch.empty((0,), dtype=torch.float32)
        ),
        "_range_phase_delta_values": (
            range_phase_delta.detach().float().cpu()
            if range_phase_delta is not None
            else torch.empty((0,), dtype=torch.float32)
        ),
        "num_obj": int(len(matched_b)),
    }


def init_sums():
    return {
        "loss": 0.0,
        "heatmap": 0.0,
        "delta": 0.0,
        "phys_l1": 0.0,
        "phys_l2": 0.0,
        "phys_branch": 0.0,
        "direct": 0.0,
        "range_phase": 0.0,
        "r_acc": 0.0,
        "v_acc": 0.0,
        "theta_acc": 0.0,
        "r_w1": 0.0,
        "v_w1": 0.0,
        "theta_w1": 0.0,
        "mae_r": 0.0,
        "mae_v": 0.0,
        "mae_theta": 0.0,
        "rmse_r": 0.0,
        "rmse_v": 0.0,
        "rmse_theta": 0.0,
        "delta_mae_r": 0.0,
        "delta_mae_v": 0.0,
        "delta_mae_theta": 0.0,
        "pred_delta_mean_r": 0.0,
        "pred_delta_mean_v": 0.0,
        "pred_delta_mean_theta": 0.0,
        "pred_delta_std_r": 0.0,
        "pred_delta_std_v": 0.0,
        "pred_delta_std_theta": 0.0,
        "target_delta_mean_r": 0.0,
        "target_delta_mean_v": 0.0,
        "target_delta_mean_theta": 0.0,
        "target_delta_std_r": 0.0,
        "target_delta_std_v": 0.0,
        "target_delta_std_theta": 0.0,
        "delta_corr_r": 0.0,
        "delta_corr_v": 0.0,
        "delta_corr_theta": 0.0,
        "r_phys_delta_mae": 0.0,
        "r_phys_delta_std": 0.0,
        "r_phys_delta_corr": 0.0,
        "v_phys_delta_mae": 0.0,
        "v_phys_delta_std": 0.0,
        "v_phys_delta_corr": 0.0,
        "direct_delta_mae_r": 0.0,
        "direct_delta_mae_v": 0.0,
        "direct_delta_mae_theta": 0.0,
        "direct_delta_corr_r": 0.0,
        "direct_delta_corr_v": 0.0,
        "direct_delta_corr_theta": 0.0,
        "direct_gate_r": 0.0,
        "direct_gate_v": 0.0,
        "direct_gate_theta": 0.0,
        "range_phase_delta_mae": 0.0,
        "range_phase_delta_corr": 0.0,
        "range_phase_gate": 0.0,
        "range_physical_gain": 0.0,
        "velocity_physical_gain": 0.0,
        "inside_r": 0.0,
        "inside_v": 0.0,
        "inside_theta": 0.0,
        "inside_all": 0.0,
        "coarse_exact_r": 0.0,
        "coarse_exact_v": 0.0,
        "coarse_exact_theta": 0.0,
        "coarse_w1_r": 0.0,
        "coarse_w1_v": 0.0,
        "coarse_w1_theta": 0.0,
        "conditional_mae_r": 0.0,
        "conditional_mae_v": 0.0,
        "conditional_mae_theta": 0.0,
        "conditional_rmse_r": 0.0,
        "conditional_rmse_v": 0.0,
        "conditional_rmse_theta": 0.0,
        "num_obj": 0.0,
        "_pred_delta_chunks": [],
        "_target_delta_chunks": [],
        "_phys_err_chunks": [],
        "_direct_delta_chunks": [],
        "_r_phys_delta_chunks": [],
        "_v_phys_delta_chunks": [],
        "_range_phase_delta_chunks": [],
        "_inside_phys_err_chunks": [],
    }


def update_sums(sums, loss, stats):
    weight = max(int(stats["num_obj"]), 1)
    sums["loss"] += tensor_to_float(loss) * weight
    sums["heatmap"] += tensor_to_float(stats["loss_heatmap"]) * weight
    sums["delta"] += tensor_to_float(stats["loss_delta"]) * weight
    sums["phys_l1"] += tensor_to_float(stats["loss_phys_l1"]) * weight
    sums["phys_l2"] += tensor_to_float(stats["loss_phys_l2"]) * weight
    sums["phys_branch"] += tensor_to_float(stats["loss_phys_branch"]) * weight
    sums["direct"] += tensor_to_float(stats["loss_direct"]) * weight
    sums["range_phase"] += tensor_to_float(stats["loss_range_phase"]) * weight
    for k in [
        "r_acc",
        "v_acc",
        "theta_acc",
        "r_w1",
        "v_w1",
        "theta_w1",
        "mae_r",
        "mae_v",
        "mae_theta",
        "rmse_r",
        "rmse_v",
        "rmse_theta",
        "delta_mae_r",
        "delta_mae_v",
        "delta_mae_theta",
        "pred_delta_mean_r",
        "pred_delta_mean_v",
        "pred_delta_mean_theta",
        "pred_delta_std_r",
        "pred_delta_std_v",
        "pred_delta_std_theta",
        "target_delta_mean_r",
        "target_delta_mean_v",
        "target_delta_mean_theta",
        "target_delta_std_r",
        "target_delta_std_v",
        "target_delta_std_theta",
        "delta_corr_r",
        "delta_corr_v",
        "delta_corr_theta",
        "r_phys_delta_mae",
        "r_phys_delta_std",
        "r_phys_delta_corr",
        "v_phys_delta_mae",
        "v_phys_delta_std",
        "v_phys_delta_corr",
        "direct_delta_mae_r",
        "direct_delta_mae_v",
        "direct_delta_mae_theta",
        "direct_delta_corr_r",
        "direct_delta_corr_v",
        "direct_delta_corr_theta",
        "direct_gate_r",
        "direct_gate_v",
        "direct_gate_theta",
        "range_phase_delta_mae",
        "range_phase_delta_corr",
        "range_phase_gate",
        "range_physical_gain",
        "velocity_physical_gain",
        "inside_r",
        "inside_v",
        "inside_theta",
        "inside_all",
        "coarse_exact_r",
        "coarse_exact_v",
        "coarse_exact_theta",
        "coarse_w1_r",
        "coarse_w1_v",
        "coarse_w1_theta",
        "conditional_mae_r",
        "conditional_mae_v",
        "conditional_mae_theta",
        "conditional_rmse_r",
        "conditional_rmse_v",
        "conditional_rmse_theta",
    ]:
        sums[k] += tensor_to_float(stats[k]) * weight
    sums["num_obj"] += int(stats["num_obj"])
    if int(stats["num_obj"]) > 0:
        sums["_pred_delta_chunks"].append(stats["_pred_delta_values"])
        sums["_target_delta_chunks"].append(stats["_target_delta_values"])
        sums["_phys_err_chunks"].append(stats["_phys_err_values"])
        if stats["_direct_delta_values"].numel() > 0:
            sums["_direct_delta_chunks"].append(stats["_direct_delta_values"])
        if stats["_r_phys_delta_values"].numel() > 0:
            sums["_r_phys_delta_chunks"].append(stats["_r_phys_delta_values"])
        if stats["_v_phys_delta_values"].numel() > 0:
            sums["_v_phys_delta_chunks"].append(stats["_v_phys_delta_values"])
        if stats["_range_phase_delta_values"].numel() > 0:
            sums["_range_phase_delta_chunks"].append(stats["_range_phase_delta_values"])
        if stats["_inside_phys_err_values"].numel() > 0:
            sums["_inside_phys_err_chunks"].append(stats["_inside_phys_err_values"])


def finalize_sums(sums):
    denom = max(sums["num_obj"], 1.0)
    out = {k: v / denom for k, v in sums.items() if k != "num_obj" and not k.startswith("_")}
    out["num_obj"] = sums["num_obj"]

    def corr_columns(pred, target):
        pred_centered = pred - pred.mean(dim=0, keepdim=True)
        target_centered = target - target.mean(dim=0, keepdim=True)
        pred_std = pred_centered.square().mean(dim=0).sqrt()
        target_std = target_centered.square().mean(dim=0).sqrt()
        corr = (pred_centered * target_centered).mean(dim=0) / (
            pred_std * target_std
        ).clamp_min(1e-12)
        return corr, pred_std, target_std

    if sums["_pred_delta_chunks"]:
        pred_delta = torch.cat(sums["_pred_delta_chunks"], dim=0)
        target_delta = torch.cat(sums["_target_delta_chunks"], dim=0)
        phys_err = torch.cat(sums["_phys_err_chunks"], dim=0)
        corr, pred_std, target_std = corr_columns(pred_delta, target_delta)

        for axis, name in enumerate(("r", "v", "theta")):
            out[f"mae_{name}"] = float(phys_err[:, axis].abs().mean())
            out[f"rmse_{name}"] = float(phys_err[:, axis].square().mean().sqrt())
            out[f"delta_mae_{name}"] = float((pred_delta[:, axis] - target_delta[:, axis]).abs().mean())
            out[f"pred_delta_mean_{name}"] = float(pred_delta[:, axis].mean())
            out[f"pred_delta_std_{name}"] = float(pred_std[axis])
            out[f"target_delta_mean_{name}"] = float(target_delta[:, axis].mean())
            out[f"target_delta_std_{name}"] = float(target_std[axis])
            out[f"delta_corr_{name}"] = float(corr[axis])

        if sums["_direct_delta_chunks"]:
            direct_delta = torch.cat(sums["_direct_delta_chunks"], dim=0)
            direct_corr, _, _ = corr_columns(direct_delta, target_delta)
            for axis, name in enumerate(("r", "v", "theta")):
                out[f"direct_delta_mae_{name}"] = float(
                    (direct_delta[:, axis] - target_delta[:, axis]).abs().mean()
                )
                out[f"direct_delta_corr_{name}"] = float(direct_corr[axis])

        if sums["_r_phys_delta_chunks"]:
            r_phys_delta = torch.cat(sums["_r_phys_delta_chunks"], dim=0)
            r_corr, r_std, _ = corr_columns(r_phys_delta[:, None], target_delta[:, 0:1])
            out["r_phys_delta_mae"] = float((r_phys_delta - target_delta[:, 0]).abs().mean())
            out["r_phys_delta_std"] = float(r_std[0])
            out["r_phys_delta_corr"] = float(r_corr[0])

        if sums["_v_phys_delta_chunks"]:
            v_phys_delta = torch.cat(sums["_v_phys_delta_chunks"], dim=0)
            v_corr, v_std, _ = corr_columns(v_phys_delta[:, None], target_delta[:, 1:2])
            out["v_phys_delta_mae"] = float((v_phys_delta - target_delta[:, 1]).abs().mean())
            out["v_phys_delta_std"] = float(v_std[0])
            out["v_phys_delta_corr"] = float(v_corr[0])

        if sums["_range_phase_delta_chunks"]:
            range_phase_delta = torch.cat(sums["_range_phase_delta_chunks"], dim=0)
            range_corr, _, _ = corr_columns(range_phase_delta[:, None], target_delta[:, 0:1])
            out["range_phase_delta_mae"] = float(
                (range_phase_delta - target_delta[:, 0]).abs().mean()
            )
            out["range_phase_delta_corr"] = float(range_corr[0])
        if sums["_inside_phys_err_chunks"]:
            inside_phys_err = torch.cat(sums["_inside_phys_err_chunks"], dim=0)
            for axis, name in enumerate(("r", "v", "theta")):
                out[f"conditional_mae_{name}"] = float(
                    inside_phys_err[:, axis].abs().mean()
                )
                out[f"conditional_rmse_{name}"] = float(
                    inside_phys_err[:, axis].square().mean().sqrt()
                )
    return out


def print_metrics(title, m):
    print(
        f"{title} | loss={m['loss']:.5f} hm={m['heatmap']:.4f} delta={m['delta']:.4f} "
        f"phys_l1={m['phys_l1']:.4f} branch={m['phys_branch']:.4f} direct={m['direct']:.4f} | "
        f"mae=({m['mae_r']:.3f},{m['mae_v']:.3f},{m['mae_theta']:.5f}) "
        f"rmse=({m['rmse_r']:.3f},{m['rmse_v']:.3f},{m['rmse_theta']:.5f}) | "
        f"delta_mae=({m['delta_mae_r']:.4f},{m['delta_mae_v']:.4f},{m['delta_mae_theta']:.4f}) | "
        f"direct=({m['direct_delta_mae_r']:.4f},{m['direct_delta_mae_v']:.4f},{m['direct_delta_mae_theta']:.4f};"
        f"corr={m['direct_delta_corr_r']:.2f},{m['direct_delta_corr_v']:.2f},{m['direct_delta_corr_theta']:.2f};"
        f"g={m['direct_gate_r']:.2f},{m['direct_gate_v']:.2f},{m['direct_gate_theta']:.2f}) | "
        f"phys_r=({m['r_phys_delta_mae']:.4f},corr={m['r_phys_delta_corr']:.3f},g={m['range_physical_gain']:.2f}) "
        f"phys_v=({m['v_phys_delta_mae']:.4f},corr={m['v_phys_delta_corr']:.3f},g={m['velocity_physical_gain']:.2f}) | "
        f"phase_r=({m['range_phase_delta_mae']:.4f},corr={m['range_phase_delta_corr']:.3f},g={m['range_phase_gate']:.2f}) | "
        f"inside=({m['inside_r']:.3f},{m['inside_v']:.3f},{m['inside_theta']:.3f};all={m['inside_all']:.3f}) "
        f"conditional_rmse=({m['conditional_rmse_r']:.3f},{m['conditional_rmse_v']:.3f},"
        f"{m['conditional_rmse_theta']:.5f}) "
        f"coarse_exact=({m['coarse_exact_r']:.3f},{m['coarse_exact_v']:.3f},{m['coarse_exact_theta']:.3f})"
    )


def select_k_indices(dataset, k_value, max_samples=None):
    indices = [i for i, k in enumerate(dataset.K_list_all) if int(k) == int(k_value)]
    if max_samples is not None and int(max_samples) > 0:
        indices = indices[: int(max_samples)]
    if not indices:
        raise RuntimeError(f"No samples found for K={k_value}")
    return indices


def select_positive_k_indices(dataset, max_samples=None, seed=0, balanced=False):
    groups = {}
    for index, k in enumerate(dataset.K_list_all):
        k = int(k)
        if k > 0:
            groups.setdefault(k, []).append(index)
    if not groups:
        raise RuntimeError("No positive-K samples found")

    rng = np.random.default_rng(int(seed))
    for indices in groups.values():
        rng.shuffle(indices)

    if max_samples is None or int(max_samples) <= 0:
        selected = [index for indices in groups.values() for index in indices]
    elif balanced:
        total = int(max_samples)
        ks = sorted(groups)
        base = total // len(ks)
        remainder = total % len(ks)
        selected = []
        for position, k in enumerate(ks):
            take = min(len(groups[k]), base + (1 if position < remainder else 0))
            selected.extend(groups[k][:take])
    else:
        selected = [index for indices in groups.values() for index in indices]
        rng.shuffle(selected)
        selected = selected[: int(max_samples)]

    rng.shuffle(selected)
    return selected


def describe_k_subset(dataset, indices):
    counts = {}
    for index in indices:
        k = int(dataset.K_list_all[index])
        counts[k] = counts.get(k, 0) + 1
    return dict(sorted(counts.items()))


class IndexedParamDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices):
        self.base_dataset = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.K_list_all = np.asarray(base_dataset.K_list_all)[self.indices]
        self.cfg = dict(getattr(base_dataset, "cfg", {}))
        self.P_h_ref = float(getattr(base_dataset, "P_h_ref", 1.0))

    def __len__(self):
        return int(len(self.indices))

    def __getitem__(self, index):
        return self.base_dataset[int(self.indices[int(index)])]


def split_dataset_indices(num_samples, train_ratio=0.8, seed=2026):
    indices = np.arange(int(num_samples), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(indices)
    split = int(round(len(indices) * float(train_ratio)))
    split = min(max(split, 1), len(indices) - 1)
    return indices[:split], indices[split:]


def summarize_pred_crop_offsets(
    offsets,
    k_values,
    candidate_spans=(0.75, 1.0, 1.5, 2.0, 2.5, 3.0),
    quantile=0.99,
    margin=0.15,
    round_step=0.25,
    min_span=(1.5, 1.5, 1.5),
    max_span=(3.0, 3.0, 3.0),
):
    offsets = np.asarray(offsets, dtype=np.float64)
    k_values = np.asarray(k_values, dtype=np.int64)
    if offsets.ndim != 2 or offsets.shape[1] != 3:
        raise ValueError(f"offsets must have shape [N, 3], got {offsets.shape}")
    if len(offsets) != len(k_values):
        raise ValueError("offsets and k_values must have the same length")
    if len(offsets) == 0:
        raise ValueError("cannot summarize an empty predicted-crop audit")

    axis_names = ("r", "v", "theta")
    rows = []
    groups = [(int(k), k_values == int(k)) for k in sorted(set(k_values.tolist()))]
    groups.append(("overall", np.ones(len(k_values), dtype=bool)))
    for group_name, mask in groups:
        values = np.abs(offsets[mask])
        row = {"K": group_name, "num_targets": int(mask.sum())}
        for axis, axis_name in enumerate(axis_names):
            for percentile in (50, 90, 95, 99):
                row[f"abs_offset_{axis_name}_p{percentile}"] = float(
                    np.quantile(values[:, axis], percentile / 100.0)
                )
            row[f"abs_offset_{axis_name}_max"] = float(values[:, axis].max())
        for span in candidate_spans:
            span = float(span)
            label = f"{span:g}"
            inside = values <= span
            for axis, axis_name in enumerate(axis_names):
                row[f"coverage_{axis_name}_span_{label}"] = float(inside[:, axis].mean())
            row[f"coverage_all_span_{label}"] = float(inside.all(axis=1).mean())
        rows.append(row)

    overall_abs = np.abs(offsets)
    recommended = []
    for axis in range(3):
        raw = float(np.quantile(overall_abs[:, axis], float(quantile))) + float(margin)
        rounded = math.ceil(raw / float(round_step) - 1e-12) * float(round_step)
        rounded = max(float(min_span[axis]), min(float(max_span[axis]), rounded))
        recommended.append(float(rounded))
    return rows, tuple(recommended)


@torch.no_grad()
def run_pred_crop_audit(coarse_model, loader, device, args):
    coarse_model.eval()
    offset_chunks = []
    integer_error_chunks = []
    k_chunks = []

    for batch in loader:
        X_h, X_l, K_true, Y_phys, Y_bin, Y_delta_norm = prepare_batch(
            batch,
            device,
            args,
            train=False,
        )
        coarse_out = run_coarse_teacher(coarse_model, X_h, X_l, K_true, args)
        pairs_list = build_matching_pairs(coarse_out, Y_phys, Y_bin, Y_delta_norm, args)
        for batch_index, pairs in enumerate(pairs_list):
            if not pairs:
                continue
            pred_indices = torch.tensor(
                [pair[0] for pair in pairs],
                device=device,
                dtype=torch.long,
            )
            target_indices = torch.tensor(
                [pair[1] for pair in pairs],
                device=device,
                dtype=torch.long,
            )
            pred_bins = coarse_out["pred_bins"][batch_index, pred_indices].float()
            true_bins = Y_bin[batch_index, target_indices].float()
            true_delta = Y_delta_norm[batch_index, target_indices].float()
            offset_chunks.append((true_bins + true_delta - pred_bins).detach().cpu())
            integer_error_chunks.append((pred_bins - true_bins).detach().cpu())
            k_chunks.append(
                torch.full(
                    (len(pairs),),
                    int(K_true[batch_index].item()),
                    dtype=torch.long,
                )
            )

    if not offset_chunks:
        raise RuntimeError("Predicted-crop audit found no matched targets")

    offsets = torch.cat(offset_chunks, dim=0).numpy()
    integer_errors = torch.cat(integer_error_chunks, dim=0).numpy()
    k_values = torch.cat(k_chunks, dim=0).numpy()
    rows, recommended = summarize_pred_crop_offsets(
        offsets,
        k_values,
        candidate_spans=getattr(
            args,
            "pred_crop_audit_candidate_spans",
            (0.75, 1.0, 1.5, 2.0, 2.5, 3.0),
        ),
        quantile=getattr(args, "pred_crop_auto_quantile", 0.99),
        margin=getattr(args, "pred_crop_auto_margin", 0.15),
        round_step=getattr(args, "pred_crop_auto_round_step", 0.25),
        min_span=getattr(args, "pred_crop_auto_min_span", args.patch_half_span_bins),
        max_span=getattr(args, "pred_crop_auto_max_span", (3.0, 3.0, 3.0)),
    )

    axis_names = ("r", "v", "theta")
    for row in rows:
        mask = np.ones(len(k_values), dtype=bool)
        if row["K"] != "overall":
            mask = k_values == int(row["K"])
        group_errors = np.abs(integer_errors[mask])
        for axis, axis_name in enumerate(axis_names):
            row[f"coarse_exact_{axis_name}"] = float(
                (group_errors[:, axis] == 0).mean()
            )
            row[f"coarse_w1_{axis_name}"] = float(
                (group_errors[:, axis] <= 1).mean()
            )
            row[f"coarse_abs_bin_error_{axis_name}_max"] = float(
                group_errors[:, axis].max()
            )
    return rows, recommended


def print_pred_crop_audit(rows, recommended):
    print("\n========== Stage D0 Predicted-Crop Audit ==========")
    for row in rows:
        print(
            f"K={row['K']} targets={row['num_targets']} "
            f"offset_p99=({row['abs_offset_r_p99']:.3f},"
            f"{row['abs_offset_v_p99']:.3f},"
            f"{row['abs_offset_theta_p99']:.3f}) "
            f"coarse_exact=({row['coarse_exact_r']:.3f},"
            f"{row['coarse_exact_v']:.3f},"
            f"{row['coarse_exact_theta']:.3f})"
        )
    print(
        "Recommended patch/axis half span (bin units):",
        tuple(float(v) for v in recommended),
    )
    print("===================================================")


def apply_runtime_refiner_geometry(refiner, args):
    spans = tuple(float(v) for v in args.patch_half_span_bins)
    limits = tuple(float(v) for v in args.axis_limits)
    refiner.axis_limits = limits
    refiner.patch_extractor.patch_half_span_bins = spans

    with torch.no_grad():
        refiner.r_axis.copy_(
            torch.linspace(-limits[0], limits[0], refiner.axis_bins[0]).to(
                refiner.r_axis
            )
        )
        refiner.v_axis.copy_(
            torch.linspace(-limits[1], limits[1], refiner.axis_bins[1]).to(
                refiner.v_axis
            )
        )
        refiner.theta_axis.copy_(
            torch.linspace(-limits[2], limits[2], refiner.axis_bins[2]).to(
                refiner.theta_axis
            )
        )
        patch_size = refiner.patch_extractor.patch_size
        refiner.axis_scales.copy_(
            torch.tensor(
                [
                    float(patch_size[0] - 1) / (2.0 * spans[0]),
                    float(patch_size[1] - 1) / (2.0 * spans[1]),
                    float(patch_size[2] - 1) / (2.0 * spans[2]),
                ],
                device=refiner.axis_scales.device,
                dtype=refiner.axis_scales.dtype,
            )
        )
        refiner.direct_offset_head.max_delta.copy_(
            torch.tensor(
                getattr(args, "direct_max_delta", limits),
                device=refiner.direct_offset_head.max_delta.device,
                dtype=refiner.direct_offset_head.max_delta.dtype,
            )
        )
    refiner.range_phase_delta_head.max_delta = float(limits[0])


def curriculum_stage(epoch, args):
    if epoch <= int(args.stage_a_k1_clean_epochs):
        return "A_k1_clean", epoch
    boundary_b = int(args.stage_a_k1_clean_epochs) + int(args.stage_b_k1_snr_epochs)
    if epoch <= boundary_b:
        return "B_k1_snr", epoch - int(args.stage_a_k1_clean_epochs)
    return getattr(args, "all_k_stage_label", "C_all_snr"), epoch - boundary_b


def linear_schedule(start, end, step, total_steps):
    if total_steps <= 1:
        return float(end)
    progress = min(max((int(step) - 1) / max(int(total_steps) - 1, 1), 0.0), 1.0)
    return float(start) + progress * (float(end) - float(start))


def apply_curriculum_args(args, stage, stage_epoch):
    args.current_epoch = int(stage_epoch)
    if stage == "A_k1_clean":
        args.train_use_awgn = False
        args.val_snr_db = None
        args.use_snr_curriculum = False
    elif stage == "B_k1_snr":
        args.train_use_awgn = True
        args.val_snr_db = linear_schedule(
            args.stage_b_val_snr_start,
            args.stage_b_val_snr_end,
            stage_epoch,
            args.stage_b_k1_snr_epochs,
        )
        args.train_snr_min = args.stage_b_train_snr_min
        args.train_snr_max = args.stage_b_train_snr_max
        args.snr_curriculum_start_min = args.stage_b_snr_curriculum_start_min
        args.snr_curriculum_epochs = args.stage_b_k1_snr_epochs
        args.use_snr_curriculum = True
    else:
        args.train_use_awgn = True
        args.val_snr_db = args.stage_c_val_snr_db
        args.train_snr_min = args.stage_c_train_snr_min
        args.train_snr_max = args.stage_c_train_snr_max
        args.snr_curriculum_start_min = args.stage_c_snr_curriculum_start_min
        args.snr_curriculum_epochs = args.stage_c_snr_curriculum_epochs
        args.use_snr_curriculum = True


def load_patch_denoiser_if_needed(refiner, args, device):
    ckpt_path = str(getattr(args, "patch_denoiser_ckpt", "") or "").strip()
    mode = str(getattr(args, "patch_denoiser_mode", "none")).lower()
    if not ckpt_path:
        print("Patch denoiser checkpoint: none")
        return
    print("Loading patch denoiser checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    missing, unexpected = refiner.patch_denoise.load_state_dict(state, strict=False)
    print("Patch denoiser loaded. missing:", missing, "unexpected:", unexpected)
    if mode == "freeze":
        for p in refiner.patch_denoise.parameters():
            p.requires_grad = False
        refiner.patch_denoise.eval()
        print("Patch denoiser mode: freeze")
    elif mode == "semifreeze":
        print(f"Patch denoiser mode: semifreeze, lr_scale={args.patch_denoiser_lr_scale}")
    elif mode in ("finetune", "none"):
        print("Patch denoiser mode: finetune")
    else:
        raise ValueError(f"Unknown patch_denoiser_mode: {mode}")


def load_refiner_init_checkpoint(refiner, args, device):
    ckpt_path = str(getattr(args, "init_refiner_ckpt", "") or "").strip()
    if not ckpt_path:
        print("Refiner initialization checkpoint: none (training from scratch)")
        return
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Refiner initialization checkpoint not found: {ckpt_path}")

    print("Loading refiner initialization checkpoint:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = refiner.load_state_dict(state, strict=True)
    source_epoch = ckpt.get("epoch", "unknown") if isinstance(ckpt, dict) else "unknown"
    source_metric = ckpt.get("best_metric", "unknown") if isinstance(ckpt, dict) else "unknown"
    print(
        "Refiner initialized.",
        f"source_epoch={source_epoch}",
        f"source_metric={source_metric}",
        f"missing={missing}",
        f"unexpected={unexpected}",
    )


def build_refiner_optimizer(refiner, args):
    mode = str(getattr(args, "patch_denoiser_mode", "none")).lower()
    ckpt_path = str(getattr(args, "patch_denoiser_ckpt", "") or "").strip()
    lr_scale = float(getattr(args, "patch_denoiser_lr_scale", 1.0))
    if ckpt_path and mode == "semifreeze" and lr_scale != 1.0:
        denoise_params = [p for p in refiner.patch_denoise.parameters() if p.requires_grad]
        denoise_ids = {id(p) for p in denoise_params}
        other_params = [p for p in refiner.parameters() if p.requires_grad and id(p) not in denoise_ids]
        return torch.optim.AdamW(
            [
                {"params": other_params, "lr": args.lr, "weight_decay": args.weight_decay},
                {"params": denoise_params, "lr": args.lr * lr_scale, "weight_decay": args.weight_decay},
            ]
        )
    return torch.optim.AdamW([p for p in refiner.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)


def run_epoch(refiner, coarse_model, loader, optimizer, scheduler, scaler, device, args, train, global_step=0):
    refiner.train(train)
    if train and str(getattr(args, "patch_denoiser_mode", "none")).lower() == "freeze":
        refiner.patch_denoise.eval()
    coarse_model.eval()
    sums = init_sums()
    if train:
        optimizer.zero_grad(set_to_none=True)

    for it, batch in enumerate(loader, 1):
        use_pair = bool(train and args.use_clean_noisy_pair and args.train_use_awgn)
        if use_pair:
            old_train_use_awgn = args.train_use_awgn
            args.train_use_awgn = False
            X_h_clean, X_l_clean, K_true, Y_phys, Y_bin, Y_delta_norm = prepare_batch(batch, device, args, train=True)
            args.train_use_awgn = old_train_use_awgn
            X_h, X_l, K_true, Y_phys, Y_bin, Y_delta_norm = prepare_batch(batch, device, args, train=True)
        else:
            X_h, X_l, K_true, Y_phys, Y_bin, Y_delta_norm = prepare_batch(batch, device, args, train=train)
        with torch.no_grad():
            coarse_out = run_coarse_configured(coarse_model, X_h, X_l, K_true, args)
            pairs_list = build_matching_pairs(coarse_out, Y_phys, Y_bin, Y_delta_norm, args)
            crop_center_phys, crop_teacher_prob = build_runtime_crop_centers_phys(
                coarse_out,
                Y_bin,
                pairs_list,
                args,
                train=train,
            )

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            with torch.cuda.amp.autocast(enabled=bool(args.use_amp and device.type == "cuda")):
                ref_out = refiner(
                    X_h,
                    X_l,
                    crop_center_phys,
                    coarse_delta=coarse_out["delta_norm"].detach(),
                    query_feat=coarse_out["query_feat"].detach(),
                )
                loss, stats = heatmap_refiner_loss(ref_out, crop_center_phys, coarse_out, Y_phys, Y_bin, Y_delta_norm, args)
                if use_pair:
                    clean_ref_out = refiner(
                        X_h_clean,
                        X_l_clean,
                        crop_center_phys,
                        coarse_delta=coarse_out["delta_norm"].detach(),
                        query_feat=coarse_out["query_feat"].detach(),
                    )
                    loss_clean, _ = heatmap_refiner_loss(clean_ref_out, crop_center_phys, coarse_out, Y_phys, Y_bin, Y_delta_norm, args)
                    loss_cons = F.smooth_l1_loss(
                        ref_out["delta_from_crop"],
                        clean_ref_out["delta_from_crop"].detach(),
                    )
                    if "range_phase_delta" in ref_out and "range_phase_delta" in clean_ref_out:
                        loss_cons = loss_cons + 0.5 * F.smooth_l1_loss(
                            ref_out["range_phase_delta"],
                            clean_ref_out["range_phase_delta"].detach(),
                        )
                    loss_feat = F.smooth_l1_loss(
                        ref_out["denoise_embed"],
                        clean_ref_out["denoise_embed"].detach(),
                    )
                    loss = (
                        loss
                        + args.lambda_clean_pair_loss * loss_clean
                        + args.lambda_clean_noisy_consistency * loss_cons
                        + args.lambda_denoise_feature_consistency * loss_feat
                    )

        if train:
            scaler.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(refiner.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1

            if args.use_swanlab and args.swanlab_log_every_batches > 0 and it % args.swanlab_log_every_batches == 0:
                log_swanlab(
                    {
                        "batch/loss": tensor_to_float(loss),
                        "batch/rmse_r": tensor_to_float(stats["rmse_r"]),
                        "batch/mae_r": tensor_to_float(stats["mae_r"]),
                        "batch/crop_teacher_prob": crop_teacher_prob,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )

            if it % args.print_every == 0:
                print(
                    f"  iter {it:04d}/{len(loader)} "
                    f"loss={tensor_to_float(loss):.5f} "
                    f"rmse_r={tensor_to_float(stats['rmse_r']):.3f} "
                    f"mae_r={tensor_to_float(stats['mae_r']):.3f}"
                )

        update_sums(sums, loss, stats)

    return finalize_sums(sums), global_step


def main(args):
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print("Using device:", device)
    print(
        "Crop mode:",
        getattr(args, "crop_center_mode", "teacher"),
        "| coarse count mode:",
        getattr(args, "coarse_count_mode", "teacher"),
    )
    print(
        "Noise power mode:",
        getattr(args, "noise_power_mode", "scene_power"),
        "| reference power:",
        getattr(args, "noise_power_ref", "per-sample scene power"),
    )

    if args.use_swanlab:
        if swanlab is None:
            print("WARNING: swanlab is not installed, disabling swanlab.")
            args.use_swanlab = False
        else:
            swanlab.init(
                project=args.swanlab_project,
                experiment_name=args.swanlab_experiment_name,
                config=args_to_config(args),
                mode="cloud",
            )

    if bool(getattr(args, "use_sionna_h5", False)):
        if not os.path.exists(args.sionna_h5):
            raise FileNotFoundError(
                f"Sionna HDF5 cache not found: {args.sionna_h5}\n"
                "Run: python data_tools/convert_sionna_npz_to_h5.py"
            )
        full_set = MultiTargetParamDataset(args.sionna_h5)
        train_indices, val_indices = split_dataset_indices(
            len(full_set),
            train_ratio=getattr(args, "train_ratio", 0.8),
            seed=getattr(args, "split_seed", args.seed),
        )
        train_set = IndexedParamDataset(full_set, train_indices)
        val_set = IndexedParamDataset(full_set, val_indices)
        args.P_h_ref = full_set.P_h_ref
        args.radar_params = {}
        print(f"Sionna HDF5 split: train={len(train_set)}, val={len(val_set)}")
    else:
        train_set = MultiTargetParamDataset(args.train_mat)
        val_set = MultiTargetParamDataset(args.test_mat)
        args.P_h_ref = train_set.P_h_ref
        args.radar_params = load_radar_params(args.train_mat)
    for k, v in train_set.cfg.items():
        if hasattr(args, k):
            setattr(args, k, v)

    train_k1_indices = select_k_indices(train_set, 1, args.stage_k1_train_samples)
    val_k1_indices = select_k_indices(val_set, 1, args.stage_k1_val_samples)
    train_all_indices = select_positive_k_indices(
        train_set,
        max_samples=getattr(args, "stage_all_train_samples", None),
        seed=args.seed + 301,
        balanced=bool(getattr(args, "stage_all_balance_k", False)),
    )
    val_all_indices = select_positive_k_indices(
        val_set,
        max_samples=getattr(args, "stage_all_val_samples", None),
        seed=args.seed + 302,
        balanced=bool(getattr(args, "stage_all_balance_k", False)),
    )
    print(f"Curriculum K=1 train samples: {len(train_k1_indices)}")
    print(f"Curriculum K=1 val samples  : {len(val_k1_indices)}")
    print(
        f"Curriculum K=1..{args.Kmax} train samples: {len(train_all_indices)} "
        f"{describe_k_subset(train_set, train_all_indices)}"
    )
    print(
        f"Curriculum K=1..{args.Kmax} val samples  : {len(val_all_indices)} "
        f"{describe_k_subset(val_set, val_all_indices)}"
    )
    print(
        "Curriculum:",
        f"A K=1 clean {args.stage_a_k1_clean_epochs} epochs,",
        f"B K=1 SNR {args.stage_b_k1_snr_epochs} epochs,",
        f"{getattr(args, 'all_k_stage_label', 'C')} all-K SNR "
        f"{max(args.epochs - args.stage_a_k1_clean_epochs - args.stage_b_k1_snr_epochs, 0)} epochs",
    )

    train_loader_all = make_loader(
        Subset(train_set, train_all_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        seed=args.seed,
        prefetch_factor=args.prefetch_factor,
    )
    train_loader_k1 = make_loader(
        Subset(train_set, train_k1_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        seed=args.seed + 101,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader_all = make_loader(
        Subset(val_set, val_all_indices),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        seed=args.seed + 999,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader_k1 = make_loader(
        Subset(val_set, val_k1_indices),
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        seed=args.seed + 1999,
        prefetch_factor=args.prefetch_factor,
    )

    coarse_model = build_coarse_model(args, device)
    if bool(getattr(args, "run_pred_crop_audit", False)):
        audit_seed = int(getattr(args, "pred_crop_audit_seed", args.seed + 4001))
        cuda_devices = []
        if device.type == "cuda":
            cuda_devices = [
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            ]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(audit_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(audit_seed)
            audit_rows, recommended_spans = run_pred_crop_audit(
                coarse_model,
                val_loader_all,
                device,
                args,
            )
        print_pred_crop_audit(audit_rows, recommended_spans)
        audit_path = os.path.join(args.save_dir, "pred_crop_audit.csv")
        save_history_csv(audit_rows, audit_path)
        print("Predicted-crop audit saved:", audit_path)

        if bool(getattr(args, "auto_configure_pred_crop_span", False)):
            args.patch_half_span_bins = tuple(float(v) for v in recommended_spans)
            args.axis_limits = tuple(float(v) for v in recommended_spans)
            args.direct_max_delta = tuple(float(v) for v in recommended_spans)
            print(
                "Auto-configured Stage D geometry:",
                f"patch_half_span_bins={args.patch_half_span_bins}",
                f"axis_limits={args.axis_limits}",
            )
        if bool(getattr(args, "pred_crop_audit_only", False)):
            print("D0-only mode requested; stopping before refiner training.")
            if args.use_swanlab:
                swanlab.finish()
            return

    refiner_cls = getattr(args, "refiner_class", AxisHeatmapPatchRefinerNet)
    refiner_kwargs = dict(
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
        feat_size=args.refiner_feat_size,
        patch_size=args.patch_size,
        patch_half_span_bins=args.patch_half_span_bins,
        axis_bins=args.axis_bins,
        axis_limits=args.axis_limits,
        query_dim=args.decoder_dim,
        channels=args.refiner_channels,
        hidden_dim=args.refiner_hidden_dim,
        dropout=args.refiner_dropout,
        softargmax_temp=args.softargmax_temp,
        use_physical_axis_branch=args.use_physical_axis_branch,
        use_full_patch_direct_branch=args.use_full_patch_direct_branch,
        use_range_phase_direct_branch=args.use_range_phase_direct_branch,
        direct_max_delta=args.direct_max_delta,
        patch_denoiser_arch=args.patch_denoiser_arch,
        patch_denoise_hidden=args.patch_denoise_hidden,
        patch_denoise_base_ch=args.patch_denoise_base_ch,
        patch_denoise_residual_scale=args.patch_denoise_residual_scale,
        use_attention_projection=bool(getattr(args, "use_attention_projection", False)),
        radar_params=args.radar_params,
        fft_mode=args.densefft_fft_mode,
    )
    refiner = refiner_cls(**refiner_kwargs).to(device)
    load_refiner_init_checkpoint(refiner, args, device)
    apply_runtime_refiner_geometry(refiner, args)
    load_patch_denoiser_if_needed(refiner, args, device)
    count_parameters(refiner)
    verify_densefft_crop_theory(refiner, coarse_model, val_loader_k1, device, args)
    save_densefft_visualizations(refiner, coarse_model, val_loader_k1, val_loader_all, device, args)

    optimizer = build_refiner_optimizer(refiner, args)
    scheduler = build_scheduler(optimizer, args, max(len(train_loader_all), len(train_loader_k1)))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.use_amp and device.type == "cuda"))

    best_rmse_r = float("inf")
    best_loss = float("inf")
    history = []
    global_step = 0
    best_rmse_path = os.path.join(args.save_dir, "patchrefiner_heatmap_best_rmse_r.pt")
    best_loss_path = os.path.join(args.save_dir, "patchrefiner_heatmap_best_loss.pt")
    last_path = os.path.join(args.save_dir, "patchrefiner_heatmap_last.pt")
    csv_path = os.path.join(
        args.save_dir,
        getattr(
            args,
            "history_filename",
            "patchrefiner_heatmap_teacher_history.csv",
        ),
    )

    for epoch in range(1, args.epochs + 1):
        stage, stage_epoch = curriculum_stage(epoch, args)
        apply_curriculum_args(args, stage, stage_epoch)
        train_loader = train_loader_k1 if stage in ("A_k1_clean", "B_k1_snr") else train_loader_all
        val_loader = val_loader_k1 if stage in ("A_k1_clean", "B_k1_snr") else val_loader_all
        print(
            f"\n========== Epoch {epoch}/{args.epochs} | stage={stage}({stage_epoch}) "
            f"| lr={optimizer.param_groups[0]['lr']:.3e} | train_awgn={args.train_use_awgn} "
            f"| val_snr={args.val_snr_db} =========="
        )
        train_m, global_step = run_epoch(refiner, coarse_model, train_loader, optimizer, scheduler, scaler, device, args, train=True, global_step=global_step)
        val_noise_seed = getattr(args, "val_noise_seed", None)
        if val_noise_seed is None:
            val_m, _ = run_epoch(
                refiner,
                coarse_model,
                val_loader,
                optimizer,
                scheduler,
                scaler,
                device,
                args,
                train=False,
                global_step=global_step,
            )
        else:
            cuda_devices = []
            if device.type == "cuda":
                cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(int(val_noise_seed))
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(int(val_noise_seed))
                val_m, _ = run_epoch(
                    refiner,
                    coarse_model,
                    val_loader,
                    optimizer,
                    scheduler,
                    scaler,
                    device,
                    args,
                    train=False,
                    global_step=global_step,
                )

        print_metrics(f"TRAIN epoch={epoch}", train_m)
        print_metrics(
            f"VAL {getattr(args, 'crop_center_mode', 'teacher')} epoch={epoch}",
            val_m,
        )

        row = {
            "epoch": epoch,
            "stage": stage,
            "stage_epoch": stage_epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_awgn": args.train_use_awgn,
            "val_snr_db": args.val_snr_db if args.val_snr_db is not None else "clean",
            "noise_power_mode": getattr(args, "noise_power_mode", "scene_power"),
            "noise_power_ref": getattr(args, "noise_power_ref", "scene"),
        }
        for k, v in train_m.items():
            row[f"train_{k}"] = v
        for k, v in val_m.items():
            row[f"val_{k}"] = v
        history.append(row)
        save_history_csv(history, csv_path)

        save_checkpoint(last_path, refiner, optimizer, scheduler, epoch, val_m["rmse_r"], args)
        if val_m["rmse_r"] < best_rmse_r:
            best_rmse_r = val_m["rmse_r"]
            save_checkpoint(best_rmse_path, refiner, optimizer, scheduler, epoch, best_rmse_r, args)
            print(f"Saved best RMSE-r heatmap refiner: epoch={epoch}, rmse_r={best_rmse_r:.5f}")

        if val_m["loss"] < best_loss:
            best_loss = val_m["loss"]
            save_checkpoint(best_loss_path, refiner, optimizer, scheduler, epoch, best_loss, args)
            print(f"Saved best loss heatmap refiner: epoch={epoch}, loss={best_loss:.5f}")

        if args.use_swanlab:
            log_dict = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
            log_dict.update({f"train/{k}": float(v) for k, v in train_m.items()})
            log_dict.update({f"val/{k}": float(v) for k, v in val_m.items()})
            log_swanlab(log_dict, step=global_step)

    print("\n========== Heatmap Teacher Refiner Training Finished ==========")
    print(f"Best val RMSE-r: {best_rmse_r:.5f}")
    print(f"Best val loss  : {best_loss:.5f}")
    print("History saved  :", csv_path)
    print("Best RMSE path :", best_rmse_path)
    print("Best loss path :", best_loss_path)
    print("Last path      :", last_path)

    if args.use_swanlab:
        swanlab.finish()


if __name__ == "__main__":
    class Args:
        _this_dir = os.path.dirname(os.path.abspath(__file__))

        train_mat = ""
        test_mat = ""
        save_dir = os.path.join(RELEASE_ROOT, "outputs", "refiner_stage_a")

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
        patch_denoiser_arch = "stem"  # stem / unet
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
        lambda_denoise_feature_consistency = 0.05
        patch_denoiser_ckpt = ""
        patch_denoiser_mode = "none"  # none / finetune / semifreeze / freeze
        patch_denoiser_lr_scale = 0.10

        epochs = 90
        batch_size = 1
        test_batch_size = 1
        lr = 8.0e-5
        min_lr = 6.0e-6
        lr_warmup_batches = 800
        weight_decay = 2.5e-3
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

        train_use_awgn = False
        train_snr_mode = "uniform"
        train_snr_min = -10
        train_snr_max = 20
        use_snr_curriculum = True
        snr_curriculum_start_min = 20
        snr_curriculum_epochs = 120
        train_snr_list = [-10, -8, -6, -4, -2, 0, 5, 10, 15, 20]
        val_snr_db = None
        P_h_ref = None

        stage_k1_train_samples = 4096
        stage_k1_val_samples = 1024
        stage_a_k1_clean_epochs = 90
        stage_b_k1_snr_epochs = 0
        stage_b_train_snr_min = -10
        stage_b_train_snr_max = 20
        stage_b_snr_curriculum_start_min = 20
        stage_b_val_snr_start = 20
        stage_b_val_snr_end = -10
        stage_c_train_snr_min = -12
        stage_c_train_snr_max = 0
        stage_c_snr_curriculum_start_min = -3
        stage_c_snr_curriculum_epochs = 60
        stage_c_val_snr_db = -10

        seed = 2031
        cpu = False
        num_workers = 8
        prefetch_factor = 4
        print_every = 100
        save_densefft_visuals = True
        verify_densefft_crop = True

        use_swanlab = True
        swanlab_project = "countnet-isac"
        swanlab_experiment_name = "patchrefiner_sionna_v1_stageA_teacher"
        swanlab_log_every_batches = 20

    main(Args())

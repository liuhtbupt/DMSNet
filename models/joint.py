import os
from types import SimpleNamespace

import torch
import torch.nn as nn

from .countnet import DualBandCrossFusionCountNet
from .paramnet import VelocityEnhancedParamNet
from .refiner import AxisHeatmapPatchRefinerNet


def load_checkpoint_state(path, map_location="cpu"):
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt


def ordinal_logits_to_class_log_prob(ord_logits):
    prob_gt = torch.sigmoid(ord_logits).clamp(1e-6, 1.0 - 1e-6)
    probs = []
    probs.append(1.0 - prob_gt[:, 0])
    for k in range(1, ord_logits.shape[1]):
        probs.append(prob_gt[:, k - 1] * (1.0 - prob_gt[:, k]))
    probs.append(prob_gt[:, -1])
    prob = torch.stack(probs, dim=1).clamp(1e-8, 1.0)
    return torch.log(prob)


def get_refined_count_logits(out, high_weight=0.45, k4_weight=0.0, ordinal_weight=0.0, high_min_prob=0.40):
    logits = out["K_logits"]
    if (
        ordinal_weight > 0
        and "K_ord_logits" in out
        and out["K_ord_logits"].shape[1] + 1 == logits.shape[1]
    ):
        logits = logits + ordinal_weight * ordinal_logits_to_class_log_prob(out["K_ord_logits"])

    refined = logits.clone()
    prob = torch.softmax(logits, dim=1)
    high_prob = prob[:, 3:6].sum(dim=1) if logits.shape[1] >= 6 else prob.new_zeros(prob.shape[0])
    high_gate = (torch.argmax(logits, dim=1) >= 3) | (high_prob >= high_min_prob)

    if high_weight > 0 and "K_high_logits" in out and logits.shape[1] >= 6 and torch.any(high_gate):
        refined[high_gate, 3:6] = refined[high_gate, 3:6] + high_weight * out["K_high_logits"][high_gate]

    if k4_weight > 0 and "K4_logit" in out and logits.shape[1] >= 5 and torch.any(high_gate):
        refined[high_gate, 4] = refined[high_gate, 4] + k4_weight * out["K4_logit"][high_gate]

    return refined


class JointDetectionParamEstimationNet(nn.Module):
    """
    Frozen end-to-end inference wrapper:
      1. CountNet estimates target count K.
      2. ParamNet predicts coarse bins for K selected object slots.
      3. Dense-FFT Refiner crops around the selected coarse-bin centers and outputs final R/V/theta.

    Missed detections and false alarms are handled at the count/selection level:
    only CountNet's predicted top-K slots are sent to the refiner. Parameter evaluation
    should then be conditional on matched predicted/true targets.
    """

    def __init__(
        self,
        count_ckpt=None,
        param_ckpt=None,
        refiner_ckpt=None,
        root_dir=None,
        load_weights=True,
        Kmax=5,
        Nbin_r=32,
        Nbin_v=32,
        Nbin_theta=32,
        r_min=50.0,
        r_max=300.0,
        v_min=-30.0,
        v_max=30.0,
        theta_min=-1.0471975511965976,
        theta_max=1.0471975511965976,
    ):
        super().__init__()
        self.Kmax = int(Kmax)
        self.Nbin_r = int(Nbin_r)
        self.Nbin_v = int(Nbin_v)
        self.Nbin_theta = int(Nbin_theta)
        self.register_buffer("mins", torch.tensor([r_min, v_min, theta_min], dtype=torch.float32))
        self.register_buffer(
            "bin_widths",
            torch.tensor(
                [
                    (r_max - r_min) / Nbin_r,
                    (v_max - v_min) / Nbin_v,
                    (theta_max - theta_min) / Nbin_theta,
                ],
                dtype=torch.float32,
            ),
        )

        root_dir = root_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        count_ckpt = count_ckpt or os.path.join(
            root_dir,
            "weights",
            "countnet_sionna_best.pt",
        )
        param_ckpt = param_ckpt or os.path.join(
            root_dir,
            "weights",
            "paramnet_sionna_best.pt",
        )
        refiner_ckpt = refiner_ckpt or os.path.join(
            root_dir,
            "weights",
            "refiner_sionna_stage_d_best.pt",
        )

        count_kwargs = dict(
            Kmax=Kmax,
            spec_size=(32, 32, 32),
            token_grid=(8, 8, 8),
            use_high=True,
            use_low=True,
            base_ch=32,
            embed_dim=128,
            num_heads=4,
            cross_attn_layers=1,
            fusion_attn_layers=0,
            mlp_ratio=2.0,
            dropout=0.25,
            attn_dropout=0.10,
            token_dropout=0.10,
            hidden_dim=256,
        )
        self.count_net = DualBandCrossFusionCountNet(**count_kwargs)
        self.param_net = VelocityEnhancedParamNet(
            Kmax=Kmax,
            Nbin_r=Nbin_r,
            Nbin_v=Nbin_v,
            Nbin_theta=Nbin_theta,
            r_min=r_min,
            r_max=r_max,
            v_min=v_min,
            v_max=v_max,
            theta_min=theta_min,
            theta_max=theta_max,
            count_kwargs=count_kwargs,
            count_model_path=None,
            freeze_count_backbone=True,
            spec_size=(32, 32, 32),
            decoder_dim=192,
            decoder_heads=6,
            decoder_layers=3,
            decoder_ffn_dim=768,
            hidden_dim=384,
            dropout=0.10,
            spectrum_base_ch=32,
            velocity_base_ch=24,
            velocity_dropout=0.14,
        )
        # The Sionna Stage-D checkpoint uses the refiner's default physical
        # radar parameters and registers no legacy MATLAB-only buffers.
        refiner_radar_params = {}
        self.refiner = AxisHeatmapPatchRefinerNet(
            Kmax=Kmax,
            Nbin_r=Nbin_r,
            Nbin_v=Nbin_v,
            Nbin_theta=Nbin_theta,
            r_min=r_min,
            r_max=r_max,
            v_min=v_min,
            v_max=v_max,
            theta_min=theta_min,
            theta_max=theta_max,
            feat_size=(256, 256, 256),
            patch_size=(25, 17, 15),
            patch_half_span_bins=(1.5, 1.5, 1.5),
            axis_bins=(257, 193, 161),
            axis_limits=(1.5, 1.5, 1.5),
            query_dim=192,
            channels=96,
            hidden_dim=224,
            dropout=0.12,
            softargmax_temp=0.055,
            use_physical_axis_branch=False,
            use_full_patch_direct_branch=True,
            use_range_phase_direct_branch=False,
            direct_max_delta=(1.5, 1.5, 1.5),
            patch_denoiser_arch="stem",
            patch_denoise_hidden=64,
            patch_denoise_base_ch=32,
            patch_denoise_residual_scale=0.35,
            use_attention_projection=False,
            radar_params=refiner_radar_params,
            fft_mode="current",
        )
        self.refiner.feat_size = (256, 256, 256)

        if load_weights:
            missing = [path for path in (count_ckpt, param_ckpt, refiner_ckpt) if not os.path.isfile(path)]
            if missing:
                formatted = "\n".join(f"  - {path}" for path in missing)
                raise FileNotFoundError(
                    "Pretrained checkpoints are missing:\n"
                    f"{formatted}\n"
                    "Download the release assets described in weights/README.md."
                )
            self.count_net.load_state_dict(load_checkpoint_state(count_ckpt), strict=True)
            self.param_net.load_state_dict(load_checkpoint_state(param_ckpt), strict=True)
            self.refiner.load_state_dict(load_checkpoint_state(refiner_ckpt), strict=True)

        self.count_refine = SimpleNamespace(
            high_weight=0.45,
            k4_weight=0.0,
            ordinal_weight=0.0,
            high_min_prob=0.40,
        )
        self.eval()

    def bin_centers_to_phys(self, bins):
        mins = self.mins.to(device=bins.device, dtype=torch.float32).view(1, 1, 3)
        widths = self.bin_widths.to(device=bins.device, dtype=torch.float32).view(1, 1, 3)
        return mins + (bins.to(torch.float32) + 0.5) * widths

    @staticmethod
    def _gather_slots(x, slot_idx):
        idx = slot_idx.unsqueeze(-1).expand(*slot_idx.shape, x.shape[-1])
        return torch.gather(x, dim=1, index=idx)

    def forward(self, X_h, X_l):
        count_out = self.count_net(X_h, X_l)
        count_logits = get_refined_count_logits(
            count_out,
            high_weight=self.count_refine.high_weight,
            k4_weight=self.count_refine.k4_weight,
            ordinal_weight=self.count_refine.ordinal_weight,
            high_min_prob=self.count_refine.high_min_prob,
        )
        count_prob = torch.softmax(count_logits, dim=1)
        K_pred = torch.argmax(count_logits, dim=1).clamp(0, self.Kmax)

        param_out = self.param_net(X_h, X_l, count_prior=count_prob, count_out=count_out)
        B = X_h.shape[0]
        device = X_h.device
        pred_phys = torch.zeros(B, self.Kmax, 3, device=device, dtype=X_h.dtype)
        pred_bins = torch.zeros(B, self.Kmax, 3, device=device, dtype=torch.long)
        pred_mask = torch.zeros(B, self.Kmax, device=device, dtype=torch.bool)
        selected_slots = torch.zeros(B, self.Kmax, device=device, dtype=torch.long)

        max_k = int(K_pred.max().item()) if B > 0 else 0
        if max_k > 0:
            obj_score = torch.sigmoid(param_out["p_logit"])
            slot_idx = torch.topk(obj_score, k=max_k, dim=1).indices
            sel_bins = self._gather_slots(param_out["pred_bins"], slot_idx)
            crop_center_phys = self.bin_centers_to_phys(sel_bins)
            coarse_delta = self._gather_slots(param_out["delta_norm"], slot_idx)
            query_feat = self._gather_slots(param_out["query_feat"], slot_idx)
            ref_out = self.refiner(
                X_h,
                X_l,
                crop_center_phys,
                coarse_delta=coarse_delta.detach(),
                query_feat=query_feat.detach(),
            )

            for b in range(B):
                k = int(K_pred[b].item())
                if k <= 0:
                    continue
                pred_phys[b, :k] = ref_out["pred_phys"][b, :k]
                pred_bins[b, :k] = ref_out["pred_bins"][b, :k]
                pred_mask[b, :k] = True
                selected_slots[b, :k] = slot_idx[b, :k]

        return {
            "K_pred": K_pred,
            "count_logits": count_logits,
            "count_prob": count_prob,
            "count_out": count_out,
            "param_out": param_out,
            "pred_phys": pred_phys,
            "pred_bins": pred_bins,
            "pred_mask": pred_mask,
            "selected_slots": selected_slots,
        }


def build_joint_model(device=None, **kwargs):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = JointDetectionParamEstimationNet(**kwargs).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model

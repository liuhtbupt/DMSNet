import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total / 1e6:.3f} M")
    print(f"Trainable parameters: {trainable / 1e6:.3f} M")
    return total, trainable


def load_countnet_class(count_model_path=None):
    if count_model_path is None or str(count_model_path).strip() == "":
        from .countnet import DualBandCrossFusionCountNet
        return DualBandCrossFusionCountNet

    path = Path(count_model_path)
    if not path.exists():
        raise FileNotFoundError(f"CountNet model file not found: {path}")

    spec = importlib.util.spec_from_file_location("best_countnet_module", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import CountNet model file: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "DualBandCrossFusionCountNet"):
        raise AttributeError(f"{path} does not define DualBandCrossFusionCountNet")

    return module.DualBandCrossFusionCountNet


def to_3tuple(x):
    if isinstance(x, int):
        return (x, x, x)
    if isinstance(x, (tuple, list)) and len(x) == 3:
        return tuple(int(v) for v in x)
    raise ValueError(f"Expected int or 3-tuple, got {x}")


def make_group_norm(num_channels, max_groups=8):
    num_groups = min(max_groups, num_channels)
    while num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


def make_mlp(in_dim, hidden_dim, out_dim, dropout=0.1):
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class FFTPowerSpectrumPreprocessor(nn.Module):
    """
    Converts raw complex high/low tensors into normalized 3D FFT power spectra.

    Input:
        X_h: [B, 2, 64, 112, 32]
        X_l: [B, 2, 64, 14, 4]

    Output:
        spec_h/spec_l: [B, 1, 32, 32, 32]
    """
    def __init__(self, out_size=(32, 32, 32), eps=1e-6):
        super().__init__()
        self.out_size = to_3tuple(out_size)
        self.eps = float(eps)

    @staticmethod
    def _complex_from_2ch(x):
        return torch.complex(x[:, 0].float(), x[:, 1].float())

    def _one_band(self, x):
        z = self._complex_from_2ch(x)

        z = torch.fft.ifft(z, dim=1)
        z = torch.fft.fftshift(z, dim=1)

        z = torch.fft.fft(z, dim=2)
        z = torch.fft.fftshift(z, dim=2)

        z = torch.fft.fft(z, dim=3)
        z = torch.fft.fftshift(z, dim=3)

        spec = (z.real ** 2 + z.imag ** 2).unsqueeze(1)
        spec = F.interpolate(
            spec,
            size=self.out_size,
            mode="trilinear",
            align_corners=False,
        )

        mean_power = spec.mean(dim=(2, 3, 4), keepdim=True).clamp_min(self.eps)
        spec = torch.log1p(spec / mean_power)

        mean = spec.mean(dim=(2, 3, 4), keepdim=True)
        std = spec.std(dim=(2, 3, 4), keepdim=True).clamp_min(self.eps)
        return (spec - mean) / std

    def forward(self, X_h, X_l):
        return self._one_band(X_h), self._one_band(X_l)


class FFTComplexFeaturePreprocessor(nn.Module):
    """
    Builds complex FFT feature volumes for the velocity branch.

    Channels per band:
        real, imag, log magnitude, sin phase, cos phase
    """
    def __init__(self, out_size=(32, 32, 32), eps=1e-6):
        super().__init__()
        self.out_size = to_3tuple(out_size)
        self.eps = float(eps)

    @staticmethod
    def _complex_from_2ch(x):
        return torch.complex(x[:, 0].float(), x[:, 1].float())

    def _fft3(self, x):
        z = self._complex_from_2ch(x)

        z = torch.fft.ifft(z, dim=1)
        z = torch.fft.fftshift(z, dim=1)

        z = torch.fft.fft(z, dim=2)
        z = torch.fft.fftshift(z, dim=2)

        z = torch.fft.fft(z, dim=3)
        z = torch.fft.fftshift(z, dim=3)
        return z

    def _one_band(self, x):
        z = self._fft3(x)
        mag = torch.sqrt(z.real ** 2 + z.imag ** 2 + self.eps)
        phase_sin = z.imag / mag.clamp_min(self.eps)
        phase_cos = z.real / mag.clamp_min(self.eps)

        rms = mag.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(self.eps)
        real = z.real / rms
        imag = z.imag / rms

        log_mag = torch.log1p(mag / mag.mean(dim=(1, 2, 3), keepdim=True).clamp_min(self.eps))
        log_mag = (log_mag - log_mag.mean(dim=(1, 2, 3), keepdim=True)) / (
            log_mag.std(dim=(1, 2, 3), keepdim=True).clamp_min(self.eps)
        )

        feat = torch.stack([real, imag, log_mag, phase_sin, phase_cos], dim=1)
        return F.interpolate(
            feat,
            size=self.out_size,
            mode="trilinear",
            align_corners=False,
        )

    def forward(self, X_h, X_l):
        return self._one_band(X_h), self._one_band(X_l)


class ResDownBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm1 = make_group_norm(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = make_group_norm(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

        if in_ch != out_ch or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                make_group_norm(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        s = self.skip(x)
        y = self.act(self.norm1(self.conv1(x)))
        y = self.drop(y)
        y = self.norm2(self.conv2(y))
        return self.act(y + s)


class SpectrumBandStem3D(nn.Module):
    def __init__(self, out_dim=192, base_ch=32, dropout=0.08):
        super().__init__()
        mid_dim = max(out_dim // 2, 64)
        self.net = nn.Sequential(
            nn.Conv3d(1, base_ch, 3, padding=1, bias=False),
            make_group_norm(base_ch),
            nn.GELU(),
            ResDownBlock3D(base_ch, 64, stride=2, dropout=dropout),
            ResDownBlock3D(64, mid_dim, stride=2, dropout=dropout),
            ResDownBlock3D(mid_dim, mid_dim, stride=1, dropout=dropout),
            nn.Conv3d(mid_dim, out_dim, 1, bias=True),
        )

    def forward(self, x):
        return self.net(x)


class DualBandSpectrumLocalizationEncoder(nn.Module):
    """
    Dedicated localization branch fed by normalized FFT power spectra.

    Output tokens are aligned to an 8x8x8 grid, which is intentionally parallel
    to CountNet's token grid but optimized for bin localization instead of count.
    """
    def __init__(self, decoder_dim=192, base_ch=32, dropout=0.08):
        super().__init__()
        self.stem_h = SpectrumBandStem3D(decoder_dim, base_ch=base_ch, dropout=dropout)
        self.stem_l = SpectrumBandStem3D(decoder_dim, base_ch=base_ch, dropout=dropout)
        self.fuse = nn.Sequential(
            nn.Conv3d(decoder_dim * 4, decoder_dim * 2, 1, bias=False),
            make_group_norm(decoder_dim * 2),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(decoder_dim * 2, decoder_dim, 1, bias=True),
        )

    @staticmethod
    def _tokenize(x):
        return x.flatten(2).transpose(1, 2)

    def forward(self, spec_h, spec_l):
        f_h = self.stem_h(spec_h)
        f_l = self.stem_l(spec_l)
        f = self.fuse(torch.cat([f_h, f_l, torch.abs(f_h - f_l), f_h * f_l], dim=1))
        return self._tokenize(f)


class VelocityBandStem3D(nn.Module):
    def __init__(self, in_ch=5, out_dim=192, base_ch=24, dropout=0.10):
        super().__init__()
        mid_dim = max(out_dim // 2, 64)
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, base_ch, kernel_size=(1, 5, 1), padding=(0, 2, 0), bias=False),
            make_group_norm(base_ch),
            nn.GELU(),
            nn.Conv3d(base_ch, base_ch, kernel_size=3, padding=1, bias=False),
            make_group_norm(base_ch),
            nn.GELU(),
            ResDownBlock3D(base_ch, 64, stride=2, dropout=dropout),
            ResDownBlock3D(64, mid_dim, stride=2, dropout=dropout),
            ResDownBlock3D(mid_dim, mid_dim, stride=1, dropout=dropout),
            nn.Conv3d(mid_dim, out_dim, 1, bias=True),
        )

    def forward(self, x):
        return self.net(x)


class DualBandComplexVelocityEncoder(nn.Module):
    """
    Velocity-oriented branch using complex FFT channels.

    The first convolution emphasizes the Doppler axis before the features are
    downsampled to the same 8x8x8 token grid as the global spectrum branch.
    """
    def __init__(self, decoder_dim=192, base_ch=24, dropout=0.10):
        super().__init__()
        self.stem_h = VelocityBandStem3D(5, decoder_dim, base_ch=base_ch, dropout=dropout)
        self.stem_l = VelocityBandStem3D(5, decoder_dim, base_ch=base_ch, dropout=dropout)
        self.fuse = nn.Sequential(
            nn.Conv3d(decoder_dim * 4, decoder_dim * 2, 1, bias=False),
            make_group_norm(decoder_dim * 2),
            nn.GELU(),
            nn.Dropout3d(dropout),
            nn.Conv3d(decoder_dim * 2, decoder_dim, 1, bias=True),
        )

    @staticmethod
    def _tokenize(x):
        return x.flatten(2).transpose(1, 2)

    def forward(self, vel_h, vel_l):
        f_h = self.stem_h(vel_h)
        f_l = self.stem_l(vel_l)
        f = self.fuse(torch.cat([f_h, f_l, torch.abs(f_h - f_l), f_h * f_l], dim=1))
        return self._tokenize(f)


class VelocityEnhancedParamNet(nn.Module):
    """
    Count-conditioned ParamNet with an additional spectrum localization branch.

    CountNet remains frozen and supplies:
        - p_K / K prior
        - global count context
        - count-trained fusion tokens

    The spectrum branch supplies localization-oriented tokens directly from the
    high/low 3D FFT power spectra.
    """
    def __init__(
        self,
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
        count_kwargs=None,
        count_model_path=None,
        freeze_count_backbone=True,
        spec_size=(32, 32, 32),
        decoder_dim=192,
        decoder_heads=6,
        decoder_layers=3,
        decoder_ffn_dim=768,
        hidden_dim=384,
        dropout=0.08,
        spectrum_base_ch=32,
        velocity_base_ch=24,
        velocity_dropout=0.12,
    ):
        super().__init__()

        self.Kmax = int(Kmax)
        self.Nbin_r = int(Nbin_r)
        self.Nbin_v = int(Nbin_v)
        self.Nbin_theta = int(Nbin_theta)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.theta_min = float(theta_min)
        self.theta_max = float(theta_max)

        count_kwargs = dict(count_kwargs or {})
        count_kwargs.setdefault("Kmax", self.Kmax)
        self.token_grid = tuple(int(v) for v in count_kwargs.get("token_grid", (8, 8, 8)))

        countnet_cls = load_countnet_class(count_model_path)
        self.count_net = countnet_cls(**count_kwargs)
        self.freeze_count_backbone = bool(freeze_count_backbone)
        if self.freeze_count_backbone:
            for p in self.count_net.parameters():
                p.requires_grad_(False)

        count_embed_dim = int(count_kwargs.get("embed_dim", 128))

        self.spec_pre = FFTPowerSpectrumPreprocessor(out_size=spec_size)
        self.spec_encoder = DualBandSpectrumLocalizationEncoder(
            decoder_dim=decoder_dim,
            base_ch=spectrum_base_ch,
            dropout=dropout,
        )
        self.velocity_pre = FFTComplexFeaturePreprocessor(out_size=spec_size)
        self.velocity_encoder = DualBandComplexVelocityEncoder(
            decoder_dim=decoder_dim,
            base_ch=velocity_base_ch,
            dropout=velocity_dropout,
        )

        self.count_memory_proj = nn.Sequential(
            nn.LayerNorm(count_embed_dim),
            nn.Linear(count_embed_dim, decoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.global_proj = nn.Sequential(
            nn.LayerNorm(count_embed_dim * 3),
            nn.Linear(count_embed_dim * 3, decoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.count_prior_proj = nn.Sequential(
            nn.LayerNorm(self.Kmax + 1),
            nn.Linear(self.Kmax + 1, decoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_dim, decoder_dim),
        )

        self.coord_proj = nn.Sequential(
            nn.Linear(3, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, decoder_dim),
        )
        self.source_embed_count = nn.Parameter(torch.randn(1, 1, decoder_dim) * 0.02)
        self.source_embed_spec = nn.Parameter(torch.randn(1, 1, decoder_dim) * 0.02)
        self.source_embed_velocity = nn.Parameter(torch.randn(1, 1, decoder_dim) * 0.02)
        self.global_embed = nn.Parameter(torch.randn(1, 1, decoder_dim) * 0.02)

        self.query_embed = nn.Parameter(torch.randn(1, self.Kmax, decoder_dim) * 0.02)
        self.query_index_embed = nn.Parameter(torch.randn(1, self.Kmax, decoder_dim) * 0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.out_norm = nn.LayerNorm(decoder_dim)
        self.velocity_attn = nn.MultiheadAttention(
            embed_dim=decoder_dim,
            num_heads=decoder_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.velocity_norm = nn.LayerNorm(decoder_dim)
        self.velocity_fuse = nn.Sequential(
            nn.LayerNorm(decoder_dim * 2),
            nn.Linear(decoder_dim * 2, decoder_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.obj_head = make_mlp(decoder_dim, hidden_dim, 1, dropout)
        self.r_head = make_mlp(decoder_dim, hidden_dim, self.Nbin_r, dropout)

        # Base velocity head: use the main decoder feature z
        self.v_head = make_mlp(decoder_dim, hidden_dim, self.Nbin_v, dropout)

        # Residual velocity head: use the velocity-enhanced feature z_v
        self.v_velocity_head = make_mlp(decoder_dim, hidden_dim, self.Nbin_v, dropout)

        self.theta_head = make_mlp(decoder_dim, hidden_dim, self.Nbin_theta, dropout)
        self.delta_head = make_mlp(decoder_dim, hidden_dim, 3, dropout)

        # Gated residual strength for velocity branch
        alpha_init = 0.05
        alpha_max = 0.40
        self.velocity_alpha_max = float(alpha_max)
        self.velocity_alpha_logit = nn.Parameter(
            torch.tensor(
                math.log(alpha_init / (alpha_max - alpha_init)),
                dtype=torch.float32,
            )
        )

        self.register_buffer("r_centers", self._make_centers(self.r_min, self.r_max, self.Nbin_r))
        self.register_buffer("v_centers", self._make_centers(self.v_min, self.v_max, self.Nbin_v))
        self.register_buffer("theta_centers", self._make_centers(self.theta_min, self.theta_max, self.Nbin_theta))
        self.register_buffer(
            "bin_widths",
            torch.tensor(
                [
                    (self.r_max - self.r_min) / self.Nbin_r,
                    (self.v_max - self.v_min) / self.Nbin_v,
                    (self.theta_max - self.theta_min) / self.Nbin_theta,
                ],
                dtype=torch.float32,
            ),
        )
        self.register_buffer("token_coords", self._make_token_coords(self.token_grid))

        self._init_param_modules()

    @staticmethod
    def _make_centers(min_val, max_val, nbin):
        width = (float(max_val) - float(min_val)) / int(nbin)
        return float(min_val) + (torch.arange(int(nbin)).float() + 0.5) * width

    @staticmethod
    def _make_token_coords(token_grid):
        axes = [torch.linspace(-1.0, 1.0, steps=int(n)) for n in token_grid]
        zz, yy, xx = torch.meshgrid(*axes, indexing="ij")
        return torch.stack([zz, yy, xx], dim=-1).view(1, -1, 3)

    def _init_param_modules(self):
        for name, module in self.named_modules():
            if name.startswith("count_net"):
                continue

            if isinstance(module, (nn.Conv3d, nn.Linear)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        for p in [
            self.source_embed_count,
            self.source_embed_spec,
            self.source_embed_velocity,
            self.global_embed,
            self.query_embed,
            self.query_index_embed,
        ]:
            nn.init.trunc_normal_(p, std=0.02)

    def load_count_checkpoint(self, ckpt_path, map_location="cpu", strict=True):
        ckpt = torch.load(ckpt_path, map_location=map_location)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        return self.count_net.load_state_dict(state, strict=strict)

    def _run_count_net(self, X_h, X_l):
        if self.freeze_count_backbone:
            self.count_net.eval()
            with torch.no_grad():
                return self.count_net(X_h, X_l)
        return self.count_net(X_h, X_l)

    def _decode_physical(self, r_logits, v_logits, theta_logits, delta_norm):
        r_bin = torch.argmax(r_logits, dim=-1)
        v_bin = torch.argmax(v_logits, dim=-1)
        theta_bin = torch.argmax(theta_logits, dim=-1)

        delta_phys = delta_norm * self.bin_widths.view(1, 1, 3).to(delta_norm)
        r = self.r_centers.to(delta_norm)[r_bin] + delta_phys[..., 0]
        v = self.v_centers.to(delta_norm)[v_bin] + delta_phys[..., 1]
        theta = self.theta_centers.to(delta_norm)[theta_bin] + delta_phys[..., 2]

        pred_bins = torch.stack([r_bin, v_bin, theta_bin], dim=-1)
        pred_phys = torch.stack([r, v, theta], dim=-1)
        return pred_bins, delta_phys, pred_phys

    def forward(self, X_h, X_l, count_prior=None, count_out=None):
        if count_out is None:
            count_out = self._run_count_net(X_h, X_l)

        count_logits = count_out["K_logits"]
        pred_count_prior = torch.softmax(count_logits, dim=1)
        if count_prior is None:
            count_prior = pred_count_prior
        else:
            count_prior = count_prior.to(device=count_logits.device, dtype=count_logits.dtype)

        coord = self.coord_proj(self.token_coords.to(device=X_h.device, dtype=count_logits.dtype))

        count_tokens = self.count_memory_proj(count_out["tokens"])
        count_tokens = count_tokens + coord + self.source_embed_count

        spec_h, spec_l = self.spec_pre(X_h, X_l)
        spec_tokens = self.spec_encoder(spec_h, spec_l)
        spec_tokens = spec_tokens + coord + self.source_embed_spec

        vel_h, vel_l = self.velocity_pre(X_h, X_l)
        velocity_tokens = self.velocity_encoder(vel_h, vel_l)
        velocity_tokens = velocity_tokens + coord + self.source_embed_velocity

        global_token = self.global_proj(count_out["feat"]).unsqueeze(1) + self.global_embed
        memory = torch.cat([global_token, count_tokens, spec_tokens], dim=1)

        B = X_h.shape[0]
        count_cond = self.count_prior_proj(count_prior).unsqueeze(1)
        query = self.query_embed.expand(B, -1, -1) + self.query_index_embed + count_cond

        z = self.decoder(tgt=query, memory=memory)
        z = self.out_norm(z)
        v_context, _ = self.velocity_attn(
            query=z,
            key=velocity_tokens,
            value=velocity_tokens,
            need_weights=False,
        )
        z_v = self.velocity_fuse(torch.cat([z, self.velocity_norm(v_context)], dim=-1))

        p_logit = self.obj_head(z).squeeze(-1)
        r_logits = self.r_head(z)

        # Base velocity prediction from the main decoder feature
        v_logits_base = self.v_head(z)

        # Residual velocity prediction from the velocity-enhanced feature
        v_logits_vel = self.v_velocity_head(z_v)

        # Learnable gated residual fusion
        velocity_alpha = self.velocity_alpha_max * torch.sigmoid(self.velocity_alpha_logit)
        v_logits = v_logits_base + velocity_alpha * v_logits_vel

        theta_logits = self.theta_head(z)
        delta_norm = 0.5 * torch.tanh(self.delta_head(z))

        pred_bins, delta_phys, pred_phys = self._decode_physical(
            r_logits,
            v_logits,
            theta_logits,
            delta_norm,
        )

        return {
            "p_logit": p_logit,
            "r_logits": r_logits,
            "v_logits": v_logits,
            "v_logits_base": v_logits_base,
            "v_logits_vel": v_logits_vel,
            "velocity_alpha": velocity_alpha,
            "theta_logits": theta_logits,
            "delta_norm": delta_norm,
            "delta_phys": delta_phys,
            "pred_bins": pred_bins,
            "pred_phys": pred_phys,
            "count_logits": count_logits,
            "count_prior": count_prior,
            "count_pred_prior": pred_count_prior,
            "K_pred": torch.argmax(count_logits, dim=1),
            "query_feat": z,
            "query_feat_velocity": z_v,
        }


SpectrumEnhancedParamNet = VelocityEnhancedParamNet


if __name__ == "__main__":
    count_kwargs = {
        "Kmax": 5,
        "spec_size": (32, 32, 32),
        "token_grid": (8, 8, 8),
        "use_high": True,
        "use_low": True,
        "base_ch": 32,
        "embed_dim": 128,
        "num_heads": 4,
        "cross_attn_layers": 1,
        "mlp_ratio": 2.0,
        "attn_dropout": 0.10,
        "token_dropout": 0.10,
        "hidden_dim": 256,
        "dropout": 0.25,
    }
    model = VelocityEnhancedParamNet(count_kwargs=count_kwargs)
    count_parameters(model)
    Xh = torch.randn(1, 2, 64, 112, 32)
    Xl = torch.randn(1, 2, 64, 14, 4)
    with torch.no_grad():
        out = model(Xh, Xl)
    for k, v in out.items():
        if torch.is_tensor(v):
            print(k, tuple(v.shape))

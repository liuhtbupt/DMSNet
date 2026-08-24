import torch
import torch.nn as nn
import torch.nn.functional as F


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total / 1e6:.3f} M")
    print(f"Trainable parameters: {trainable / 1e6:.3f} M")
    return total, trainable


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


class ComplexSpectrumPatchExtractor(nn.Module):
    """
    Build high/low dense complex 3D FFT features and crop local patches around bins.

    Channels per band:
        real, imag, log magnitude, sin phase, cos phase

    The feature grid is deliberately much finer than the coarse 32x32x32
    bin grid. Crop centers are supplied in coarse bin coordinates and scaled
    internally, so the patch still isolates the local target while exposing a
    denser spectral peak shape to the refiner.
    """
    def __init__(
        self,
        bin_size=(32, 32, 32),
        feat_size=(256, 256, 256),
        patch_size=(25, 17, 15),
        patch_half_span_bins=(1.5, 1.5, 1.5),
        r_min=50.0,
        r_max=300.0,
        v_min=-30.0,
        v_max=30.0,
        theta_min=-1.0471975511965976,
        theta_max=1.0471975511965976,
        radar_params=None,
        fft_mode="current",
        eps=1e-6,
    ):
        super().__init__()
        self.bin_size = to_3tuple(bin_size)
        self.feat_size = to_3tuple(feat_size)
        self.patch_size = to_3tuple(patch_size)
        self.patch_half_span_bins = tuple(float(v) for v in patch_half_span_bins)
        self.fft_mode = str(fft_mode)
        self.eps = float(eps)
        self.register_buffer("bin_widths", torch.tensor([
            (float(r_max) - float(r_min)) / float(self.bin_size[0]),
            (float(v_max) - float(v_min)) / float(self.bin_size[1]),
            (float(theta_max) - float(theta_min)) / float(self.bin_size[2]),
        ], dtype=torch.float32))

        radar_params = dict(radar_params or {})
        defaults = {
            "c0": 3.0e8,
            "delta_f_h": 240000.0,
            "delta_f_l": 30000.0,
            "lambda_h": 0.010714285714285714,
            "lambda_l": 0.08571428571428572,
            "Tsym_h": 5.208333333333333e-6,
            "Tsym_l": 4.1666666666666665e-5,
            "d_over_lambda_h": 0.5,
            "d_over_lambda_l": 0.5,
        }
        defaults.update(radar_params)
        for key, value in defaults.items():
            self.register_buffer(key, torch.tensor(float(value), dtype=torch.float32))

    @staticmethod
    def _complex_from_2ch(x):
        return torch.complex(x[:, 0].float(), x[:, 1].float())

    def _fft3(self, x):
        z = self._complex_from_2ch(x)
        # Dense FFT branch: use zero-padding to expose a finer local spectral
        # peak before cropping. This keeps the refinement problem in the FFT
        # patch domain rather than switching back to raw phase sequences.
        if self.fft_mode == "current":
            z = torch.fft.ifft(z, n=self.feat_size[0], dim=1)
            z = torch.fft.fftshift(z, dim=1)
        elif self.fft_mode == "range_ifft_no_shift":
            z = torch.fft.ifft(z, n=self.feat_size[0], dim=1)
        elif self.fft_mode == "all_fft":
            z = torch.fft.fft(z, n=self.feat_size[0], dim=1)
            z = torch.fft.fftshift(z, dim=1)
        else:
            raise ValueError(f"Unknown fft_mode: {self.fft_mode}")
        z = torch.fft.fft(z, n=self.feat_size[1], dim=2)
        z = torch.fft.fftshift(z, dim=2)
        z = torch.fft.fft(z, n=self.feat_size[2], dim=3)
        z = torch.fft.fftshift(z, dim=3)
        if tuple(z.shape[-3:]) != self.feat_size:
            raise RuntimeError(
                f"Dense FFT shape mismatch: got {tuple(z.shape[-3:])}, expected {self.feat_size}"
            )
        return z

    def _features_one_band(self, x):
        z = self._fft3(x)
        mag = torch.sqrt(z.real.square() + z.imag.square() + self.eps)
        mag_mean = mag.mean(dim=(1, 2, 3), keepdim=True).clamp_min(self.eps)
        phase_conf = (mag / (mag + mag_mean)).clamp(0.0, 1.0)
        phase_sin = (z.imag / mag.clamp_min(self.eps)) * phase_conf
        phase_cos = (z.real / mag.clamp_min(self.eps)) * phase_conf

        rms = mag.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(self.eps)
        real = z.real / rms
        imag = z.imag / rms

        log_mag = torch.log1p(mag / mag_mean)
        log_mag = (log_mag - log_mag.mean(dim=(1, 2, 3), keepdim=True)) / (
            log_mag.std(dim=(1, 2, 3), keepdim=True).clamp_min(self.eps)
        )

        feat = torch.stack([real, imag, log_mag, phase_sin, phase_cos], dim=1)
        return F.interpolate(feat, size=self.feat_size, mode="trilinear", align_corners=False)

    def _physical_centers_to_indices(self, centers_phys, band):
        centers_phys = centers_phys.float()
        r, v, theta = centers_phys.unbind(dim=-1)
        suffix = "h" if band == "high" else "l"
        c0 = self.c0.to(centers_phys)
        delta_f = getattr(self, f"delta_f_{suffix}").to(centers_phys)
        lamb = getattr(self, f"lambda_{suffix}").to(centers_phys)
        tsym = getattr(self, f"Tsym_{suffix}").to(centers_phys)
        d_over_lambda = getattr(self, f"d_over_lambda_{suffix}").to(centers_phys)

        nr, nv, nt = [float(x) for x in self.feat_size]
        r_unamb = c0 / (2.0 * delta_f)
        r_raw = r / r_unamb * nr
        if self.fft_mode == "current":
            r_idx = torch.remainder(r_raw + 0.5 * nr, nr)
        elif self.fft_mode == "range_ifft_no_shift":
            r_idx = r_raw
        elif self.fft_mode == "all_fft":
            r_idx = torch.remainder(0.5 * nr - r_raw, nr)
        else:
            raise ValueError(f"Unknown fft_mode: {self.fft_mode}")

        v_idx = 0.5 * nv + (2.0 * v / lamb) * tsym * nv
        theta_idx = 0.5 * nt + d_over_lambda * torch.sin(theta) * nt
        out = torch.stack([r_idx, v_idx, theta_idx], dim=-1)
        max_idx = torch.tensor([self.feat_size[0] - 1, self.feat_size[1] - 1, self.feat_size[2] - 1], device=out.device, dtype=out.dtype)
        return torch.maximum(torch.minimum(out, max_idx), torch.zeros_like(max_idx))

    def _crop(self, feat, centers_phys, band):
        B, C, R, V, T = feat.shape
        _, P, _ = centers_phys.shape
        pr, pv, pt = self.patch_size
        half_span_phys = self.bin_widths.to(centers_phys) * torch.tensor(
            self.patch_half_span_bins,
            device=centers_phys.device,
            dtype=centers_phys.dtype,
        )
        r_off = torch.linspace(-1.0, 1.0, pr, device=feat.device, dtype=centers_phys.dtype) * half_span_phys[0]
        v_off = torch.linspace(-1.0, 1.0, pv, device=feat.device, dtype=centers_phys.dtype) * half_span_phys[1]
        t_off = torch.linspace(-1.0, 1.0, pt, device=feat.device, dtype=centers_phys.dtype) * half_span_phys[2]
        rr, vv, tt = torch.meshgrid(r_off, v_off, t_off, indexing="ij")
        offsets = torch.stack([rr, vv, tt], dim=-1)
        sample_phys = centers_phys[:, :, None, None, None, :] + offsets.view(1, 1, pr, pv, pt, 3)
        sample_idx = self._physical_centers_to_indices(sample_phys, band)

        grid_r = 2.0 * sample_idx[..., 0] / max(float(R - 1), 1.0) - 1.0
        grid_v = 2.0 * sample_idx[..., 1] / max(float(V - 1), 1.0) - 1.0
        grid_t = 2.0 * sample_idx[..., 2] / max(float(T - 1), 1.0) - 1.0
        grid = torch.stack([grid_t, grid_v, grid_r], dim=-1).reshape(B * P, pr, pv, pt, 3)

        feat_bp = feat[:, None].expand(B, P, C, R, V, T).reshape(B * P, C, R, V, T)
        patches = F.grid_sample(
            feat_bp,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return patches.reshape(B, P, C, pr, pv, pt)

    def _complex_patch_features(self, z_patch):
        mag = torch.sqrt(z_patch.real.square() + z_patch.imag.square() + self.eps)
        mag_mean = mag.mean(dim=(2, 3, 4), keepdim=True).clamp_min(self.eps)
        phase_conf = (mag / (mag + mag_mean)).clamp(0.0, 1.0)
        phase_sin = (z_patch.imag / mag.clamp_min(self.eps)) * phase_conf
        phase_cos = (z_patch.real / mag.clamp_min(self.eps)) * phase_conf

        rms = mag.square().mean(dim=(2, 3, 4), keepdim=True).sqrt().clamp_min(self.eps)
        real = z_patch.real / rms
        imag = z_patch.imag / rms

        log_mag = torch.log1p(mag / mag_mean)
        log_mag = (log_mag - log_mag.mean(dim=(2, 3, 4), keepdim=True)) / (
            log_mag.std(dim=(2, 3, 4), keepdim=True).clamp_min(self.eps)
        )
        return torch.stack([real, imag, log_mag, phase_sin, phase_cos], dim=2)

    def forward(self, X_h, X_l, centers_phys):
        z_h = self._fft3(X_h)
        patch_h_complex = torch.complex(
            self._crop(z_h.real[:, None], centers_phys, "high").squeeze(2),
            self._crop(z_h.imag[:, None], centers_phys, "high").squeeze(2),
        )
        del z_h
        z_l = self._fft3(X_l)
        patch_l_complex = torch.complex(
            self._crop(z_l.real[:, None], centers_phys, "low").squeeze(2),
            self._crop(z_l.imag[:, None], centers_phys, "low").squeeze(2),
        )
        del z_l
        patch_h = self._complex_patch_features(patch_h_complex)
        patch_l = self._complex_patch_features(patch_l_complex)
        return torch.cat([patch_h, patch_l], dim=2)


class AxisHeatmapHead(nn.Module):
    def __init__(self, channels, out_bins):
        super().__init__()
        self.out_bins = int(out_bins)
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1, bias=False),
            make_group_norm(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels // 2, 3, padding=1, bias=False),
            make_group_norm(channels // 2),
            nn.GELU(),
            nn.Conv1d(channels // 2, 1, 1),
        )

    def forward(self, x):
        if x.shape[-1] != self.out_bins:
            x = F.interpolate(x, size=self.out_bins, mode="linear", align_corners=False)
        return self.net(x).squeeze(1)


class PhysicalSequenceHead(nn.Module):
    def __init__(self, out_bins, base_ch=64, pooled_len=16, dropout=0.10):
        super().__init__()
        self.out_bins = int(out_bins)
        self.net = nn.Sequential(
            nn.Conv1d(8, base_ch, 5, padding=2, bias=False),
            make_group_norm(base_ch),
            nn.GELU(),
            nn.Conv1d(base_ch, base_ch * 2, 5, padding=2, bias=False),
            make_group_norm(base_ch * 2),
            nn.GELU(),
            nn.Conv1d(base_ch * 2, base_ch * 2, 3, padding=1, bias=False),
            make_group_norm(base_ch * 2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(pooled_len),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_ch * 2 * pooled_len, base_ch * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(base_ch * 4, self.out_bins),
        )

    def forward(self, x):
        return self.net(x)


class PhaseDeltaHead(nn.Module):
    def __init__(self, context_dim, base_ch=96, pooled_len=16, dropout=0.08, max_delta=0.75):
        super().__init__()
        self.max_delta = float(max_delta)
        self.seq_net = nn.Sequential(
            nn.Conv1d(8, base_ch, 5, padding=2, bias=False),
            make_group_norm(base_ch),
            nn.GELU(),
            ResidualBlock1D(base_ch, dropout=dropout),
            nn.Conv1d(base_ch, base_ch * 2, 5, padding=2, bias=False),
            make_group_norm(base_ch * 2),
            nn.GELU(),
            ResidualBlock1D(base_ch * 2, dropout=dropout),
            nn.AdaptiveAvgPool1d(pooled_len),
            nn.Flatten(),
        )
        in_dim = base_ch * 2 * pooled_len + context_dim
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, base_ch * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(base_ch * 4, base_ch * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(base_ch * 2, 1),
        )

    def forward(self, seq_feat, context):
        x = self.seq_net(seq_feat)
        delta = self.head(torch.cat([x, context], dim=1)).squeeze(-1)
        return torch.tanh(delta) * self.max_delta


class ResidualBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=5, dilation=1, dropout=0.0):
        super().__init__()
        padding = (kernel_size // 2) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.norm1 = make_group_norm(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.norm2 = make_group_norm(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        y = self.act(self.norm1(self.conv1(x)))
        y = self.drop(y)
        y = self.norm2(self.conv2(y))
        return self.act(x + y)


class ResidualBlock3D(nn.Module):
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.norm1 = make_group_norm(channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = make_group_norm(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        y = self.act(self.norm1(self.conv1(x)))
        y = self.drop(y)
        y = self.norm2(self.conv2(y))
        return self.act(x + y)


class PatchDenoiseStem(nn.Module):
    def __init__(self, channels=10, hidden=48, dropout=0.05, residual_scale=0.35):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.in_proj = nn.Sequential(
            nn.Conv3d(channels, hidden, 3, padding=1, bias=False),
            make_group_norm(hidden),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualBlock3D(hidden, dropout=dropout),
            ResidualBlock3D(hidden, dropout=dropout),
            ResidualBlock3D(hidden, dropout=dropout),
        )
        self.out_proj = nn.Conv3d(hidden, channels, 3, padding=1)
        self.embed_proj = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.LayerNorm(hidden),
        )

    def forward(self, x):
        feat = self.blocks(self.in_proj(x))
        residual = torch.tanh(self.out_proj(feat)) * self.residual_scale
        denoised = x + residual
        embed = self.embed_proj(feat)
        return denoised, embed


class DenoiseConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            make_group_norm(out_ch),
            nn.GELU(),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            make_group_norm(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class PatchUNetDenoiseStem(nn.Module):
    """
    Stronger patch denoiser for pretraining.

    It keeps the same interface as PatchDenoiseStem:
        denoised_patch, embed = model(noisy_patch)

    The final projection is initialized to zero, so training starts from an
    identity denoiser and learns a bounded residual correction.
    """
    def __init__(self, channels=10, base_ch=32, dropout=0.05, residual_scale=0.45):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.enc0 = DenoiseConvBlock3D(channels, base_ch, dropout=dropout)
        self.down1 = nn.Sequential(
            nn.Conv3d(base_ch, base_ch * 2, 3, stride=2, padding=1, bias=False),
            make_group_norm(base_ch * 2),
            nn.GELU(),
        )
        self.enc1 = DenoiseConvBlock3D(base_ch * 2, base_ch * 2, dropout=dropout)
        self.down2 = nn.Sequential(
            nn.Conv3d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1, bias=False),
            make_group_norm(base_ch * 4),
            nn.GELU(),
        )
        self.bottleneck = nn.Sequential(
            DenoiseConvBlock3D(base_ch * 4, base_ch * 4, dropout=dropout),
            ResidualBlock3D(base_ch * 4, dropout=dropout),
        )
        self.dec1 = DenoiseConvBlock3D(base_ch * 6, base_ch * 2, dropout=dropout)
        self.dec0 = DenoiseConvBlock3D(base_ch * 3, base_ch, dropout=dropout)
        self.out_proj = nn.Conv3d(base_ch, channels, 3, padding=1)
        self.embed_proj = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.LayerNorm(base_ch * 4),
        )
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(self.down1(e0))
        b = self.bottleneck(self.down2(e1))
        u1 = F.interpolate(b, size=e1.shape[-3:], mode="trilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        u0 = F.interpolate(d1, size=e0.shape[-3:], mode="trilinear", align_corners=False)
        d0 = self.dec0(torch.cat([u0, e0], dim=1))
        residual = torch.tanh(self.out_proj(d0)) * self.residual_scale
        denoised = x + residual
        embed = self.embed_proj(b)
        return denoised, embed


class FullPatchDirectOffsetHead(nn.Module):
    def __init__(self, channels, context_dim, hidden_dim=256, dropout=0.10, max_delta=(0.75, 0.50, 0.50)):
        super().__init__()
        self.register_buffer("max_delta", torch.tensor(max_delta, dtype=torch.float32))
        in_dim = channels * 3 + context_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, feat, context):
        pooled = torch.cat(
            [
                feat.mean(dim=(2, 3, 4)),
                feat.amax(dim=(2, 3, 4)),
                feat.std(dim=(2, 3, 4), unbiased=False),
            ],
            dim=1,
        )
        raw = self.net(torch.cat([pooled, context], dim=1))
        return torch.tanh(raw) * self.max_delta.to(raw).view(1, 3)


class AxisHeatmapPatchRefinerNet(nn.Module):
    """
    Teacher-crop oriented patch refiner.

    It predicts three separable local offset distributions and decodes the
    continuous offset with soft-argmax. Offsets are in coarse-bin units.
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
        feat_size=(256, 256, 256),
        patch_size=(25, 17, 15),
        patch_half_span_bins=(1.5, 1.5, 1.5),
        axis_bins=(257, 193, 161),
        axis_limits=(0.75, 0.75, 0.75),
        query_dim=192,
        channels=128,
        hidden_dim=256,
        dropout=0.12,
        softargmax_temp=0.08,
        use_physical_axis_branch=True,
        use_full_patch_direct_branch=True,
        use_range_phase_direct_branch=False,
        direct_max_delta=(0.75, 0.50, 0.50),
        patch_denoiser_arch="stem",
        patch_denoise_hidden=64,
        patch_denoise_base_ch=32,
        patch_denoise_residual_scale=0.35,
        use_attention_projection=False,
        radar_params=None,
        fft_mode="current",
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
        self.axis_bins = to_3tuple(axis_bins)
        self.axis_limits = tuple(float(v) for v in axis_limits)
        self.softargmax_temp = float(softargmax_temp)
        self.use_physical_axis_branch = bool(use_physical_axis_branch)
        self.use_full_patch_direct_branch = bool(use_full_patch_direct_branch)
        self.use_range_phase_direct_branch = bool(use_range_phase_direct_branch)
        self.use_attention_projection = bool(use_attention_projection)
        self.patch_denoiser_arch = str(patch_denoiser_arch).lower()

        self.patch_extractor = ComplexSpectrumPatchExtractor(
            bin_size=(self.Nbin_r, self.Nbin_v, self.Nbin_theta),
            feat_size=feat_size,
            patch_size=patch_size,
            patch_half_span_bins=patch_half_span_bins,
            r_min=self.r_min,
            r_max=self.r_max,
            v_min=self.v_min,
            v_max=self.v_max,
            theta_min=self.theta_min,
            theta_max=self.theta_max,
            radar_params=radar_params,
            fft_mode=fft_mode,
        )
        if self.patch_denoiser_arch == "stem":
            self.patch_denoise = PatchDenoiseStem(
                channels=10,
                hidden=patch_denoise_hidden,
                dropout=dropout * 0.5,
                residual_scale=patch_denoise_residual_scale,
            )
        elif self.patch_denoiser_arch == "unet":
            self.patch_denoise = PatchUNetDenoiseStem(
                channels=10,
                base_ch=patch_denoise_base_ch,
                dropout=dropout * 0.5,
                residual_scale=patch_denoise_residual_scale,
            )
        else:
            raise ValueError(f"Unknown patch_denoiser_arch: {patch_denoiser_arch}")

        self.backbone = nn.Sequential(
            nn.Conv3d(10, 48, 3, padding=1, bias=False),
            make_group_norm(48),
            nn.GELU(),
            nn.Conv3d(48, 96, 3, padding=1, bias=False),
            make_group_norm(96),
            nn.GELU(),
            nn.Conv3d(96, channels, 3, padding=1, bias=False),
            make_group_norm(channels),
            nn.GELU(),
            ResidualBlock3D(channels, dropout=dropout * 0.5),
            ResidualBlock3D(channels, dropout=dropout * 0.5),
            nn.Dropout3d(dropout),
        )

        self.query_proj = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.meta_proj = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, hidden_dim),
            nn.GELU(),
        )
        self.context_to_bias = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, channels),
            nn.GELU(),
        )
        self.projection_attention = (
            nn.Conv3d(channels, 1, kernel_size=1)
            if self.use_attention_projection
            else None
        )

        axis_profile_ch = channels * 2
        self.r_head = AxisHeatmapHead(axis_profile_ch, self.axis_bins[0])
        self.v_head = AxisHeatmapHead(axis_profile_ch, self.axis_bins[1])
        self.theta_head = AxisHeatmapHead(axis_profile_ch, self.axis_bins[2])
        self.direct_offset_head = FullPatchDirectOffsetHead(
            channels=channels,
            context_dim=hidden_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            max_delta=direct_max_delta,
        )
        self.direct_delta_gate_logits = nn.Parameter(torch.tensor([2.0, -2.0, -2.0], dtype=torch.float32))
        self.range_phase_delta_head = PhaseDeltaHead(
            context_dim=hidden_dim,
            base_ch=96,
            dropout=dropout,
            max_delta=self.axis_limits[0],
        )
        self.range_phase_gate_logit = nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        self.range_physical_head = PhysicalSequenceHead(self.axis_bins[0], dropout=dropout)
        self.velocity_physical_head = PhysicalSequenceHead(self.axis_bins[1], dropout=dropout)
        self.range_physical_gain = nn.Parameter(torch.tensor(0.15, dtype=torch.float32))
        self.velocity_physical_gain = nn.Parameter(torch.tensor(0.15, dtype=torch.float32))

        self.register_buffer("r_axis", torch.linspace(-self.axis_limits[0], self.axis_limits[0], self.axis_bins[0]))
        self.register_buffer("v_axis", torch.linspace(-self.axis_limits[1], self.axis_limits[1], self.axis_bins[1]))
        self.register_buffer("theta_axis", torch.linspace(-self.axis_limits[2], self.axis_limits[2], self.axis_bins[2]))
        self.register_buffer("axis_scales", torch.tensor([
            float(patch_size[0] - 1) / (2.0 * float(patch_half_span_bins[0])),
            float(patch_size[1] - 1) / (2.0 * float(patch_half_span_bins[1])),
            float(patch_size[2] - 1) / (2.0 * float(patch_half_span_bins[2])),
        ], dtype=torch.float32))
        self.register_buffer("bin_widths", torch.tensor([
            (self.r_max - self.r_min) / self.Nbin_r,
            (self.v_max - self.v_min) / self.Nbin_v,
            (self.theta_max - self.theta_min) / self.Nbin_theta,
        ], dtype=torch.float32))
        self.register_buffer("mins", torch.tensor([self.r_min, self.v_min, self.theta_min], dtype=torch.float32))

        self._init_modules()

    def _init_modules(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv3d, nn.Linear)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.patch_denoise.out_proj.weight)
        nn.init.zeros_(self.patch_denoise.out_proj.bias)
        nn.init.zeros_(self.direct_offset_head.net[-1].weight)
        nn.init.zeros_(self.direct_offset_head.net[-1].bias)
        if self.projection_attention is not None:
            nn.init.zeros_(self.projection_attention.weight)
            nn.init.zeros_(self.projection_attention.bias)

    def _axis_projection(self, feat, keep_dim):
        reduce_dims = tuple(dim for dim in (2, 3, 4) if dim != keep_dim)
        if not self.use_attention_projection:
            return torch.cat([feat.mean(dim=reduce_dims), feat.amax(dim=reduce_dims)], dim=1)

        logits = self.projection_attention(feat)
        flat_shape = list(logits.shape)
        keep_size = flat_shape[keep_dim]
        if keep_dim == 2:
            logits_flat = logits.permute(0, 2, 1, 3, 4).reshape(logits.shape[0], keep_size, -1)
            feat_flat = feat.permute(0, 2, 1, 3, 4).reshape(feat.shape[0], keep_size, feat.shape[1], -1)
        elif keep_dim == 3:
            logits_flat = logits.permute(0, 3, 1, 2, 4).reshape(logits.shape[0], keep_size, -1)
            feat_flat = feat.permute(0, 3, 1, 2, 4).reshape(feat.shape[0], keep_size, feat.shape[1], -1)
        else:
            logits_flat = logits.permute(0, 4, 1, 2, 3).reshape(logits.shape[0], keep_size, -1)
            feat_flat = feat.permute(0, 4, 1, 2, 3).reshape(feat.shape[0], keep_size, feat.shape[1], -1)

        weights = torch.softmax(logits_flat, dim=-1).unsqueeze(2)
        mean = (feat_flat * weights).sum(dim=-1).transpose(1, 2)
        second = (feat_flat.square() * weights).sum(dim=-1).clamp_min(1e-8).sqrt().transpose(1, 2)
        return torch.cat([mean, second], dim=1)

    def _normalise_bins(self, bins):
        denom = torch.tensor(
            [self.Nbin_r - 1, self.Nbin_v - 1, self.Nbin_theta - 1],
            device=bins.device,
            dtype=bins.dtype,
        ).clamp_min(1)
        return bins / denom.view(1, 1, 3) * 2.0 - 1.0

    def physical_to_bin_coords(self, centers_phys):
        return (
            (centers_phys - self.mins.to(centers_phys).view(1, 1, 3))
            / self.bin_widths.to(centers_phys).view(1, 1, 3)
            - 0.5
        )

    def _softargmax(self, logits, axis):
        probs = torch.softmax(logits / self.softargmax_temp, dim=-1)
        delta = (probs * axis.to(logits).view(1, -1)).sum(dim=-1)
        return delta, probs

    def _sample_axis_window(self, x, axis, sample_scale):
        """Sample an axis profile at coordinates expressed in coarse-bin units."""
        N, C, L = x.shape
        if torch.is_tensor(sample_scale):
            sample_scale = float(sample_scale.detach().cpu().item())
        else:
            sample_scale = float(sample_scale)
        center = (float(L) - 1.0) * 0.5
        pos = center + axis.to(x).view(1, -1) * sample_scale
        grid_x = 2.0 * pos / max(float(L - 1), 1.0) - 1.0
        grid_y = torch.zeros_like(grid_x)
        grid = torch.stack([grid_x.expand(N, -1), grid_y.expand(N, -1)], dim=-1).view(N, 1, -1, 2)
        sampled = F.grid_sample(
            x.unsqueeze(2),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.squeeze(2)

    @staticmethod
    def _complex_from_2ch(x):
        return torch.complex(x[:, 0].float(), x[:, 1].float())

    @staticmethod
    def _index_from_bins(bins, n_src, n_bin):
        idx = ((bins.float() + 0.5) * float(n_src) / float(n_bin) - 0.5).round().long()
        return idx.clamp(0, int(n_src) - 1)

    def _sequence_features(self, seq):
        mag = torch.sqrt(seq.real.square() + seq.imag.square() + 1e-6)
        mag_mean = mag.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        phase_conf = (mag / (mag + mag_mean)).clamp(0.0, 1.0)
        rms = mag.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        real = seq.real / rms
        imag = seq.imag / rms
        mag_norm = (torch.log1p(mag / mag_mean))
        mag_norm = (mag_norm - mag_norm.mean(dim=-1, keepdim=True)) / mag_norm.std(dim=-1, keepdim=True).clamp_min(1e-6)
        phase_sin = (seq.imag / mag.clamp_min(1e-6)) * phase_conf
        phase_cos = (seq.real / mag.clamp_min(1e-6)) * phase_conf
        phase_step = seq[:, 1:] * seq[:, :-1].conj()
        step_mag = torch.sqrt(phase_step.real.square() + phase_step.imag.square() + 1e-6)
        step_conf = torch.minimum(phase_conf[:, 1:], phase_conf[:, :-1])
        step_sin = (phase_step.imag / step_mag.clamp_min(1e-6)) * step_conf
        step_cos = (phase_step.real / step_mag.clamp_min(1e-6)) * step_conf
        step_sin = torch.cat([step_sin[:, :1], step_sin], dim=-1)
        step_cos = torch.cat([step_cos[:, :1], step_cos], dim=-1)
        coord = torch.linspace(-1.0, 1.0, seq.shape[-1], device=seq.device, dtype=real.dtype).view(1, -1).expand_as(real)
        return torch.stack([real, imag, mag_norm, phase_sin, phase_cos, step_sin, step_cos, coord], dim=1)

    def _extract_physical_sequences(self, X_h, crop_bins):
        z = self._complex_from_2ch(X_h)
        B, Nc, Ms, Nr = z.shape
        _, P, _ = crop_bins.shape
        flat_b = torch.arange(B, device=X_h.device).view(B, 1).expand(B, P).reshape(-1)
        bins = crop_bins.reshape(B * P, 3)

        theta_idx = self._index_from_bins(bins[:, 2], Nr, self.Nbin_theta)
        velocity_idx = self._index_from_bins(bins[:, 1], Ms, self.Nbin_v)
        range_idx = self._index_from_bins(bins[:, 0], Nc, self.Nbin_r)

        # Range offset lives in the subcarrier phase slope. Keep the raw
        # subcarrier axis, but project slow-time/angle to the coarse target
        # bin so the range sequence is less contaminated by other cells.
        z_vt = torch.fft.fftshift(torch.fft.fft(z, dim=2), dim=2)
        z_vt = torch.fft.fftshift(torch.fft.fft(z_vt, dim=3), dim=3)
        range_terms = []
        for dv in (-1, 0, 1):
            vv = (velocity_idx + dv).clamp(0, Ms - 1)
            for dt in (-1, 0, 1):
                tt = (theta_idx + dt).clamp(0, Nr - 1)
                range_terms.append(z_vt[flat_b, :, vv, tt])
        range_seq = torch.stack(range_terms, dim=0).mean(dim=0)

        # Velocity offset lives in slow-time phase rotation. Keep the
        # slow-time axis, and project range/angle to the coarse target bin.
        z_rt = torch.fft.fftshift(torch.fft.ifft(z, dim=1), dim=1)
        z_rt = torch.fft.fftshift(torch.fft.fft(z_rt, dim=3), dim=3)
        velocity_terms = []
        for dr in (-1, 0, 1):
            rr = (range_idx + dr).clamp(0, Nc - 1)
            for dt in (-1, 0, 1):
                tt = (theta_idx + dt).clamp(0, Nr - 1)
                velocity_terms.append(z_rt[flat_b, rr, :, tt])
        velocity_seq = torch.stack(velocity_terms, dim=0).mean(dim=0)
        return range_seq, velocity_seq

    def decode_physical(self, crop_center_phys, delta_from_crop):
        pred_phys = crop_center_phys.to(delta_from_crop) + (
            delta_from_crop * self.bin_widths.to(delta_from_crop).view(1, 1, 3)
        )
        cont = (
            pred_phys - self.mins.to(pred_phys).view(1, 1, 3)
        ) / self.bin_widths.to(pred_phys).view(1, 1, 3)
        pred_bins = torch.floor(cont).long()
        pred_bins[..., 0].clamp_(0, self.Nbin_r - 1)
        pred_bins[..., 1].clamp_(0, self.Nbin_v - 1)
        pred_bins[..., 2].clamp_(0, self.Nbin_theta - 1)
        return pred_bins, pred_phys

    def forward(self, X_h, X_l, crop_center_phys, coarse_delta=None, query_feat=None):
        B, P, _ = crop_center_phys.shape
        crop_bin_coords = self.physical_to_bin_coords(crop_center_phys)
        patches = self.patch_extractor(X_h, X_l, crop_center_phys)
        patches_flat = patches.reshape(B * P, 10, *patches.shape[-3:])
        denoised_patches, denoise_embed = self.patch_denoise(patches_flat)
        feat = self.backbone(denoised_patches)

        if query_feat is None:
            query_feat = torch.zeros(B, P, self.query_proj[0].normalized_shape[0], device=X_h.device, dtype=X_h.dtype)
        if coarse_delta is None:
            coarse_delta = torch.zeros(B, P, 3, device=X_h.device, dtype=X_h.dtype)

        meta = torch.cat([self._normalise_bins(crop_bin_coords), coarse_delta.to(X_h.dtype)], dim=-1)
        context = self.query_proj(query_feat).reshape(B * P, -1) + self.meta_proj(meta).reshape(B * P, -1)
        bias = self.context_to_bias(context).view(B * P, -1, 1, 1, 1)
        feat = feat + bias

        r_feat = self._axis_projection(feat, keep_dim=2)
        v_feat = self._axis_projection(feat, keep_dim=3)
        theta_feat = self._axis_projection(feat, keep_dim=4)

        r_feat = self._sample_axis_window(r_feat, self.r_axis, self.axis_scales[0])
        v_feat = self._sample_axis_window(v_feat, self.v_axis, self.axis_scales[1])
        theta_feat = self._sample_axis_window(theta_feat, self.theta_axis, self.axis_scales[2])

        r_logits_net = self.r_head(r_feat)
        v_logits_net = self.v_head(v_feat)
        theta_logits = self.theta_head(theta_feat)

        if self.use_physical_axis_branch:
            crop_bins = crop_bin_coords.round().long()
            range_seq, velocity_seq = self._extract_physical_sequences(X_h, crop_bins)
            r_phys_logits = self.range_physical_head(self._sequence_features(range_seq))
            v_phys_logits = self.velocity_physical_head(self._sequence_features(velocity_seq))
            r_phys_delta, p_r_phys = self._softargmax(r_phys_logits, self.r_axis)
            v_phys_delta, p_v_phys = self._softargmax(v_phys_logits, self.v_axis)
            r_logits = r_logits_net + self.range_physical_gain.clamp(0.0, 3.0) * r_phys_logits
            v_logits = v_logits_net + self.velocity_physical_gain.clamp(0.0, 3.0) * v_phys_logits
        else:
            r_phys_logits = None
            v_phys_logits = None
            r_phys_delta = None
            v_phys_delta = None
            p_r_phys = None
            p_v_phys = None
            r_logits = r_logits_net
            v_logits = v_logits_net

        delta_r, p_r = self._softargmax(r_logits, self.r_axis)
        delta_v, p_v = self._softargmax(v_logits, self.v_axis)
        delta_theta, p_theta = self._softargmax(theta_logits, self.theta_axis)
        heatmap_delta = torch.stack([delta_r, delta_v, delta_theta], dim=-1)

        if self.use_full_patch_direct_branch:
            direct_delta = self.direct_offset_head(feat, context)
            gates = torch.sigmoid(self.direct_delta_gate_logits).to(direct_delta).view(1, 3)
            delta_flat = heatmap_delta + gates * (direct_delta - heatmap_delta)
            limits = torch.tensor(self.axis_limits, device=delta_flat.device, dtype=delta_flat.dtype).view(1, 3)
            delta_flat = torch.maximum(torch.minimum(delta_flat, limits), -limits)
        else:
            direct_delta = torch.zeros_like(heatmap_delta)
            gates = torch.zeros(1, 3, device=heatmap_delta.device, dtype=heatmap_delta.dtype)
            delta_flat = heatmap_delta

        if self.use_range_phase_direct_branch:
            crop_bins = crop_bin_coords.round().long()
            range_seq, _ = self._extract_physical_sequences(X_h, crop_bins)
            range_phase_delta = self.range_phase_delta_head(self._sequence_features(range_seq), context)
            range_phase_gate = torch.sigmoid(self.range_phase_gate_logit).to(delta_flat)
            delta_flat = delta_flat.clone()
            delta_flat[:, 0] = delta_flat[:, 0] + range_phase_gate * (range_phase_delta - delta_flat[:, 0])
            delta_flat[:, 0].clamp_(-self.axis_limits[0], self.axis_limits[0])
        else:
            range_phase_delta = torch.zeros_like(delta_flat[:, 0])
            range_phase_gate = torch.zeros((), device=delta_flat.device, dtype=delta_flat.dtype)

        delta_from_crop = delta_flat.view(B, P, 3)
        heatmap_delta = heatmap_delta.view(B, P, 3)
        direct_delta = direct_delta.view(B, P, 3)
        range_phase_delta = range_phase_delta.view(B, P)

        pred_bins, pred_phys = self.decode_physical(crop_center_phys, delta_from_crop)
        out = {
            "delta_from_crop": delta_from_crop,
            "heatmap_delta_from_crop": heatmap_delta,
            "direct_delta_from_crop": direct_delta,
            "direct_delta_gates": gates.view(3).detach(),
            "range_phase_delta": range_phase_delta,
            "range_phase_gate": range_phase_gate.detach(),
            "denoise_embed": denoise_embed.view(B, P, -1),
            "denoised_patch_stats": torch.stack(
                [
                    denoised_patches.mean(dim=(1, 2, 3, 4)),
                    denoised_patches.std(dim=(1, 2, 3, 4), unbiased=False),
                ],
                dim=-1,
            ).view(B, P, 2).detach(),
            "pred_bins": pred_bins,
            "pred_phys": pred_phys,
            "crop_center_phys": crop_center_phys,
            "crop_bin_coords": crop_bin_coords,
            "r_logits": r_logits.view(B, P, -1),
            "r_logits_net": r_logits_net.view(B, P, -1),
            "v_logits": v_logits.view(B, P, -1),
            "v_logits_net": v_logits_net.view(B, P, -1),
            "theta_logits": theta_logits.view(B, P, -1),
            "p_r": p_r.view(B, P, -1),
            "p_v": p_v.view(B, P, -1),
            "p_theta": p_theta.view(B, P, -1),
            "r_axis": self.r_axis,
            "v_axis": self.v_axis,
            "theta_axis": self.theta_axis,
        }
        if self.use_physical_axis_branch:
            out.update(
                {
                    "r_phys_logits": r_phys_logits.view(B, P, -1),
                    "r_phys_delta": r_phys_delta.view(B, P),
                    "p_r_phys": p_r_phys.view(B, P, -1),
                    "range_physical_gain": self.range_physical_gain.detach(),
                    "v_phys_logits": v_phys_logits.view(B, P, -1),
                    "v_phys_delta": v_phys_delta.view(B, P),
                    "p_v_phys": p_v_phys.view(B, P, -1),
                    "velocity_physical_gain": self.velocity_physical_gain.detach(),
                }
            )
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# 0. Utils
# ======================================================================
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


# ======================================================================
# 1. FFT spectrum preprocessing
# ======================================================================
class FFTSpectrumPreprocessor(nn.Module):
    """
    Input:
        X_h: [B, 2, F, T, A]
        X_l: [B, 2, F, T, A]

    Output:
        spec_h: [B, 1, 32, 32, 32]
        spec_l: [B, 1, 32, 32, 32]
    """
    def __init__(self, out_size=(32, 32, 32), use_high=True, use_low=True, eps=1e-6):
        super().__init__()
        self.out_size = to_3tuple(out_size)
        self.use_high = bool(use_high)
        self.use_low = bool(use_low)
        self.eps = float(eps)

        if not self.use_high and not self.use_low:
            raise ValueError("use_high and use_low cannot both be False.")

    @staticmethod
    def _complex_from_2ch(x):
        return torch.complex(x[:, 0].float(), x[:, 1].float())

    def _spectrum_one_band(self, x, return_stats=False):
        z = self._complex_from_2ch(x)  # [B,F,T,A]

        z = torch.fft.ifft(z, dim=1)
        z = torch.fft.fftshift(z, dim=1)

        z = torch.fft.fft(z, dim=2)
        z = torch.fft.fftshift(z, dim=2)

        z = torch.fft.fft(z, dim=3)
        z = torch.fft.fftshift(z, dim=3)

        spec = z.real ** 2 + z.imag ** 2
        spec = spec.unsqueeze(1)  # [B,1,F,T,A]

        if return_stats:
            flat = spec.flatten(1)
            raw_mean = flat.mean(dim=1).clamp_min(self.eps)
            raw_std = flat.std(dim=1).clamp_min(self.eps)
            raw_max = flat.max(dim=1).values.clamp_min(self.eps)
            raw_peak_mean = (raw_max / raw_mean).clamp_min(self.eps)
            stats = torch.stack(
                [
                    torch.log(raw_mean),
                    torch.log(raw_std),
                    torch.log(raw_max),
                    torch.log(raw_peak_mean),
                ],
                dim=1,
            )

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
        spec = (spec - mean) / std

        if return_stats:
            return spec, stats

        return spec

    def forward(self, X_h, X_l, return_stats=False):
        spec_h = stats_h = None
        spec_l = stats_l = None

        if self.use_high:
            out_h = self._spectrum_one_band(X_h, return_stats=return_stats)
            if return_stats:
                spec_h, stats_h = out_h
            else:
                spec_h = out_h

        if self.use_low:
            out_l = self._spectrum_one_band(X_l, return_stats=return_stats)
            if return_stats:
                spec_l, stats_l = out_l
            else:
                spec_l = out_l

        if return_stats:
            return spec_h, spec_l, stats_h, stats_l

        return spec_h, spec_l


# ======================================================================
# 2. 3D CNN stem
# ======================================================================
class ResDownBlock3D(nn.Module):
    """
    Input:
        x: [B, C_in, D, H, W]

    Output:
        y: [B, C_out, D/stride, H/stride, W/stride]
    """
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.0):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_ch,
            out_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm1 = make_group_norm(out_ch)

        self.conv2 = nn.Conv3d(
            out_ch,
            out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.norm2 = make_group_norm(out_ch)

        self.act = nn.GELU()
        self.drop = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

        if in_ch != out_ch or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                make_group_norm(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        s = self.skip(x)

        y = self.conv1(x)
        y = self.norm1(y)
        y = self.act(y)
        y = self.drop(y)

        y = self.conv2(y)
        y = self.norm2(y)

        y = self.act(y + s)
        return y


class BandStem3D(nn.Module):
    """
    Input:
        spec: [B, 1, 32, 32, 32]

    Output:
        feat: [B, embed_dim, 8, 8, 8]
    """
    def __init__(self, in_ch=1, base_ch=32, embed_dim=128, dropout=0.15):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, base_ch, kernel_size=3, stride=1, padding=1, bias=False),
            make_group_norm(base_ch),
            nn.GELU(),
        )

        self.layer1 = ResDownBlock3D(base_ch, 48, stride=2, dropout=dropout)
        self.layer2 = ResDownBlock3D(48, 96, stride=2, dropout=dropout)
        self.layer3 = ResDownBlock3D(96, 128, stride=1, dropout=dropout)
        self.proj = nn.Conv3d(128, embed_dim, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.proj(x)
        return x


# ======================================================================
# 3. Cross-attention block
# ======================================================================
class BidirectionalCrossAttentionBlock(nn.Module):
    """
    Input:
        T_h: [B, N, C]
        T_l: [B, N, C]

    Output:
        T_h_out: [B, N, C]
        T_l_out: [B, N, C]
    """
    def __init__(self, embed_dim=128, num_heads=4, mlp_ratio=2.0, dropout=0.15, attn_dropout=0.10):
        super().__init__()

        hidden_dim = int(embed_dim * mlp_ratio)

        self.norm_h_q = nn.LayerNorm(embed_dim)
        self.norm_l_kv = nn.LayerNorm(embed_dim)
        self.norm_l_q = nn.LayerNorm(embed_dim)
        self.norm_h_kv = nn.LayerNorm(embed_dim)

        self.h_to_l_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.l_to_h_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.drop = nn.Dropout(dropout)

        self.norm_h_ffn = nn.LayerNorm(embed_dim)
        self.norm_l_ffn = nn.LayerNorm(embed_dim)

        self.ffn_h = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.ffn_l = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, T_h, T_l):
        h_q = self.norm_h_q(T_h)
        l_kv = self.norm_l_kv(T_l)
        h_attn, _ = self.h_to_l_attn(
            query=h_q,
            key=l_kv,
            value=l_kv,
            need_weights=False,
        )
        T_h_cross = T_h + self.drop(h_attn)

        l_q = self.norm_l_q(T_l)
        h_kv = self.norm_h_kv(T_h)
        l_attn, _ = self.l_to_h_attn(
            query=l_q,
            key=h_kv,
            value=h_kv,
            need_weights=False,
        )
        T_l_cross = T_l + self.drop(l_attn)

        T_h_out = T_h_cross + self.drop(self.ffn_h(self.norm_h_ffn(T_h_cross)))
        T_l_out = T_l_cross + self.drop(self.ffn_l(self.norm_l_ffn(T_l_cross)))

        return T_h_out, T_l_out


# ======================================================================
# 4. Token fusion and pooling
# ======================================================================
class TokenFusionProjection(nn.Module):
    """
    Input:
        T_h: [B, N, C]
        T_l: [B, N, C]

    Output:
        T_f: [B, N, C]
    """
    def __init__(self, embed_dim=128, dropout=0.15):
        super().__init__()

        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim * 4),
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, T_h, T_l):
        T_cat = torch.cat(
            [
                T_h,
                T_l,
                torch.abs(T_h - T_l),
                T_h * T_l,
            ],
            dim=-1,
        )
        return self.proj(T_cat)


class FusionSelfAttentionBlock(nn.Module):
    """
    Input:
        T: [B, N, C]

    Output:
        T_out: [B, N, C]
    """
    def __init__(self, embed_dim=128, num_heads=4, mlp_ratio=2.0, dropout=0.15, attn_dropout=0.10):
        super().__init__()

        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm_attn = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, T):
        T_norm = self.norm_attn(T)
        T_attn, _ = self.attn(
            query=T_norm,
            key=T_norm,
            value=T_norm,
            need_weights=False,
        )
        T = T + self.drop(T_attn)
        T = T + self.drop(self.ffn(self.norm_ffn(T)))
        return T


class TokenPooling(nn.Module):
    """
    Input:
        T_f: [B, N, C]

    Output:
        z: [B, 3C]
    """
    def __init__(self, embed_dim=128):
        super().__init__()
        self.attn_score = nn.Linear(embed_dim, 1)

    def forward(self, T_f):
        z_avg = T_f.mean(dim=1)
        z_max = T_f.max(dim=1).values

        score = self.attn_score(T_f)
        weight = torch.softmax(score, dim=1)
        z_attn = torch.sum(weight * T_f, dim=1)

        z = torch.cat([z_avg, z_max, z_attn], dim=-1)
        return z


def make_count_head(in_dim, hidden_dim, num_classes, dropout):
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 128),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(128, num_classes),
    )


# ======================================================================
# 5. DualBandCrossFusionCountNet
# ======================================================================
class DualBandCrossFusionCountNet(nn.Module):
    """
    Dual-band cross-fusion CountNet.

    Input:
        X_h: [B, 2, 64, 112, 32]
        X_l: [B, 2, 64, 14, 4]

    Output dict:
        K_logits: [B, Kmax+1]
        K_ord_logits: [B, Kmax]
        K_count_logit: [B]
        feat: [B, 3*embed_dim]
        tokens: [B, 512, embed_dim]
    """
    def __init__(
        self,
        Kmax=5,
        spec_size=(32, 32, 32),
        token_grid=(8, 8, 8),
        embed_dim=128,
        base_ch=32,
        num_heads=4,
        cross_attn_layers=1,
        fusion_attn_layers=0,
        mlp_ratio=2.0,
        dropout=0.15,
        attn_dropout=0.10,
        token_dropout=0.10,
        hidden_dim=256,
        use_high=True,
        use_low=True,
    ):
        super().__init__()

        self.Kmax = int(Kmax)
        self.num_classes = self.Kmax + 1
        self.spec_size = to_3tuple(spec_size)
        self.token_grid = to_3tuple(token_grid)
        self.embed_dim = int(embed_dim)
        self.num_tokens = self.token_grid[0] * self.token_grid[1] * self.token_grid[2]
        self.use_high = bool(use_high)
        self.use_low = bool(use_low)

        if not (self.use_high and self.use_low):
            raise NotImplementedError("DualBandCrossFusionCountNet currently requires both high and low bands.")

        if self.token_grid != (8, 8, 8):
            raise ValueError("This implementation expects token_grid=(8,8,8) for 512 tokens.")

        if self.num_tokens != 512:
            raise ValueError(f"Expected 512 tokens, got {self.num_tokens}")

        self.pre = FFTSpectrumPreprocessor(
            out_size=self.spec_size,
            use_high=self.use_high,
            use_low=self.use_low,
        )

        self.stem_h = BandStem3D(
            in_ch=1,
            base_ch=base_ch,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        self.stem_l = BandStem3D(
            in_ch=1,
            base_ch=base_ch,
            embed_dim=embed_dim,
            dropout=dropout,
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, embed_dim))
        self.band_embed_h = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.band_embed_l = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.token_drop = nn.Dropout(token_dropout)

        if cross_attn_layers not in (1, 2):
            raise ValueError("cross_attn_layers should be 1 or 2.")

        self.cross_blocks = nn.ModuleList(
            [
                BidirectionalCrossAttentionBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(cross_attn_layers)
            ]
        )

        self.fusion = TokenFusionProjection(embed_dim=embed_dim, dropout=dropout)
        self.fusion_blocks = nn.ModuleList(
            [
                FusionSelfAttentionBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                )
                for _ in range(int(fusion_attn_layers))
            ]
        )
        self.pool = TokenPooling(embed_dim=embed_dim)
        self.pool_h = TokenPooling(embed_dim=embed_dim)
        self.pool_l = TokenPooling(embed_dim=embed_dim)

        pooled_dim = embed_dim * 3
        self.stats_proj = nn.Sequential(
            nn.Linear(8, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, pooled_dim),
        )
        self.stats_proj_h = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, pooled_dim),
        )
        self.stats_proj_l = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, pooled_dim),
        )

        self.count_head = make_count_head(
            pooled_dim,
            hidden_dim,
            self.num_classes,
            dropout,
        )
        self.band_head_h = make_count_head(
            pooled_dim,
            hidden_dim,
            self.num_classes,
            dropout,
        )
        self.band_head_l = make_count_head(
            pooled_dim,
            hidden_dim,
            self.num_classes,
            dropout,
        )

        self.ordinal_head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.Kmax),
        )

        self.count_reg_head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.high_count_head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

        self.k4_head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.band_embed_h, std=0.02)
        nn.init.trunc_normal_(self.band_embed_l, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _tokenize(x):
        return x.flatten(2).transpose(1, 2)

    def forward(self, X_h, X_l):
        spec_h, spec_l, stats_h, stats_l = self.pre(X_h, X_l, return_stats=True)

        F_h = self.stem_h(spec_h)  # [B,128,8,8,8]
        F_l = self.stem_l(spec_l)  # [B,128,8,8,8]

        T_h = self._tokenize(F_h)  # [B,512,128]
        T_l = self._tokenize(F_l)  # [B,512,128]

        if T_h.shape[1] != self.num_tokens or T_l.shape[1] != self.num_tokens:
            raise RuntimeError(
                f"Unexpected token count: high={T_h.shape[1]}, low={T_l.shape[1]}, expected={self.num_tokens}"
            )

        T_h = self.token_drop(T_h + self.pos_embed + self.band_embed_h)
        T_l = self.token_drop(T_l + self.pos_embed + self.band_embed_l)

        # Single-band auxiliary heads must branch before cross-attention;
        # otherwise they already contain information from the other band.
        T_h_single = T_h
        T_l_single = T_l

        for block in self.cross_blocks:
            T_h, T_l = block(T_h, T_l)

        T_f = self.token_drop(self.fusion(T_h, T_l))  # [B,512,128]

        for block in self.fusion_blocks:
            T_f = block(T_f)

        z_h = self.pool_h(T_h_single) + self.stats_proj_h(stats_h)
        z_l = self.pool_l(T_l_single) + self.stats_proj_l(stats_l)
        z = self.pool(T_f)           # [B,384]
        stats = torch.cat([stats_h, stats_l], dim=1)
        z = z + self.stats_proj(stats)

        K_logits = self.count_head(z)
        K_band_h_logits = self.band_head_h(z_h)
        K_band_l_logits = self.band_head_l(z_l)
        K_ord_logits = self.ordinal_head(z)
        K_count_logit = self.count_reg_head(z).squeeze(-1)
        K_high_logits = self.high_count_head(z)
        K4_logit = self.k4_head(z).squeeze(-1)

        return {
            "K_logits": K_logits,
            "K_band_h_logits": K_band_h_logits,
            "K_band_l_logits": K_band_l_logits,
            "K_ord_logits": K_ord_logits,
            "K_count_logit": K_count_logit,
            "K_high_logits": K_high_logits,
            "K4_logit": K4_logit,
            "feat": z,
            "tokens": T_f,
        }


# ======================================================================
# 6. Test main
# ======================================================================
if __name__ == "__main__":
    model = DualBandCrossFusionCountNet()
    count_parameters(model)

    Xh = torch.randn(2, 2, 64, 112, 32)
    Xl = torch.randn(2, 2, 64, 14, 4)

    with torch.no_grad():
        out = model(Xh, Xl)

    for k, v in out.items():
        if torch.is_tensor(v):
            print(f"{k}: {tuple(v.shape)}")

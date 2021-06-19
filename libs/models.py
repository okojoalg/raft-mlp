from abc import ABC
from typing import List, Dict

from torch import nn
from einops.layers.torch import Rearrange, Reduce

from libs.const import PATCH_SIZE, DIM, DEPTH, SEP_HW_LN_CODIM, TOKEN_MIXING_TYPES, ORIGINAL


class Block(nn.Module):
    def __init__(self, dim, norm_dim, expansion_factor=4, dropout=0., channel_norm=False):
        super().__init__()
        if dim == norm_dim:
            self.norm = nn.LayerNorm(norm_dim)
        elif channel_norm:
            self.norm = nn.Sequential(*[
                Rearrange('b (c o1) o2 -> b (o1 o2) c', o2=dim),
                nn.LayerNorm(norm_dim),
                Rearrange('b (o1 o2) c -> b (c o1) o2', o2=dim),
            ])
        else:
            self.norm = nn.Sequential(*[
                Rearrange('b c o -> b o c'),
                nn.LayerNorm(norm_dim),
                Rearrange('b o c -> b c o'),
            ])

        self.fn = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class Level(nn.Module, ABC):
    def __init__(self, image_size=224, patch_size=4):
        super().__init__()
        self._h = self._w = image_size // patch_size

    def forward(self, input):
        return self.fn(input)


class SeparateLNCodimLevel(Level):
    def __init__(self, in_channels, out_channels, depth=4, image_size=224, patch_size=4, expansion_factor=4,
                 dropout=0.):
        super().__init__(image_size, patch_size)
        self.fn = nn.Sequential(*[
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear((patch_size ** 2) * in_channels, out_channels),
            *[nn.Sequential(*[
                # height mixer
                Rearrange('b (h w) c -> b (c w) h', h=self._h),
                Block(self._h, out_channels * self._w, expansion_factor, dropout),
                # width mixer
                Rearrange('b (c w) h -> b (c h) w', h=self._h, w=self._w),
                Block(self._w, out_channels * self._h, expansion_factor, dropout),
                # channel mixer
                Rearrange('b (c h) w -> b (h w) c', h=self._h, w=self._w),
                Block(out_channels, out_channels, expansion_factor, dropout),
            ])
              for _ in range(depth)],
            Rearrange('b (h w) c -> b c h w', h=self._h, w=self._w)])


class SeparateLNChannelLevel(Level):
    def __init__(self, in_channels, out_channels, depth=4, image_size=224, patch_size=4, expansion_factor=4,
                 dropout=0.):
        super().__init__(image_size, patch_size)
        self.fn = nn.Sequential(*[
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear((patch_size ** 2) * in_channels, out_channels),
            *[nn.Sequential(*[
                # height mixer
                Rearrange('b (h w) c -> b (c w) h', h=self._h),
                Block(self._h, out_channels * self._w, expansion_factor, dropout, channel_norm=True),
                # width mixer
                Rearrange('b (c w) h -> b (c h) w', h=self._h, w=self._w),
                Block(self._w, out_channels * self._h, expansion_factor, dropout, channel_norm=True),
                # channel mixer
                Rearrange('b (c h) w -> b (h w) c', h=self._h, w=self._w),
                Block(out_channels, out_channels, expansion_factor, dropout),
            ])
              for _ in range(depth)],
            Rearrange('b (h w) c -> b c h w', h=self._h, w=self._w)])


class OriginalLevel(Level):
    def __init__(self, in_channels, out_channels, depth=4, image_size=224, patch_size=4, expansion_factor=4,
                 dropout=0.):
        super().__init__(image_size, patch_size)
        self.fn = nn.Sequential(*[
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear((patch_size ** 2) * in_channels, out_channels),
            *[nn.Sequential(*[
                # token mixer
                Rearrange('b (h w) c -> b c (h w)', h=self._h, w=self._w),
                Block(self._h * self._w, out_channels, expansion_factor, dropout),
                # channel mixer
                Rearrange('b c (h w) -> b (h w) c', h=self._h, w=self._w),
                Block(out_channels, out_channels, expansion_factor, dropout),
            ])
              for _ in range(depth)],
            Rearrange('b (h w) c -> b c h w', h=self._h, w=self._w)])


class PyramidMixer(nn.Module):
    def __init__(
            self,
            layers: List[Dict],
            in_channels: int = 3,
            image_size: int = 224,
            num_classes: int = 1000,
            expansion_factor: int = 4,
            dropout: float = 0.,
            token_mixing_type: str = SEP_HW_LN_CODIM,
    ):
        assert token_mixing_type in TOKEN_MIXING_TYPES
        for layer in layers:
            assert DEPTH in layer
            assert DIM in layer
            assert PATCH_SIZE in layer
        super().__init__()
        if token_mixing_type == ORIGINAL:
            level = OriginalLevel
        elif token_mixing_type == SEP_HW_LN_CODIM:
            level = SeparateLNCodimLevel
        else:
            level = SeparateLNChannelLevel
        modules = []
        for i, layer in enumerate(layers):
            modules.append(level(
                in_channels if i == 0 else layers[i - 1].get(DIM),
                layer.get(DIM),
                depth=layer.get(DEPTH),
                image_size=image_size,
                patch_size=layer.get(PATCH_SIZE),
                expansion_factor=expansion_factor,
                dropout=dropout,
            ))
            image_size = image_size // layer.get(PATCH_SIZE)
        self.levels = nn.Sequential(*modules)
        self.head = nn.Sequential(*[
            Rearrange('b c h w -> b h w c', c=layers[-1].get(DIM)),
            nn.LayerNorm(layers[-1].get(DIM)),
            Reduce('b h w c -> b c', 'mean'),
            nn.Linear(layers[-1].get(DIM), num_classes)
        ])

    def forward(self, input):
        output = self.levels(input)
        return self.head(output)

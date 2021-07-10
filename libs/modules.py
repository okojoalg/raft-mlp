import math
from abc import ABC

from einops.layers.torch import Rearrange
from torch import nn
from torch.nn import functional as F

from libs.regularizations import DropPath


class Block(nn.Module):
    def __init__(
        self, dim, expansion_factor=4, dropout=0.0, drop_path_rate=0.0
    ):
        super().__init__()
        self.norm = nn.Identity()
        self.drop = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        self.fn = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.drop(self.fn(self.norm(x))) + x


class ChannelBlock(Block):
    def __init__(
        self, dim, expansion_factor=4, dropout=0.0, drop_path_rate=0.0
    ):
        super().__init__(dim, expansion_factor, dropout, drop_path_rate)
        self.norm = nn.LayerNorm(dim)


class TokenBlock(Block):
    def __init__(
        self,
        dim,
        channels,
        expansion_factor=4,
        dropout=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__(dim, expansion_factor, dropout, drop_path_rate)
        self.norm = nn.Sequential(
            *[
                Rearrange("b c o -> b o c"),
                nn.LayerNorm(channels),
                Rearrange("b o c -> b c o"),
            ]
        )


class SpatiallySeparatedTokenBlock(Block):
    def __init__(
        self,
        dim,
        channels,
        expansion_factor=4,
        dropout=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__(dim, expansion_factor, dropout, drop_path_rate)
        self.norm = nn.Sequential(
            *[
                Rearrange("b (c o1) o2 -> b (o1 o2) c", c=channels, o2=dim),
                nn.LayerNorm(channels),
                Rearrange("b (o1 o2) c -> b (c o1) o2", c=channels, o2=dim),
            ]
        )


class PermutedBlock(Block):
    def __init__(
        self,
        spatial_dim,
        channels,
        raft_size,
        expansion_factor=4,
        dropout=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__(
            spatial_dim * raft_size,
            expansion_factor,
            dropout,
            drop_path_rate,
        )
        self.norm = nn.Sequential(
            *[
                Rearrange(
                    "b (c1 o1) (c2 o2) -> b (o1 o2) (c1 c2)",
                    c1=channels // raft_size,
                    c2=raft_size,
                    o2=spatial_dim,
                ),
                nn.LayerNorm(channels),
                Rearrange(
                    "b (o1 o2) (c1 c2) -> b (c1 o1) (c2 o2)",
                    c1=channels // raft_size,
                    c2=raft_size,
                    o2=spatial_dim,
                ),
            ]
        )


class Level(nn.Module, ABC):
    def __init__(self, image_size=224, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.fn = nn.Identity()
        self._bh = self._bw = image_size // patch_size
        self._h = self._w = math.ceil(image_size / patch_size)

    def forward(self, input):
        if not (self._bh == self._h and self._bw == self._w):
            input = F.interpolate(
                input,
                (self._h * self.patch_size, self._w * self.patch_size),
                mode="bilinear",
                align_corners=False,
            )
        return self.fn(input)


class SeparatedLNCodimLevel(Level):
    def __init__(
        self,
        in_channels,
        out_channels,
        depth=4,
        image_size=224,
        patch_size=4,
        token_expansion_factor=2,
        channel_expansion_factor=4,
        dropout=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__(image_size, patch_size)
        self.fn = nn.Sequential(
            *[
                Rearrange(
                    "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                    p1=patch_size,
                    p2=patch_size,
                ),
                nn.Linear((patch_size ** 2) * in_channels, out_channels)
                if patch_size != 1
                or (patch_size == 1 and in_channels == out_channels)
                else nn.Identity(),
                *[
                    nn.Sequential(
                        *[
                            # vertical mixer
                            Rearrange("b (h w) c -> b (c w) h", h=self._h),
                            TokenBlock(
                                self._h,
                                out_channels * self._w,
                                token_expansion_factor,
                                dropout,
                                drop_path_rate,
                            ),
                            # horizontal mixer
                            Rearrange(
                                "b (c w) h -> b (c h) w", h=self._h, w=self._w
                            ),
                            TokenBlock(
                                self._w,
                                out_channels * self._h,
                                token_expansion_factor,
                                dropout,
                                drop_path_rate,
                            ),
                            # channel mixer
                            Rearrange(
                                "b (c h) w -> b (h w) c", h=self._h, w=self._w
                            ),
                            ChannelBlock(
                                out_channels,
                                channel_expansion_factor,
                                dropout,
                                drop_path_rate,
                            ),
                        ]
                    )
                    for _ in range(depth)
                ],
                Rearrange("b (h w) c -> b c h w", h=self._h, w=self._w),
            ]
        )


class SeparatedLNChannelLevel(Level):
    def __init__(
        self,
        in_channels,
        out_channels,
        depth=4,
        image_size=224,
        patch_size=4,
        token_expansion_factor=2,
        channel_expansion_factor=4,
        dropout=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__(image_size, patch_size)
        self.fn = nn.Sequential(
            *[
                Rearrange(
                    "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                    p1=patch_size,
                    p2=patch_size,
                ),
                nn.Linear((patch_size ** 2) * in_channels, out_channels)
                if patch_size != 1
                or (patch_size == 1 and in_channels == out_channels)
                else nn.Identity(),
                *[
                    nn.Sequential(
                        *[
                            # vertical mixer
                            Rearrange("b (h w) c -> b (c w) h", h=self._h),
                            SpatiallySeparatedTokenBlock(
                                self._h,
                                out_channels,
                                token_expansion_factor,
                                dropout,
                                drop_path_rate,
                            ),
                            # horizontal mixer
                            Rearrange(
                                "b (c w) h -> b (c h) w", h=self._h, w=self._w
                            ),
                            SpatiallySeparatedTokenBlock(
                                self._w,
                                out_channels,
                                token_expansion_factor,
                                dropout,
                                drop_path_rate,
                            ),
                            # channel mixer
                            Rearrange(
                                "b (c h) w -> b (h w) c", h=self._h, w=self._w
                            ),
                            ChannelBlock(
                                out_channels,
                                channel_expansion_factor,
                                dropout,
                                drop_path_rate,
                            ),
                        ]
                    )
                    for _ in range(depth)
                ],
                Rearrange("b (h w) c -> b c h w", h=self._h, w=self._w),
            ]
        )


class SerialPermutedLevel(Level):
    def __init__(
        self,
        in_channels,
        out_channels,
        depth=4,
        image_size=224,
        patch_size=4,
        token_expansion_factor=2,
        channel_expansion_factor=4,
        dropout=0.0,
        drop_path_rate=0.0,
        raft_size=4,
    ):
        super().__init__(image_size, patch_size)

        assert out_channels % raft_size == 0
        self.fn = nn.Sequential(
            *[
                Rearrange(
                    "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                    p1=patch_size,
                    p2=patch_size,
                ),
                nn.Linear((patch_size ** 2) * in_channels, out_channels)
                if patch_size != 1
                or (patch_size == 1 and in_channels == out_channels)
                else nn.Identity(),
                *[
                    nn.Sequential(
                        *[
                            # vertical-channel mixer
                            Rearrange(
                                "b (h w) (chw co) -> b (co w) (chw h)",
                                h=self._h,
                                w=self._w,
                                chw=raft_size,
                            ),
                            PermutedBlock(
                                self._h,
                                out_channels,
                                raft_size,
                                token_expansion_factor,
                                dropout,
                                drop_path_rate,
                            ),
                            # horizontal-channel mixer
                            Rearrange(
                                "b (co w) (chw h) -> b (co h) (chw w)",
                                h=self._h,
                                w=self._w,
                                chw=raft_size,
                            ),
                            PermutedBlock(
                                self._w,
                                out_channels,
                                raft_size,
                                token_expansion_factor,
                                dropout,
                                drop_path_rate,
                            ),
                            # channel mixer
                            Rearrange(
                                "b (co h) (chw w) -> b (h w) (chw co)",
                                h=self._h,
                                w=self._w,
                                chw=raft_size,
                            ),
                            ChannelBlock(
                                out_channels,
                                channel_expansion_factor,
                                dropout,
                                drop_path_rate,
                            ),
                        ]
                    )
                    for _ in range(depth)
                ],
                Rearrange("b (h w) c -> b c h w", h=self._h, w=self._w),
            ]
        )


class OriginalLevel(Level):
    def __init__(
        self,
        in_channels,
        out_channels,
        depth=4,
        image_size=224,
        patch_size=4,
        token_expansion_factor=2,
        channel_expansion_factor=4,
        dropout=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__(image_size, patch_size)
        self.fn = nn.Sequential(
            *[
                Rearrange(
                    "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                    p1=patch_size,
                    p2=patch_size,
                ),
                nn.Linear((patch_size ** 2) * in_channels, out_channels),
                *[
                    nn.Sequential(
                        *[
                            # token mixer
                            Rearrange(
                                "b (h w) c -> b c (h w)", h=self._h, w=self._w
                            ),
                            TokenBlock(
                                self._h * self._w,
                                out_channels,
                                token_expansion_factor,
                                dropout,
                                drop_path_rate,
                            ),
                            # channel mixer
                            Rearrange(
                                "b c (h w) -> b (h w) c", h=self._h, w=self._w
                            ),
                            ChannelBlock(
                                out_channels,
                                channel_expansion_factor,
                                dropout,
                                drop_path_rate,
                            ),
                        ]
                    )
                    for _ in range(depth)
                ],
                Rearrange("b (h w) c -> b c h w", h=self._h, w=self._w),
            ]
        )

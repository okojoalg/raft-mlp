import math
from abc import ABC
from typing import List

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn import functional as F

from libs.consts import EMB_MIXER, EMB_CROSS_MLP, EMBEDDING_TYPES
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
    def __init__(self, pretrained_image_size=224, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.fn = nn.Identity()
        self._ph = self._pw = self.h = self.w = math.ceil(pretrained_image_size / patch_size)

    def forward(self, input):
        _, _, h, w = input.shape
        self.h = h
        self.w = w
        self.update_hw(self.fn)
        if not (h % self.patch_size == 0 and w % self.patch_size == 0):
            input = F.interpolate(
                input,
                (math.ceil(h / self.patch_size) * self.patch_size, math.ceil(w / self.patch_size) * self.patch_size),
                mode="bicubic",
                align_corners=False,
            )
        return self.fn(input)

    def update_hw(self, seq):
        for mod in seq:
            if isinstance(mod, (nn.Sequential)):
                self.update_hw(mod)
            elif isinstance(mod, (RaftShrink, RaftExpansion, Shrink, Expansion, ToPixel)):
                mod.h = math.ceil(self.h / self.patch_size)
                mod.w = math.ceil(self.w / self.patch_size)


class SerialPermutedLevel(Level):
    def __init__(
            self,
            in_channels,
            out_channels,
            depth=4,
            pretrained_image_size=224,
            patch_size=4,
            token_expansion_factor=2,
            channel_expansion_factor=4,
            embedding_type: str = EMB_MIXER,
            embedding_kernels: List[int] = [4, 8, 16, 32],
            dropout=0.0,
            drop_path_rate=0.0,
            raft_size=4,
    ):
        super().__init__(pretrained_image_size, patch_size)
        assert out_channels % raft_size == 0
        assert embedding_type in EMBEDDING_TYPES
        if embedding_type == EMB_MIXER:
            embedding = Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_size,
                p2=patch_size,
            )
        elif embedding_type == EMB_CROSS_MLP:
            assert embedding_kernels[0] == patch_size
            embedding = CrossScaleMLPEmbedding(
                in_channels,
                out_channels,
                embedding_kernels,
            )
        self.fn = nn.Sequential(
            *[
                embedding,
                nn.Linear((patch_size ** 2) * in_channels, out_channels)
                if embedding_type == EMB_MIXER and (patch_size != 1 or (
                        patch_size == 1 and in_channels == out_channels))
                else nn.Identity(),
                *[
                    nn.Sequential(
                        *[
                            RaftShrink(
                                ph=self._ph,
                                pw=self._pw,
                                raft_size=raft_size,
                            ),
                            PermutedBlock(
                                self._ph,
                                out_channels,
                                raft_size,
                                token_expansion_factor,
                                dropout,
                                drop_path_rate,
                            ),
                            # horizontal-channel mixer
                            Rearrange(
                                "b (co w) (chw h) -> b (co h) (chw w)",
                                h=self._ph,
                                w=self._pw,
                                chw=raft_size,
                            ),
                            PermutedBlock(
                                self._pw,
                                out_channels,
                                raft_size,
                                token_expansion_factor,
                                dropout,
                                drop_path_rate,
                            ),
                            # channel mixer
                            RaftExpansion(
                                ph=self._ph,
                                pw=self._pw,
                                raft_size=raft_size,
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
                ToPixel(ph=self._ph, pw=self._pw),
            ]
        )


class OriginalLevel(Level):
    def __init__(
            self,
            in_channels,
            out_channels,
            depth=4,
            pretrained_image_size=224,
            patch_size=4,
            token_expansion_factor=2,
            channel_expansion_factor=4,
            embedding_type: str = EMB_MIXER,
            embedding_kernels: List[int] = [4, 8, 16, 32],
            dropout=0.0,
            drop_path_rate=0.0,
    ):
        super().__init__(pretrained_image_size, patch_size)
        assert embedding_type in EMBEDDING_TYPES
        if embedding_type == EMB_MIXER:
            embedding = Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_size,
                p2=patch_size,
            )
        elif embedding_type == EMB_CROSS_MLP:
            assert embedding_kernels[0] == patch_size
            embedding = CrossScaleMLPEmbedding(
                in_channels,
                out_channels,
                embedding_kernels,
            )
        self.fn = nn.Sequential(
            *[
                embedding,
                nn.Linear((patch_size ** 2) * in_channels, out_channels)
                if embedding_type == EMB_MIXER and (patch_size != 1 or (
                        patch_size == 1 and in_channels == out_channels))
                else nn.Identity(),
                *[
                    nn.Sequential(
                        *[
                            # token mixer
                            Shrink(
                                ph=self._ph,
                                pw=self._pw,
                            ),
                            TokenBlock(
                                self._ph * self._pw,
                                out_channels,
                                token_expansion_factor,
                                dropout,
                                drop_path_rate,
                            ),
                            # channel mixer
                            Expansion(
                                ph=self._ph,
                                pw=self._pw,
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
                ToPixel(ph=self._ph, pw=self._pw),
            ]
        )


class CrossScaleMLPEmbedding(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernels: List[int] = [4, 8, 16, 32],
    ):
        super().__init__()
        self.stride = min(kernels)
        for k in kernels:
            assert k % self.stride == 0

        mlp_in_channels = 0
        for k in kernels:
            mlp_in_channels += k ** 2
        mlp_in_channels *= in_channels
        self.embeddings = nn.ModuleList([
            nn.Sequential(
                *[
                    nn.Unfold(
                        kernel_size=k,
                        stride=self.stride,
                        padding=(k - self.stride) // 2),
                    Rearrange("b c hw -> b hw c")
                ]) for k in kernels
        ])
        self.fc = nn.Linear(mlp_in_channels, out_channels)

    def forward(self, input):
        b, _, h, w = input.shape
        outputs = []
        for emb in self.embeddings:
            output = emb(input)
            outputs.append(output)
        return self.fc(torch.cat(outputs, dim=2))


class RaftShrink(nn.Module):
    def __init__(self, ph, pw, raft_size):
        super().__init__()
        self.h = ph
        self.w = pw
        self.ph = ph
        self.pw = pw
        self.raft_size = raft_size

    def forward(self, input):
        if self.h == self.ph and self.w == self.pw:
            output = rearrange(input, "b (h w) (chw co) -> b (co w) (chw h)", h=self.h, w=self.w, chw=self.raft_size)
        else:
            output = rearrange(input, "b (h w) (chw co) -> b (chw co) h w", h=self.h, w=self.w, chw=self.raft_size)
            output = F.interpolate(output, (self.ph, self.pw), mode="bicubic", align_corners=False)
            output = rearrange(output, "b (chw co) h w -> b (co w) (chw h)", h=self.ph, w=self.pw, chw=self.raft_size)
        return output


class RaftExpansion(nn.Module):
    def __init__(self, ph, pw, raft_size):
        super().__init__()
        self.h = ph
        self.w = pw
        self.ph = ph
        self.pw = pw
        self.raft_size = raft_size

    def forward(self, input):
        if self.h == self.ph and self.w == self.pw:
            output = rearrange(input, "b (co h) (chw w) -> b (h w) (chw co)", h=self.ph, w=self.pw, chw=self.raft_size)
        else:
            output = rearrange(input, "b (co h) (chw w) -> b (chw co) h w", h=self.ph, w=self.pw, chw=self.raft_size)
            output = F.interpolate(output, (self.h, self.w), mode="bicubic", align_corners=False)
            output = rearrange(output, "b (chw co) h w -> b (h w) (chw co)", h=self.h, w=self.w, chw=self.raft_size)
        return output


class Shrink(nn.Module):
    def __init__(self, ph, pw):
        super().__init__()
        self.h = ph
        self.w = pw
        self.ph = ph
        self.pw = pw

    def forward(self, input):
        if self.h == self.ph and self.w == self.pw:
            output = rearrange(input, "b (h w) c -> b c (h w)", h=self.h, w=self.w)
        else:
            output = rearrange(input, "b (h w) c -> b c h w", h=self.h, w=self.w)
            output = F.interpolate(output, (self.ph, self.pw), mode="bicubic", align_corners=False)
            output = rearrange(output, "b c h w -> b c (h w)", h=self.ph, w=self.pw)
        return output


class Expansion(nn.Module):
    def __init__(self, ph, pw):
        super().__init__()
        self.h = ph
        self.w = pw
        self.ph = ph
        self.pw = pw

    def forward(self, input):
        if self.h == self.ph and self.w == self.pw:
            output = rearrange(input, "b c (h w) -> b (h w) c", h=self.ph, w=self.pw)
        else:
            output = rearrange(input, "b c (h w) -> b c h w", h=self.ph, w=self.pw)
            output = F.interpolate(output, (self.h, self.w), mode="bicubic", align_corners=False)
            output = rearrange(output, "b c h w -> b (h w) c", h=self.h, w=self.w)
        return output


class ToPixel(nn.Module):
    def __init__(self, ph, pw):
        super().__init__()
        self.h = ph
        self.w = pw

    def forward(self, input):
        return rearrange(input, "b (h w) c -> b c h w", h=self.h, w=self.w)

import math
from functools import reduce
from typing import List, Dict

from einops.layers.torch import Rearrange, Reduce
from torch import nn

from libs.consts import (
    PATCH_SIZE,
    DIM,
    DEPTH,
    TOKEN_MIXING_TYPES,
    ORIGINAL_TM,
    SER_PM,
    RAFT_SIZE,
    EMB_MIXER,
    EMBEDDING_TYPES,
    EMBEDDING_KERNELS,
)
from libs.modules import (
    SerialPermutedLevel,
    OriginalLevel,
)


class RaftMLP(nn.Module):
    def __init__(
            self,
            layers: List[Dict],
            in_channels: int = 3,
            pretrained_image_size: int = 224,
            num_classes: int = 1000,
            token_expansion_factor: int = 2,
            channel_expansion_factor: int = 4,
            dropout: float = 0.0,
            token_mixing_type: str = SER_PM,
            embedding_type: str = EMB_MIXER,
            shortcut: bool = True,
            drop_path_rate: float = 0.0,
    ):
        assert token_mixing_type in TOKEN_MIXING_TYPES
        assert embedding_type in EMBEDDING_TYPES
        for i, layer in enumerate(layers):
            assert DEPTH in layer
            assert DIM in layer
            assert PATCH_SIZE in layer
            assert token_mixing_type != SER_PM or RAFT_SIZE in layer
            assert 0 < layer.get(DIM)
        super().__init__()
        self.layers = layers
        self.shortcut = shortcut
        if token_mixing_type == ORIGINAL_TM:
            level = OriginalLevel
        else:
            level = SerialPermutedLevel
        levels = []
        heads = []
        for i, layer in enumerate(self.layers):
            params = {
                "in_channels": in_channels
                if i == 0
                else self.layers[i - 1].get(DIM),
                "out_channels": layer.get(DIM),
                "depth": layer.get(DEPTH),
                "pretrained_image_size": pretrained_image_size,
                "patch_size": layer.get(PATCH_SIZE),
                "token_expansion_factor": token_expansion_factor,
                "channel_expansion_factor": channel_expansion_factor,
                "embedding_type": embedding_type,
                "embedding_kernels": layer.get(EMBEDDING_KERNELS),
                "dropout": dropout,
                "drop_path_rate": drop_path_rate,
            }
            if token_mixing_type == SER_PM:
                params["raft_size"] = layer.get(RAFT_SIZE)
            levels.append(level(**params))
            heads_seq = []
            if self.shortcut or len(self.layers) == i + 1:
                heads_seq.append(Rearrange("b c h w -> b h w c"))
                heads_seq.append(nn.LayerNorm(layer.get(DIM)))
                heads_seq.append(Rearrange("b h w c -> b c h w"))
                heads_seq.append(Reduce("b c h w -> b c", "mean"))
                if len(self.layers) != i + 1:
                    heads_seq.append(
                        nn.Linear(layer.get(DIM), self.layers[-1].get(DIM) * 2)
                    )
                heads.append(nn.Sequential(*heads_seq))
            pretrained_image_size = math.ceil(pretrained_image_size / layer.get(PATCH_SIZE))
        self.levels = nn.ModuleList(levels)
        self.heads = nn.ModuleList(heads)
        self.classifier = nn.Linear(self.layers[-1].get(DIM), num_classes)

    def forward(self, input):
        output = []
        for i, layer in enumerate(self.layers):
            input = self.levels[i](input)
            if self.shortcut:
                output.append(self.heads[i](input))
        if not self.shortcut:
            output = self.heads[0](input)
        else:
            output = (
                reduce(
                    lambda a, b: b[:, : self.layers[-1].get(DIM)] * a
                                 + b[:, self.layers[-1].get(DIM):],
                    output[::-1],
                )
            )
        return self.classifier(output)

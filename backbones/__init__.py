import copy
import math

from einops import rearrange
from mmcv.runner import BaseModule
from mmdet.models.builder import BACKBONES as DET_BACKBONES
from mmseg.models.builder import BACKBONES as SEG_BACKBONES
from torch import nn

from libs.consts import ORIGINAL_TM, EMB_MIXER, SER_PM, EMB_CROSS_MLP
from libs.models import RaftMLP


class DetRaftMLP(RaftMLP, BaseModule):
    def __init__(self,
                 layers,
                 dropout,
                 token_mixing_type,
                 embedding_type,
                 drop_path_rate,
                 init_cfg,
                 ):
        super(DetRaftMLP, self).__init__(
            layers=layers,
            in_channels=3,
            pretrained_image_size=224,
            num_classes=1000,
            token_expansion_factor=2,
            channel_expansion_factor=4,
            dropout=dropout,
            token_mixing_type=token_mixing_type,
            embedding_type=embedding_type,
            shortcut=False,
            drop_path_rate=drop_path_rate
        )
        self._is_init = False
        self.init_cfg = copy.deepcopy(init_cfg)
        self.heads = nn.Identity()
        self.classifier = nn.Identity()
        self.features = []

    def forward(self, input):
        output = []

        def forward_hook(module, inputs, outputs):
            self.features.append(rearrange(outputs, "b (h w) c -> b c h w",
                                           h=math.ceil(self.levels[0].h / self.levels[0].patch_size),
                                           w=math.ceil(self.levels[0].w / self.levels[0].patch_size)).contiguous())

        handles = [self.levels[0].fn[i].register_forward_hook(forward_hook) for i in (1,)]
        for i, layer in enumerate(self.layers):
            input = self.levels[i](input)
            if i == 0:
                output = self.features
            else:
                output.append(input.contiguous())
        self.features = []
        for handle in handles:
            handle.remove()
        return tuple(output)


@DET_BACKBONES.register_module()
class DetRaftMLPSmall(DetRaftMLP):
    def __init__(self, *args, **kwargs):
        super(DetRaftMLPSmall, self).__init__(
            layers=[
                {"depth": 2, "dim": 64, "patch_size": 4, "raft_size": 2, "embedding_kernels": [4, 8]},
                {"depth": 2, "dim": 128, "patch_size": 2, "raft_size": 2, "embedding_kernels": [2, 4]},
                {"depth": 6, "dim": 256, "patch_size": 2, "raft_size": 2, "embedding_kernels": [2, 4]},
                {"depth": 2, "dim": 512, "patch_size": 2, "raft_size": 2, "embedding_kernels": [2]},
            ],
            dropout=0.,
            token_mixing_type=SER_PM,
            embedding_type=EMB_CROSS_MLP,
            drop_path_rate=0.,
            *args, **kwargs
        )


@DET_BACKBONES.register_module()
class DetRaftMLPMedium(DetRaftMLP):
    def __init__(self, *args, **kwargs):
        super(DetRaftMLPMedium, self).__init__(
            layers=[
                {"depth": 2, "dim": 96, "patch_size": 4, "raft_size": 2, "embedding_kernels": [4, 8]},
                {"depth": 2, "dim": 192, "patch_size": 2, "raft_size": 2, "embedding_kernels": [2, 4]},
                {"depth": 6, "dim": 384, "patch_size": 2, "raft_size": 2, "embedding_kernels": [2, 4]},
                {"depth": 2, "dim": 768, "patch_size": 2, "raft_size": 2, "embedding_kernels": [2]},
            ],
            dropout=0.,
            token_mixing_type=SER_PM,
            embedding_type=EMB_CROSS_MLP,
            drop_path_rate=0.,
            *args, **kwargs
        )


@DET_BACKBONES.register_module()
class DetRaftMLPLarge(DetRaftMLP):
    def __init__(self, *args, **kwargs):
        super(DetRaftMLPLarge, self).__init__(
            layers=[
                {"depth": 2, "dim": 128, "patch_size": 4, "raft_size": 2, "embedding_kernels": [4, 8]},
                {"depth": 2, "dim": 192, "patch_size": 2, "raft_size": 2, "embedding_kernels": [2, 4]},
                {"depth": 6, "dim": 512, "patch_size": 2, "raft_size": 2, "embedding_kernels": [2, 4]},
                {"depth": 2, "dim": 1024, "patch_size": 2, "raft_size": 2, "embedding_kernels": [2]},
            ],
            dropout=0.,
            token_mixing_type=SER_PM,
            embedding_type=EMB_CROSS_MLP,
            drop_path_rate=0.,
            *args, **kwargs
        )


@DET_BACKBONES.register_module()
class DetOrgMixer(DetRaftMLP):
    def __init__(self, *args, **kwargs):
        super(DetOrgMixer, self).__init__(
            layers=[
                {"depth": 12, "dim": 768, "patch_size": 16},
            ],
            dropout=0.,
            token_mixing_type=ORIGINAL_TM,
            embedding_type=EMB_MIXER,
            drop_path_rate=0.,
            *args, **kwargs
        )

    def forward(self, input):
        output = []

        def forward_hook(module, inputs, outputs):
            self.features.append(rearrange(outputs, "b (h w) c -> b c h w",
                                           h=math.ceil(self.levels[0].h / self.levels[0].patch_size),
                                           w=math.ceil(self.levels[0].w / self.levels[0].patch_size)).contiguous())

        handles = [self.levels[0].fn[i].register_forward_hook(forward_hook) for i in (1, 5, 11)]
        for i, layer in enumerate(self.layers):
            input = self.levels[i](input)
            if i == 0:
                output = self.features
            output.append(input.contiguous())
        self.features = []
        for handle in handles:
            handle.remove()
        return tuple(output)


class SegRaftMLP(RaftMLP, BaseModule):
    def __init__(self,
                 layers,
                 dropout,
                 token_mixing_type,
                 embedding_type,
                 drop_path_rate,
                 init_cfg,
                 ):
        super(SegRaftMLP, self).__init__(
            layers=layers,
            in_channels=3,
            pretrained_image_size=224,
            num_classes=1000,
            token_expansion_factor=2,
            channel_expansion_factor=4,
            dropout=dropout,
            token_mixing_type=token_mixing_type,
            embedding_type=embedding_type,
            shortcut=False,
            drop_path_rate=drop_path_rate
        )
        self._is_init = False
        self.init_cfg = copy.deepcopy(init_cfg)
        self.heads = nn.Identity()
        self.classifier = nn.Identity()
        self.features = []

    def forward(self, input):
        output = []

        def forward_hook(module, inputs, outputs):
            self.features.append(rearrange(outputs, "b (h w) c -> b c h w",
                                           h=math.ceil(self.levels[0].h / self.levels[0].patch_size),
                                           w=math.ceil(self.levels[0].w / self.levels[0].patch_size)).contiguous())

        handles = [self.levels[0].fn[i].register_forward_hook(forward_hook) for i in (1,)]
        for i, layer in enumerate(self.layers):
            input = self.levels[i](input)
            if i == 0:
                output = self.features
            else:
                output.append(input.contiguous())
        self.features = []
        for handle in handles:
            handle.remove()
        return tuple(output)


@SEG_BACKBONES.register_module()
class SegRaftMLPSmall(SegRaftMLP):
    def __init__(self, *args, **kwargs):
        super(SegRaftMLPSmall, self).__init__(
            layers=[
                {"depth": 2, "dim": 64, "patch_size": 4, "raft_size": 2, "embedding_kernels": [4, 8]},
                {"depth": 2, "dim": 128, "patch_size": 2, "raft_size": 2, "embedding_kernels": [2, 4]},
                {"depth": 6, "dim": 256, "patch_size": 2, "raft_size": 2, "embedding_kernels": [2, 4]},
                {"depth": 2, "dim": 512, "patch_size": 2, "raft_size": 2, "embedding_kernels": [2]},
            ],
            dropout=0.,
            token_mixing_type=SER_PM,
            embedding_type=EMB_CROSS_MLP,
            drop_path_rate=0.,
            *args, **kwargs
        )


@SEG_BACKBONES.register_module()
class SegRaftMLPMedium(SegRaftMLP):
    def __init__(self, *args, **kwargs):
        super(SegRaftMLPMedium, self).__init__(
            layers=[
                {"depth": 2, "dim": 96, "patch_size": 4, "raft_size": 2, "embedding_kernels": [4, 8]},
                {"depth": 2, "dim": 192, "patch_size": 2, "raft_size": 2, "embedding_kernels": [2, 4]},
                {"depth": 6, "dim": 384, "patch_size": 2, "raft_size": 2, "embedding_kernels": [2, 4]},
                {"depth": 2, "dim": 768, "patch_size": 2, "raft_size": 2, "embedding_kernels": [2]},
            ],
            dropout=0.,
            token_mixing_type=SER_PM,
            embedding_type=EMB_CROSS_MLP,
            drop_path_rate=0.,
            *args, **kwargs
        )


@SEG_BACKBONES.register_module()
class SegRaftMLPLarge(SegRaftMLP):
    def __init__(self, *args, **kwargs):
        super(SegRaftMLPLarge, self).__init__(
            layers=[
                {"depth": 2, "dim": 128, "patch_size": 4, "raft_size": 2, "embedding_kernels": [4, 8]},
                {"depth": 2, "dim": 192, "patch_size": 2, "raft_size": 2, "embedding_kernels": [2, 4]},
                {"depth": 6, "dim": 512, "patch_size": 2, "raft_size": 2, "embedding_kernels": [2, 4]},
                {"depth": 2, "dim": 1024, "patch_size": 2, "raft_size": 2, "embedding_kernels": [2]},
            ],
            dropout=0.,
            token_mixing_type=SER_PM,
            embedding_type=EMB_CROSS_MLP,
            drop_path_rate=0.,
            *args, **kwargs
        )


@SEG_BACKBONES.register_module()
class SegOrgMixer(SegRaftMLP):
    def __init__(self, *args, **kwargs):
        super(SegOrgMixer, self).__init__(
            layers=[
                {"depth": 12, "dim": 768, "patch_size": 16},
            ],
            dropout=0.,
            token_mixing_type=ORIGINAL_TM,
            embedding_type=EMB_MIXER,
            drop_path_rate=0.,
            *args, **kwargs
        )

    def forward(self, input):
        output = []

        def forward_hook(module, inputs, outputs):
            self.features.append(rearrange(outputs, "b (h w) c -> b c h w",
                                           h=math.ceil(self.levels[0].h / self.levels[0].patch_size),
                                           w=math.ceil(self.levels[0].w / self.levels[0].patch_size)).contiguous())

        handles = [self.levels[0].fn[i].register_forward_hook(forward_hook) for i in (1, 5, 11)]
        for i, layer in enumerate(self.layers):
            input = self.levels[i](input)
            if i == 0:
                output = self.features
            output.append(input.contiguous())
        self.features = []
        for handle in handles:
            handle.remove()
        return tuple(output)

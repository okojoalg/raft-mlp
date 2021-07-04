import hydra
import torch
from omegaconf import DictConfig, OmegaConf
import torchinfo
from ptflops import get_model_complexity_info

from libs.consts import IMAGENET, CIFAR10, CIFAR100
from libs.datasets import ImageNetGetter, CIFAR10Getter, CIFAR100Getter
from libs.models import SSCRMLP


@hydra.main(config_path="configs", config_name="config")
def run_summary(params: DictConfig) -> None:
    print(OmegaConf.to_yaml(params))
    OmegaConf.set_struct(params, True)
    if params.settings.dataset_name == IMAGENET:
        dg = ImageNetGetter()
    elif params.settings.dataset_name == CIFAR10:
        dg = CIFAR10Getter()
    elif params.settings.dataset_name == CIFAR100:
        dg = CIFAR100Getter()
    else:
        raise ValueError("Invalid dataset name")
    model = SSCRMLP(
        layers=params.settings.layers,
        in_channels=dg.channels,
        image_size=dg.image_size,
        num_classes=dg.num_classes,
        token_expansion_factor=params.settings.token_expansion_factor,
        channel_expansion_factor=params.settings.channel_expansion_factor,
        dropout=params.settings.dropout,
        token_mixing_type=params.settings.token_mixing_type,
        shortcut=params.settings.shortcut,
        gap=params.settings.gap,
    )
    input_size = (dg.channels, dg.image_size, dg.image_size)
    torchinfo.summary(model, input_size=(1, *input_size))
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(
            model,
            input_size,
            as_strings=True,
            print_per_layer_stat=True,
            verbose=True,
        )
        print("{:<30}  {:<8}".format("Computational complexity: ", macs))
        print("{:<30}  {:<8}".format("Number of parameters: ", params))


if __name__ == "__main__":
    run_summary()

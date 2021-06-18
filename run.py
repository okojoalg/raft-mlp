import hydra
from omegaconf import DictConfig, OmegaConf

from libs.train import main


@hydra.main(config_path="configs", config_name="config")
def run_experiment(params: DictConfig) -> None:
    print(OmegaConf.to_yaml(params))
    OmegaConf.set_struct(params, True)
    main(params)


if __name__ == "__main__":
    run_experiment()

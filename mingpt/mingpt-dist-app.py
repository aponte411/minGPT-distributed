import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path=".", config_name="gpt2_config")
def mingpt_dist_app(cfg : DictConfig) -> None:
    print(cfg)

if __name__ == "__main__":
    mingpt_dist_app()

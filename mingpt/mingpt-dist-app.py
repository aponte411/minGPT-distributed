import hydra
from omegaconf import DictConfig
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import random_split

from trainer import GPTTrainer, GPTTrainerConfig
from model import GPT, GPTConfig, OptimizerConfig, create_optimizer
from char_dataset import CharDataset

def get_resources(
    gpt_config: GPTConfig,
    optimizer_config: OptimizerConfig,
    data_config: DictConfig,
):
    data = CharDataset(data_config.path, data_config.block_size)
    train_size = int(len(data)*data_config.train_split)
    train_split, test_split = random_split(data,[train_size, len(data) - train_size])
    gpt_config.vocab_size = data.vocab_size
    gpt_config.block_size = data.block_size
    model = GPT(gpt_config)
    optimizer = create_optimizer(model, optimizer_config)
    return model, optimizer, train_split, test_split


@hydra.main(version_base=None, config_path=".", config_name="gpt2_config")
def mingpt_dist_app(cfg : DictConfig) -> None:
    """Entry point for distributed training GPT"""
    # Setup distributed training process group
    init_process_group(backend="nccl")
    # Setup configuration
    gpt_config = GPTConfig(**cfg["gpt_config"])
    optimizer_config = OptimizerConfig(**cfg["optimizer_config"])
    data_config = cfg["data_config"]
    trainer_config = GPTTrainerConfig(**cfg["trainer_config"])

    # Get resources and pass in configuration parameters
    model, optimizer, train, test = get_resources(
        gpt_config,
        optimizer_config,
        data_config,
    )
    # Setup trainer object
    trainer = GPTTrainer(
        trainer_config,
        model,
        optimizer,
        train,
        test,
    )
    # Train model for max epochs
    trainer.train(trainer_config.max_epochs)
    # Cleanup
    destroy_process_group()

if __name__ == "__main__":
    mingpt_dist_app()

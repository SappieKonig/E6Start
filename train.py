import hydra
import wandb
from omegaconf import DictConfig
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from dataset import AlbumentationsMNIST

@hydra.main(version_base=None, config_path="configs", config_name="base_config")
def train(cfg: DictConfig) -> None:
    augmentation = instantiate(cfg.augmentation)
    
    if cfg.wandb.enabled:
        wandb.init(project=cfg.wandb.project)
    
    # Instantiate components
    model = instantiate(cfg.model)
    train_dataset = AlbumentationsMNIST(train=True, transform=augmentation)
    postprocessing = instantiate(cfg.postprocessing)
    
    # Training loop placeholder
    for epoch in range(cfg.training.epochs):
        # Your training logic here
        pass
    
    if cfg.wandb.enabled:
        wandb.finish()

if __name__ == "__main__":
    train()
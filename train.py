import hydra
import wandb
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch
from torch.utils.data import DataLoader
from dataset import AlbumentationsMNIST
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np


@hydra.main(version_base=None, config_path="configs", config_name="base_config")
def train(cfg: DictConfig) -> None:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    if torch.backends.mps.is_available():
        device = "mps"

    # creating the augmentations is much cleaner now through hydra
    augmentation = instantiate(cfg.augmentation)
    
    if cfg.wandb.enabled:
        wandb.init(project=cfg.wandb.project)
    
    # Instantiate components
    model = instantiate(cfg.model).to(device)
    train_dataset = AlbumentationsMNIST(train=True, transform=augmentation)
    val_dataset = AlbumentationsMNIST(train=False, transform=augmentation)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.training.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.training.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    
    # Training loop placeholder
    for _ in range(cfg.training.epochs):
        acc = []
        train_loop = tqdm(train_dataloader)
        for images, labels in train_loop:
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)
            loss = F.cross_entropy(preds, labels)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            accuracy = torch.sum(torch.argmax(preds, dim=1) == labels).item() / len(labels)
            acc.append(accuracy)

            train_loop.set_postfix(loss=loss.item(), accuracy=np.mean(acc))

    
        acc = []
        val_loop = tqdm(val_dataloader)
        for images, labels in val_loop:
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)
            loss = F.cross_entropy(preds, labels)

            accuracy = torch.sum(torch.argmax(preds, dim=1) == labels).item() / len(labels)
            acc.append(accuracy)

            val_loop.set_postfix(loss=loss.item(), accuracy=np.mean(acc))

        
        if cfg.wandb.enabled:
            wandb.log({"train/accuracy": np.mean(acc)})
    
    if cfg.wandb.enabled:
        wandb.finish()

if __name__ == "__main__":
    train()
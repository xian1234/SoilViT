"""
SoilViT training script.

All hyperparameters are read from a YAML config file (default:
config/defaults.yaml).  Any value can be overridden on the command line:

    python train.py --config config/defaults.yaml \
                    --training.epochs 100 \
                    --optimizer.lr 5e-4

Usage
-----
    python train.py                          # use defaults
    python train.py --config my_config.yaml  # custom config
"""

import argparse
import copy
import os

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from LandVit import GeoDataset, SoilViT, SpectralConsistencyLoss


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def deep_set(cfg: dict, dotted_key: str, value: str):
    """Set cfg[a][b][c] from dotted key 'a.b.c', auto-casting the value."""
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node[k]
    leaf = keys[-1]
    orig = node.get(leaf)
    # cast to original type when possible
    if orig is None or isinstance(orig, str):
        node[leaf] = value
    elif isinstance(orig, bool):
        node[leaf] = value.lower() in ("1", "true", "yes")
    elif isinstance(orig, int):
        node[leaf] = int(value)
    elif isinstance(orig, float):
        node[leaf] = float(value)
    elif isinstance(orig, list):
        node[leaf] = yaml.safe_load(value)
    else:
        node[leaf] = value


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config/defaults.yaml",
                   help="Path to YAML config file")
    p.add_argument("overrides", nargs="*",
                   help="key=value pairs to override config, e.g. optimizer.lr=1e-3")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class WeightedL1Loss(nn.Module):
    def __init__(self, std_values: list, device: torch.device):
        super().__init__()
        weights = 1.0 / (torch.tensor(std_values, dtype=torch.float32) + 1e-8)
        weights = weights / weights.sum()
        self.register_buffer("weights", weights.to(device))

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        l1 = torch.abs(y_pred - y_true)
        return (l1 * self.weights.view(1, -1, 1, 1)).mean()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg: dict,
                 device: torch.device, pretrained_weights=None):
        self.model = model.to(device)
        if pretrained_weights is not None:
            self.model.load_state_dict(pretrained_weights, strict=False)

        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.save_path    = cfg["training"]["save_path"]

        opt_cfg  = cfg["optimizer"]
        sch_cfg  = cfg["scheduler"]
        loss_cfg = cfg["loss"]

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg["weight_decay"],
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=sch_cfg["T_max"]
        )
        self.criterion  = WeightedL1Loss(loss_cfg["weighted_l1_stds"], device)
        self.val_loss   = nn.L1Loss()
        self.best_val   = float("inf")

    def train_epoch(self) -> float:
        self.model.train()
        total = 0.0
        for x, y, lat_lon in tqdm(self.train_loader, leave=False):
            x, y, lat_lon = x.to(self.device), y.to(self.device), lat_lon.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(x, lat_lon), y)
            loss.backward()
            self.optimizer.step()
            total += loss.item()
        self.scheduler.step()
        return total / len(self.train_loader)

    def validate(self) -> float:
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for x, y, lat_lon in self.val_loader:
                x, y, lat_lon = x.to(self.device), y.to(self.device), lat_lon.to(self.device)
                total += self.val_loss(self.model(x, lat_lon), y).item()
        avg = total / len(self.val_loader)
        if avg < self.best_val:
            self.best_val = avg
            torch.save(self.model.state_dict(), self.save_path)
        return avg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg  = load_config(args.config)

    for override in args.overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got: {override!r}")
        k, v = override.split("=", 1)
        deep_set(cfg, k, v)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {args.config}")

    mcfg = cfg["model"]
    dcfg = cfg["data"]
    tcfg = cfg["training"]

    model = SoilViT(
        img_size=mcfg["img_size"],
        patch_size=mcfg["patch_size"],
        in_chans=mcfg["in_chans"],
        out_chans=mcfg["out_chans"],
        embed_dim=mcfg["embed_dim"],
        depth=mcfg["depth"],
        num_heads=mcfg["num_heads"],
        dropout=mcfg["dropout"],
        geo_max_value=mcfg["geo_max_value"],
    )

    pretrained = None
    if tcfg["pretrained_weights"]:
        pretrained = torch.load(tcfg["pretrained_weights"], map_location=device)

    train_loader = DataLoader(
        GeoDataset(dcfg["train_json"], output_band_indices=dcfg["output_band_indices"]),
        batch_size=tcfg["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        GeoDataset(dcfg["val_json"], output_band_indices=dcfg["output_band_indices"]),
        batch_size=tcfg["val_batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    trainer = Trainer(model, train_loader, val_loader, cfg, device, pretrained)

    for epoch in range(1, tcfg["epochs"] + 1):
        train_loss = trainer.train_epoch()
        val_loss   = trainer.validate()
        print(f"Epoch {epoch:3d}/{tcfg['epochs']}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"best={trainer.best_val:.4f}")


if __name__ == "__main__":
    main()

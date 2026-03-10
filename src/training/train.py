"""
src/training/train.py

Main training script for CBCT tooth segmentation.

Usage:
    python src/training/train.py --config configs/train_config.yaml
    python src/training/train.py --config configs/train_config.yaml --gpus 4
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
from tqdm import tqdm
import yaml

# Project imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.nnunet_resencl import build_model
from src.preprocessing.dataset import build_datasets
from src.preprocessing.transforms import get_training_transforms, get_val_transforms
from src.training.losses import build_loss
from src.training.metrics import batch_dice_torch, MetricAggregator



# Reproduccibility 
def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# lr schedular 

class PolynomialLR:
    """Polynomial decs: lr =lr0*(1-epoch)/total_epochs)*exp"""

    def __init__(self,optimizer,total_epochs:int,exponent:float=0.9):
        self.optimizer=optimizer
        self.total_epochs=total_epochs
        self.exp=exponent
        self.base_lrs=[pg["lr"]for pg in optimizer.param_groups]

    def step(self,epoch:int):
        factor=(1-epoch/self.total_epochs)**self.exp
        for pg , base_lr in zip(self.optimizer.param_groups,self.base_lrs):
            pg["lr"]=base_lr*factor


# Trainer 

class Trainer:
    def __init__(self,config:dict):
        self.cfg=config
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        set_seed(config.get("seed", 42))

        # Directores 
        self.ckpt_dir = Path(config["logging"]["checkpoint_dir"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.ckpt_dir / "tensorboard"
        self.log_dir.mkdir(parents=True, exist_ok=True)


        # Data 
        logger.info("Building datasets ...")
        ds_train, ds_val, _ = build_datasets(
            processed_dir=config["dataset"]["processed_dir"],
            splits_dir=config["dataset"]["splits_dir"],
            patch_size=tuple(config["training"]["patch_size"]),
            transforms_train=get_training_transforms(),
            transforms_val=get_val_transforms(),
        )
        hw = config["hardware"]
        self.train_loader = DataLoader(
            ds_train,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=hw.get("workers", 4),
            pin_memory=hw.get("pin_memory", True),
            persistent_workers=hw.get("persistent_workers", True),
            drop_last=True,
        )
        self.val_loader = DataLoader(
            ds_val,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        logger.info(f"Train: {len(ds_train)} cases | Val: {len(ds_val)} cases")


        # modesl 
        self.model = build_model(config["model"]).to(self.device)
        n_params = self.model.count_parameters()
        logger.info(f"Model: ResEncL | Parameters: {n_params:,}")

        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs (DataParallel)")
            self.model = nn.DataParallel(self.model)

        # loss and optimizer 
        self.criterion = build_loss(config, config["dataset"]["num_classes"])

        # ── Optimizer ───────────────────────────────────────
        train_cfg = config["training"]
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=train_cfg["lr"],
            momentum=train_cfg.get("momentum", 0.99),
            weight_decay=train_cfg.get("weight_decay", 3e-5),
            nesterov=True,
        )
        self.scheduler = PolynomialLR(
            self.optimizer,
            total_epochs=train_cfg["epochs"],
            exponent=train_cfg.get("poly_exp", 0.9),
        )
        self.scaler = GradScaler(enabled=train_cfg.get("mixed_precision", True))

        # ── Logging ─────────────────────────────────────────
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.best_dice = 0.0
        self.global_step = 0
    
    def train(self):
        cfg = self.cfg
        n_epochs = cfg["training"]["epochs"]
        val_every = cfg["logging"]["val_every_n_epochs"]
        log_every = cfg["logging"]["log_every_n_steps"]
        grad_clip = cfg["training"].get("grad_clip", 1.0)

        logger.info(f"Starting training for {n_epochs} epochs ....")
        for epoch in range(1,n_epochs+1):
            self.model.train()
            epoch_loss = 0.0
            epoch_dice = 0.0
            n_batches=0
            t0=time.time()


            for batch in self.train_loader:
                images = batch["image"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=cfg["training"].get("mixed_precision", True)):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()


                if isinstance(outputs,list):
                    pred_logits=outputs[0]
                else:
                    pred_logits=outputs
                with torch.no_grad():
                    batch_dice=batch_dice_torch(pred_logits,labels,
                                                num_classes=cfg["dataset"]["num_classes"]).item()
                    

                epoch_loss+=loss.item()
                epoch_dice+=batch_dice
                n_batches+=1
                self.global_step+=1

                if self.global_step % log_every == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                    self.writer.add_scalar("train/dice", batch_dice, self.global_step)
                    self.writer.add_scalar("train/lr", lr, self.global_step)
            

            self.scheduler.step(epoch)

            elapsed = time.time() - t0
            avg_loss = epoch_loss / max(n_batches, 1)
            avg_dice = epoch_dice / max(n_batches, 1)
            lr_now = self.optimizer.param_groups[0]["lr"]

            logger.info(
                f"Epoch {epoch:04d}/{n_epochs} | "
                f"Loss: {avg_loss:.4f} | Dice: {avg_dice:.4f} | "
                f"LR: {lr_now:.2e} | Time: {elapsed:.1f}s"
            )

            # ── Validation ──────────────────────────────────
            if epoch % val_every == 0 or epoch == n_epochs:
                val_dice = self.validate(epoch)
                self.writer.add_scalar("val/mean_dice", val_dice, epoch)
                self._save_checkpoint(epoch, val_dice)


        logger.infor("Training Complete. ")
        self.writer.close()

    # Validation loop
    def validate(self,epoch:int)->float:
        self.model.eval()
        total_dice=0.0
        n_cases=0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Val @ epoch {epoch}", leave=False):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                # Inference mode (no deep supervision)
                if isinstance(self.model, nn.DataParallel):
                    self.model.module.deep_supervision = False
                else:
                    self.model.deep_supervision = False

                with autocast():
                    outputs = self.model(images)

                if isinstance(self.model, nn.DataParallel):
                    self.model.module.deep_supervision = True
                else:
                    self.model.deep_supervision = True

                dice = batch_dice_torch(
                    outputs, labels,
                    num_classes=self.cfg["dataset"]["num_classes"]
                ).item()
                total_dice += dice
                n_cases += 1

        mean_dice = total_dice / max(n_cases, 1)
        logger.info(f"  Validation Mean Dice: {mean_dice:.4f}")
        return mean_dice
    
    def _save_checkpoint(self, epoch:int, val_dice:float):
        model_state = (
            self.model.module.state_dict()
            if isinstance(self.model, nn.DataParallel)
            else self.model.state_dict()
        )

        payload = {
            "epoch": epoch,
            "val_dice": val_dice,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.cfg,
        }


        # Save last
        last_path = self.ckpt_dir / "last_model.pth"
        torch.save(payload, last_path)

        # Save best
        if val_dice > self.best_dice:
            self.best_dice = val_dice
            best_path = self.ckpt_dir / "best_model.pth"
            torch.save(payload, best_path)
            logger.info(f"  ✨ New best model saved (Dice: {val_dice:.4f}) → {best_path}")



#Cli 


def parse_args():
    p = argparse.ArgumentParser(description="Train CBCT segmentation model")
    p.add_argument("--config", default="configs/train_config.yaml")
    p.add_argument("--gpus", type=int, default=None,
                   help="Override number of GPUs (for CUDA_VISIBLE_DEVICES)")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    return p.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(args.gpus))

    trainer = Trainer(config)

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        trainer.model.load_state_dict(ckpt["model_state_dict"])
        trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logger.info(f"Resumed from {args.resume} (epoch {ckpt['epoch']})")

    trainer.train()


if __name__ == "__main__":
    main()

                






"""
Full Rail Corrugation Model with PyTorch Lightning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from typing import Dict, Any, List, Optional

from models.backbone import MultiScaleCNN
from models.heads import SideClassificationHead, DecisionFusion
from utils.metrics import MacroF1Metric, PerClassF1Metric
from .losses import FocalLoss


class RailCorrugationModel(pl.LightningModule):
    """Full model for rail corrugation detection."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        
        # Model components
        self.backbone = MultiScaleCNN(config)
        self.head_i = SideClassificationHead(config)
        self.head_ii = SideClassificationHead(config)
        self.fusion = DecisionFusion(config.model.num_classes)
        
        # Loss functions
        class_weights = torch.tensor(config.loss.class_weights, dtype=torch.float32)
        self.criterion = FocalLoss(
            gamma=config.loss.focal_gamma,
            alpha=class_weights
        )
        self.aux_criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Metrics
        self.train_macro_f1 = MacroF1Metric(num_classes=config.model.num_classes)
        self.val_macro_f1 = MacroF1Metric(num_classes=config.model.num_classes)
        self.val_per_class_f1 = PerClassF1Metric(num_classes=config.model.num_classes)
        
        # For logging
        self.validation_outputs = []
    
    def forward(self, side_i: torch.Tensor, side_ii: torch.Tensor) -> tuple:
        """
        Args:
            side_i: [B, 64, T] - Side I signals
            side_ii: [B, 64, T] - Side II signals
        Returns:
            logits_i: [B, 3] - Side I head logits
            logits_ii: [B, 3] - Side II head logits
            fused_logits: [B, 3] - Fused logits
        """
        # Shared backbone
        emb_i = self.backbone(side_i)
        emb_ii = self.backbone(side_ii)
        
        # Side-specific heads
        logits_i = self.head_i(emb_i)
        logits_ii = self.head_ii(emb_ii)
        
        # Fuse predictions
        fused_logits = self.fusion(logits_i, logits_ii)
        
        return logits_i, logits_ii, fused_logits
    
    def training_step(self, batch, batch_idx):
        side_i, side_ii, labels = batch
        logits_i, logits_ii, fused_logits = self(side_i, side_ii)
        
        # Main loss (fused prediction)
        loss_main = self.criterion(fused_logits, labels)
        
        # Auxiliary side losses
        loss_i = self.aux_criterion(logits_i, labels)
        loss_ii = self.aux_criterion(logits_ii, labels)
        
        # Combined loss
        loss = loss_main + self.config.loss.aux_weight * (loss_i + loss_ii)
        
        # Metrics
        self.train_macro_f1(fused_logits, labels)
        
        # Logging
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_loss_main", loss_main, prog_bar=False)
        self.log("train_loss_aux", loss_i + loss_ii, prog_bar=False)
        self.log("train_macro_f1", self.train_macro_f1, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        side_i, side_ii, labels = batch
        logits_i, logits_ii, fused_logits = self(side_i, side_ii)
        
        loss_main = self.criterion(fused_logits, labels)
        loss_i = self.aux_criterion(logits_i, labels)
        loss_ii = self.aux_criterion(logits_ii, labels)
        loss = loss_main + self.config.loss.aux_weight * (loss_i + loss_ii)
        
        # Metrics
        self.val_macro_f1(fused_logits, labels)
        self.val_per_class_f1(fused_logits, labels)
        
        # Store for epoch-end logging
        self.validation_outputs.append({
            "val_loss": loss,
            "val_loss_main": loss_main,
            "logits": fused_logits.detach(),
            "labels": labels.detach(),
        })
        
        return loss
    
    def on_validation_epoch_end(self):
        if not self.validation_outputs:
            return
        
        # Compute per-class F1
        per_class_f1 = self.val_per_class_f1.compute()
        class_names = ["Normal", "Side I", "Side II"]
        
        for i, (name, f1) in enumerate(zip(class_names, per_class_f1)):
            self.log(f"val_f1_{name}", f1, prog_bar=False)
        
        macro_f1 = self.val_macro_f1.compute()
        self.log("val_macro_f1", macro_f1, prog_bar=True)
        
        # Log loss
        avg_loss = torch.stack([x["val_loss"] for x in self.validation_outputs]).mean()
        self.log("val_loss", avg_loss, prog_bar=True)
        
        # Clear outputs
        self.validation_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.train.lr,
            weight_decay=self.config.train.weight_decay,
        )
        
        # Cosine annealing with warmup
        def lr_lambda(epoch):
            if epoch < self.config.train.warmup_epochs:
                return epoch / max(1, self.config.train.warmup_epochs)
            # Cosine annealing after warmup
            progress = (epoch - self.config.train.warmup_epochs) / \
                      max(1, self.config.train.max_epochs - self.config.train.warmup_epochs)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159))).item()
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        side_i, side_ii = batch
        logits_i, logits_ii, fused_logits = self(side_i, side_ii)
        probs = F.softmax(fused_logits, dim=1)
        return probs
    
    def get_embeddings(self, side_i: torch.Tensor, side_ii: torch.Tensor) -> tuple:
        """Get side embeddings for analysis."""
        with torch.no_grad():
            emb_i = self.backbone(side_i)
            emb_ii = self.backbone(side_ii)
        return emb_i, emb_ii
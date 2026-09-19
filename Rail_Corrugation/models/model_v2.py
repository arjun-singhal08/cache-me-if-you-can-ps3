"""
Enhanced Rail Corrugation Model with handcrafted feature fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from typing import Dict, Any, List, Optional

from models.backbone import MultiScaleCNN
from models.heads import SideClassificationHead, DecisionFusion
from utils.metrics import MacroF1Metric, PerClassF1Metric
from losses import FocalLoss


class FeatureEncoder(nn.Module):
    """MLP encoder for handcrafted features."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class EnhancedSideHead(nn.Module):
    """Enhanced classification head combining deep and handcrafted features."""
    
    def __init__(self, deep_dim: int, handcrafted_dim: int, hidden_dim: int = 256, num_classes: int = 3, dropout: float = 0.3):
        super().__init__()
        combined_dim = deep_dim + handcrafted_dim
        
        self.head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(self, deep_emb: torch.Tensor, handcrafted_emb: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([deep_emb, handcrafted_emb], dim=1)
        return self.head(combined)


class EnhancedRailCorrugationModel(pl.LightningModule):
    """Enhanced model combining CNN backbone with handcrafted features."""
    
    def __init__(self, config: Dict[str, Any], handcrafted_dim: int = 200):
        super().__init__()
        self.save_hyperparameters({k: v for k, v in config.items() if k != 'data'})
        self.config = config
        self.handcrafted_dim = handcrafted_dim
        
        # Deep learning backbone
        self.backbone = MultiScaleCNN(config)
        deep_emb_dim = config.model.attention_dim  # 256
        
        # Handcrafted feature encoder
        self.feature_encoder = FeatureEncoder(
            input_dim=handcrafted_dim,
            hidden_dim=256,
            output_dim=128,
            dropout=config.model.dropout
        )
        
        # Enhanced heads
        self.head_i = EnhancedSideHead(
            deep_dim=deep_emb_dim,
            handcrafted_dim=128,
            hidden_dim=256,
            num_classes=config.model.num_classes,
            dropout=config.model.dropout
        )
        self.head_ii = EnhancedSideHead(
            deep_dim=deep_emb_dim,
            handcrafted_dim=128,
            hidden_dim=256,
            num_classes=config.model.num_classes,
            dropout=config.model.dropout
        )
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
        
        self.validation_outputs = []
    
    def forward(self, side_i: torch.Tensor, side_ii: torch.Tensor, 
                features_i: torch.Tensor = None, features_ii: torch.Tensor = None) -> tuple:
        """
        Args:
            side_i: [B, 64, T] - Side I signals
            side_ii: [B, 64, T] - Side II signals
            features_i: [B, handcrafted_dim] - Side I handcrafted features
            features_ii: [B, handcrafted_dim] - Side II handcrafted features
        Returns:
            logits_i: [B, 3] - Side I head logits
            logits_ii: [B, 3] - Side II head logits
            fused_logits: [B, 3] - Fused logits
        """
        # Deep embeddings
        deep_emb_i = self.backbone(side_i)
        deep_emb_ii = self.backbone(side_ii)
        
        # Handcrafted embeddings
        if features_i is not None and features_ii is not None:
            hand_emb_i = self.feature_encoder(features_i)
            hand_emb_ii = self.feature_encoder(features_ii)
        else:
            # Zero embeddings if features not provided
            batch_size = deep_emb_i.size(0)
            hand_emb_i = torch.zeros(batch_size, 128, device=deep_emb_i.device)
            hand_emb_ii = torch.zeros(batch_size, 128, device=deep_emb_ii.device)
        
        # Enhanced heads
        logits_i = self.head_i(deep_emb_i, hand_emb_i)
        logits_ii = self.head_ii(deep_emb_ii, hand_emb_ii)
        
        # Fuse predictions
        fused_logits = self.fusion(logits_i, logits_ii)
        
        return logits_i, logits_ii, fused_logits
    
    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            side_i, side_ii, labels = batch
            features_i = features_ii = None
        else:
            side_i, side_ii, features_i, features_ii, labels = batch
        
        logits_i, logits_ii, fused_logits = self(side_i, side_ii, features_i, features_ii)
        
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
        if len(batch) == 3:
            side_i, side_ii, labels = batch
            features_i = features_ii = None
        else:
            side_i, side_ii, features_i, features_ii, labels = batch
        
        logits_i, logits_ii, fused_logits = self(side_i, side_ii, features_i, features_ii)
        
        loss_main = self.criterion(fused_logits, labels)
        loss_i = self.aux_criterion(logits_i, labels)
        loss_ii = self.aux_criterion(logits_ii, labels)
        loss = loss_main + self.config.loss.aux_weight * (loss_i + loss_ii)
        
        # Metrics
        self.val_macro_f1(fused_logits, labels)
        self.val_per_class_f1(fused_logits, labels)
        
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
        
        per_class_f1 = self.val_per_class_f1.compute()
        class_names = ["Normal", "Side I", "Side II"]
        
        for i, (name, f1) in enumerate(zip(class_names, per_class_f1)):
            self.log(f"val_f1_{name}", f1, prog_bar=False)
        
        macro_f1 = self.val_macro_f1.compute()
        self.log("val_macro_f1", macro_f1, prog_bar=True)
        
        avg_loss = torch.stack([x["val_loss"] for x in self.validation_outputs]).mean()
        self.log("val_loss", avg_loss, prog_bar=True)
        
        self.validation_outputs.clear()
    
    def configure_optimizers(self):
        # Different learning rates for backbone vs heads
        backbone_params = list(self.backbone.parameters())
        head_params = list(self.head_i.parameters()) + list(self.head_ii.parameters()) + list(self.feature_encoder.parameters())
        
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': self.config.train.lr * 0.5},
            {'params': head_params, 'lr': self.config.train.lr},
        ], weight_decay=self.config.train.weight_decay)
        
        # Cosine annealing with warmup
        def lr_lambda(epoch):
            if epoch < self.config.train.warmup_epochs:
                return epoch / max(1, self.config.train.warmup_epochs)
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
        if len(batch) == 2:
            side_i, side_ii = batch
            features_i = features_ii = None
        else:
            side_i, side_ii, features_i, features_ii = batch
        
        logits_i, logits_ii, fused_logits = self(side_i, side_ii, features_i, features_ii)
        probs = F.softmax(fused_logits, dim=1)
        return probs


# Backward compatibility
RailCorrugationModel = EnhancedRailCorrugationModel
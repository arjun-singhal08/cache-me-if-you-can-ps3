"""
Classification heads for side-specific prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SideClassificationHead(nn.Module):
    """Side-specific classification head."""
    
    def __init__(self, config):
        super().__init__()
        attn_dim = config.model.attention_dim
        num_classes = config.model.num_classes
        dropout = config.model.dropout
        
        self.head = nn.Sequential(
            nn.Linear(attn_dim, attn_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(attn_dim // 2, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, attn_dim]
        return self.head(x)  # [B, num_classes]


class DecisionFusion(nn.Module):
    """
    Fuse side-specific predictions into final 3-class prediction.
    
    Logic:
    - Normal: both sides normal
    - Side I: side_i predicts Side I AND side_ii predicts Normal
    - Side II: side_ii predicts Side II AND side_i predicts Normal
    - Conflict: take max confidence
    """
    
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes
    
    def forward(self, logits_i: torch.Tensor, logits_ii: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits_i: [B, 3] - Side I head logits
            logits_ii: [B, 3] - Side II head logits
        Returns:
            fused_logits: [B, 3] - Fused logits for final prediction
        """
        prob_i = F.softmax(logits_i, dim=1)  # [B, 3]
        prob_ii = F.softmax(logits_ii, dim=1)  # [B, 3]
        
        # Class indices: 0=Normal, 1=Side I, 2=Side II
        fused = torch.zeros_like(prob_i)
        
        # Normal: both sides normal
        fused[:, 0] = prob_i[:, 0] * prob_ii[:, 0]
        
        # Side I: side_i says Side I, side_ii says Normal
        fused[:, 1] = prob_i[:, 1] * prob_ii[:, 0]
        
        # Side II: side_ii says Side II, side_i says Normal
        fused[:, 2] = prob_ii[:, 2] * prob_i[:, 0]
        
        # Handle conflicts (both sides indicate fault)
        # Side I fault detected but side_ii not normal
        conflict_i = (prob_i[:, 1] > prob_i[:, 0]) & (prob_ii[:, 0] < 0.5)
        fused[conflict_i, 1] = prob_i[conflict_i, 1]
        
        # Side II fault detected but side_i not normal
        conflict_ii = (prob_ii[:, 2] > prob_ii[:, 0]) & (prob_i[:, 0] < 0.5)
        fused[conflict_ii, 2] = prob_ii[conflict_ii, 2]
        
        # Both sides fault - take max confidence
        both_fault = (prob_i[:, 1] > prob_i[:, 0]) & (prob_ii[:, 2] > prob_ii[:, 0])
        if both_fault.any():
            # Compare confidence
            conf_i = prob_i[both_fault, 1]
            conf_ii = prob_ii[both_fault, 2]
            choose_i = conf_i > conf_ii
            
            fused[both_fault, 1] = torch.where(choose_i, conf_i, torch.zeros_like(conf_i))
            fused[both_fault, 2] = torch.where(~choose_i, conf_ii, torch.zeros_like(conf_ii))
        
        # Convert back to logits (add small epsilon for numerical stability)
        fused = torch.clamp(fused, min=1e-8)
        return torch.log(fused)
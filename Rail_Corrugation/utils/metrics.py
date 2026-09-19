"""
Metrics for rail corrugation detection.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional
from torchmetrics import Metric


class MacroF1Metric(Metric):
    """
    Macro F1 score across 3 classes (Normal, Side I, Side II).
    Computes F1 per class independently, then averages (unweighted).
    """
    
    def __init__(self, num_classes: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        
        # State for accumulating predictions
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            preds: [B, num_classes] logits or probabilities
            targets: [B] class indices
        """
        if preds.dim() == 2 and preds.size(1) == self.num_classes:
            # Convert logits/probs to class indices
            preds = preds.argmax(dim=1)
        
        self.preds.append(preds)
        self.targets.append(targets)
    
    def compute(self) -> torch.Tensor:
        if len(self.preds) == 0:
            return torch.tensor(0.0, device=self.device)
        
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)
        
        macro_f1 = 0.0
        for c in range(self.num_classes):
            tp = ((preds == c) & (targets == c)).sum().float()
            fp = ((preds == c) & (targets != c)).sum().float()
            fn = ((preds != c) & (targets == c)).sum().float()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            macro_f1 += f1
        
        return macro_f1 / self.num_classes


class PerClassF1Metric(Metric):
    """Per-class F1 scores for detailed monitoring."""
    
    def __init__(self, num_classes: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        if preds.dim() == 2 and preds.size(1) == self.num_classes:
            preds = preds.argmax(dim=1)
        self.preds.append(preds)
        self.targets.append(targets)
    
    def compute(self) -> List[torch.Tensor]:
        if len(self.preds) == 0:
            return [torch.tensor(0.0, device=self.device)] * self.num_classes
        
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)
        
        f1s = []
        for c in range(self.num_classes):
            tp = ((preds == c) & (targets == c)).sum().float()
            fp = ((preds == c) & (targets != c)).sum().float()
            fn = ((preds != c) & (targets == c)).sum().float()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            f1s.append(f1)
        
        return f1s


def compute_macro_f1(preds: np.ndarray, targets: np.ndarray, num_classes: int = 3) -> float:
    """Numpy version for validation."""
    from sklearn.metrics import f1_score
    return f1_score(targets, preds, average='macro', labels=list(range(num_classes)))


def compute_per_class_f1(preds: np.ndarray, targets: np.ndarray, num_classes: int = 3) -> List[float]:
    """Numpy version for validation."""
    from sklearn.metrics import f1_score
    return f1_score(targets, preds, average=None, labels=list(range(num_classes))).tolist()
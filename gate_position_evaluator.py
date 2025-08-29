#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gate Position Prediction Task Evaluator (Classification Version)

Implements region-based classification evaluation for Gate position prediction:
- Unified interface: metrics = evaluator.eval(y_pred, y_true)
- Classification metrics: Accuracy, F1-score, Top-k accuracy, Precision, Recall
- Supports region-based classification input formats
- EDA-specific spatial distribution analysis

Author: EDA for AI Team
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
import warnings
try:
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
except ImportError:
    print("Warning: sklearn not available, some metrics will be disabled")
    f1_score = precision_score = recall_score = accuracy_score = None

warnings.filterwarnings('ignore')

# ============================================================================
# Core Evaluator Module
# ============================================================================
        
class GatePositionEvaluator:
    """
    Gate Position Prediction Evaluator (Region Classification Version)
    
    Focuses on region-based classification evaluation with unified eval(y_pred, y_true) interface:
    - Core classification metrics: Accuracy, F1-score, Precision, Recall
    - Top-k accuracy metrics for multi-class evaluation
    - EDA-specific spatial distribution analysis
    - Region congestion and uniformity metrics
    """
    
    def __init__(self, num_classes: Optional[int] = None, grid_size: Tuple[int, int] = (8, 8)):
        """
        Initialize evaluator
        
        Args:
            num_classes: Number of classification classes (auto-detected if None)
            grid_size: Grid dimensions (rows, cols) for spatial analysis
        """
        self.num_classes = num_classes
        self.grid_size = grid_size
        if num_classes is None:
            self.num_classes = grid_size[0] * grid_size[1]  # Default to grid size
    
    def eval(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
        """
        Main evaluation function for region classification tasks
        
        Args:
            y_pred: Prediction results - can be:
                   - logits format: [num_samples, num_classes] 
                   - region IDs: [num_samples] (integer region indices)
            y_true: True region labels [num_samples] (integer region indices)
            
        Returns:
            Dictionary containing classification evaluation metrics
        """
        # Input format processing and validation
        pred_classes, true_classes, pred_logits = self._process_inputs(y_pred, y_true)
        
        metrics = {}
        
        # 1. Core classification metrics
        metrics.update(self._compute_classification_metrics(pred_classes, true_classes))
        
        # 2. Top-k accuracy metrics (if logits available)
        # if pred_logits is not None:
        #     metrics.update(self._compute_topk_metrics(pred_logits, true_classes))
        
        # 3. EDA-specific spatial distribution analysis
        # metrics.update(self._compute_spatial_metrics(pred_classes, true_classes))
        
        # 4. Region congestion analysis
        metrics.update(self._compute_congestion_metrics(pred_classes, true_classes))
        
        return metrics
    
    def _process_inputs(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Process and validate input formats
        
        Args:
            y_pred: Prediction results (logits or region IDs)
            y_true: True region labels
            
        Returns:
            (pred_classes, true_classes, pred_logits): Processed tensors
        """
        pred_logits = None
        
        # Process prediction results
        if y_pred.dim() == 2:  # logits format [num_samples, num_classes]
            pred_logits = y_pred
            # print(f"pred_logits:{pred_logits}")
            # assert 0
            pred_classes = torch.argmax(y_pred, dim=1)
            if self.num_classes is None:
                self.num_classes = y_pred.size(1)
        elif y_pred.dim() == 1:  # region ID format [num_samples]
            pred_classes = y_pred.long()
            # print(f"pred_classes.unique():{pred_classes.unique()}")
            # assert 0
        else:
            raise ValueError(f"Unsupported prediction dimension: {y_pred.dim()}, expected 1D or 2D")
            
        # Process true labels
        true_classes = y_true.squeeze().long()
        # print(f"true_classes.unique():{true_classes.unique()}")
        # assert 0
        
        # Auto-detect num_classes if not set
        if self.num_classes is None:
            max_class = max(torch.max(pred_classes).item(), torch.max(true_classes).item())
            self.num_classes = max_class + 1
        
        # Validate class ranges
        pred_classes = torch.clamp(pred_classes, 0, self.num_classes - 1)
        true_classes = torch.clamp(true_classes, 0, self.num_classes - 1)
        
        # Validate shape matching
        if pred_classes.shape != true_classes.shape:
            raise ValueError(f"Prediction and true label shapes mismatch: {pred_classes.shape} vs {true_classes.shape}")
            
        return pred_classes, true_classes, pred_logits
    
    def _compute_classification_metrics(self, pred_classes: torch.Tensor, 
                                      true_classes: torch.Tensor) -> Dict[str, float]:
        """
        Compute core classification metrics
        
        Returns:
            Dictionary with accuracy, f1_macro, f1_micro, precision, recall
        """
        # Convert to numpy for sklearn metrics
        pred_np = pred_classes.cpu().numpy()
        true_np = true_classes.cpu().numpy()
        
        # 1. Accuracy (PyTorch implementation for reliability)
        accuracy = (pred_classes == true_classes).float().mean().item()
        
        metrics = {'accuracy': accuracy}
        
        # 2. F1 scores, Precision, Recall (sklearn implementation if available)
        if f1_score is not None:
            try:
                metrics.update({
                    'f1_macro': f1_score(true_np, pred_np, average='macro', zero_division=0),
                    'f1_micro': f1_score(true_np, pred_np, average='micro', zero_division=0),
                    'f1_weighted': f1_score(true_np, pred_np, average='weighted', zero_division=0),
                    'precision_macro': precision_score(true_np, pred_np, average='macro', zero_division=0),
                    'recall_macro': recall_score(true_np, pred_np, average='macro', zero_division=0)
                })
            except Exception as e:
                warnings.warn(f"Sklearn metrics calculation failed: {e}")
                metrics.update({
                    'f1_macro': 0.0, 'f1_micro': 0.0, 'f1_weighted': 0.0,
                    'precision_macro': 0.0, 'recall_macro': 0.0
                })
        else:
            # Fallback: simple per-class metrics
            metrics.update(self._compute_simple_metrics(pred_classes, true_classes))
        
        return metrics
    
    def _compute_simple_metrics(self, pred_classes: torch.Tensor, 
                              true_classes: torch.Tensor) -> Dict[str, float]:
        """
        Compute simple classification metrics without sklearn
        """
        # Per-class precision and recall
        precisions = []
        recalls = []
        
        for class_id in range(self.num_classes):
            # True positives, false positives, false negatives
            tp = ((pred_classes == class_id) & (true_classes == class_id)).sum().item()
            fp = ((pred_classes == class_id) & (true_classes != class_id)).sum().item()
            fn = ((pred_classes != class_id) & (true_classes == class_id)).sum().item()
            
            # Precision and recall for this class
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Macro averages
        precision_macro = np.mean(precisions)
        recall_macro = np.mean(recalls)
        
        # F1 scores
        f1_scores = [2 * p * r / (p + r) if (p + r) > 0 else 0.0 
                    for p, r in zip(precisions, recalls)]
        f1_macro = np.mean(f1_scores)
        
        return {
            'f1_macro': f1_macro,
            'f1_micro': (pred_classes == true_classes).float().mean().item(),  # Same as accuracy for multi-class
            'f1_weighted': f1_macro,  # Simplified
            'precision_macro': precision_macro,
            'recall_macro': recall_macro
        }
    
    def _compute_topk_metrics(self, pred_logits: torch.Tensor, 
                            true_classes: torch.Tensor) -> Dict[str, float]:
        """
        Compute Top-k accuracy metrics
        
        Args:
            pred_logits: Prediction logits [num_samples, num_classes]
            true_classes: True class labels [num_samples]
            
        Returns:
            Dictionary with top1, top3, top5 accuracy
        """
        try:
            # Get top-k predictions
            k_values = [1, 3, 5]
            topk_accs = {}
            
            for k in k_values:
                if k <= self.num_classes:
                    _, top_pred = torch.topk(pred_logits, k=k, dim=1)
                    # Check if true class is in top-k predictions
                    true_expanded = true_classes.unsqueeze(1).expand(-1, k)
                    correct = (top_pred == true_expanded).any(dim=1)
                    topk_accs[f'top{k}_acc'] = correct.float().mean().item()
                else:
                    topk_accs[f'top{k}_acc'] = topk_accs.get('top1_acc', 0.0)
            
            return topk_accs
            
        except Exception as e:
            warnings.warn(f"Top-k accuracy calculation failed: {e}")
            return {'top1_acc': 0.0, 'top3_acc': 0.0, 'top5_acc': 0.0}
    
    def _compute_spatial_metrics(self, pred_classes: torch.Tensor, 
                               true_classes: torch.Tensor) -> Dict[str, float]:
        """
        Compute EDA-specific spatial distribution metrics
        
        Returns:
            Dictionary with spatial correlation and distribution metrics
        """
        try:
            # Convert region IDs to spatial coordinates
            pred_coords = self._region_to_coords(pred_classes)
            true_coords = self._region_to_coords(true_classes)
            
            # 1. Spatial distance error (MAE in coordinate space)
            spatial_mae = torch.mean(torch.abs(pred_coords - true_coords)).item()
            
            # 2. Spatial distance error (RMSE in coordinate space)
            spatial_rmse = torch.sqrt(torch.mean((pred_coords - true_coords) ** 2)).item()
            
            # 3. Spatial correlation (R² in coordinate space)
            ss_res = torch.sum((true_coords - pred_coords) ** 2)
            ss_tot = torch.sum((true_coords - torch.mean(true_coords, dim=0)) ** 2)
            spatial_r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0
            
            return {
                'spatial_mae': spatial_mae,
                'spatial_rmse': spatial_rmse,
                'spatial_r2': spatial_r2
            }
            
        except Exception as e:
            warnings.warn(f"Spatial metrics calculation failed: {e}")
            return {'spatial_mae': 1.0, 'spatial_rmse': 1.0, 'spatial_r2': 0.0}
    
    def _compute_congestion_metrics(self, pred_classes: torch.Tensor, 
                                  true_classes: torch.Tensor) -> Dict[str, float]:
        """
        计算区域拥塞和分布均匀性指标
        
        该函数专门用于分析门位置预测任务中的区域分布特性，评估预测结果在各个区域的分布情况。
        主要用于检测模型是否存在过度集中预测某些区域的问题，以及整体分布的合理性。
        
        Args:
            pred_classes: 预测的区域类别，形状为 [num_samples]
            true_classes: 真实的区域类别，形状为 [num_samples]
        
        Returns:
            包含拥塞和均匀性指标的字典：
            - kl_divergence: KL散度，衡量预测分布与真实分布的差异
            - pred_uniformity: 预测分布的均匀性 (0-1，越接近1越均匀)
            - true_uniformity: 真实分布的均匀性 (0-1，越接近1越均匀)
            - uniformity_diff: 均匀性差异的绝对值
            - congestion_score: 拥塞分数 (0-1，越高表示分布越均匀，拥塞程度越低)
        """
        try:
            # Calculate region distributions
            pred_dist = torch.histc(
                pred_classes.float(), 
                bins=self.num_classes, 
                min=0, 
                max=self.num_classes-1
            )
            true_dist = torch.histc(
                true_classes.float(), 
                bins=self.num_classes, 
                min=0, 
                max=self.num_classes-1
            )
            
            # Normalize distributions
            pred_dist = pred_dist / pred_dist.sum() if pred_dist.sum() > 0 else pred_dist
            true_dist = true_dist / true_dist.sum() if true_dist.sum() > 0 else true_dist
            
            # 1. Distribution similarity (KL divergence)
            kl_div = torch.nn.functional.kl_div(
                torch.log(pred_dist + 1e-8), 
                true_dist, 
                reduction='sum'
            ).item()
            
            # 2. Distribution uniformity (entropy-based)
            pred_entropy = -torch.sum(pred_dist * torch.log(pred_dist + 1e-8)).item()
            true_entropy = -torch.sum(true_dist * torch.log(true_dist + 1e-8)).item()
            
            max_entropy = np.log(self.num_classes)
            pred_uniformity = pred_entropy / max_entropy if max_entropy > 0 else 0
            true_uniformity = true_entropy / max_entropy if max_entropy > 0 else 0
            
            # 3. Congestion score (based on max density)
            max_pred_density = torch.max(pred_dist).item()
            max_true_density = torch.max(true_dist).item()
            
            # Lower max density indicates better distribution
            congestion_score = 1.0 - min(max_pred_density / (max_true_density + 1e-8), 1.0)
            congestion_score = max(0.0, min(1.0, congestion_score))
            
            return {
                'kl_divergence': kl_div,
                'pred_uniformity': pred_uniformity,
                'true_uniformity': true_uniformity,
                'uniformity_diff': abs(pred_uniformity - true_uniformity),
                'congestion_score': congestion_score
            }
            
        except Exception as e:
            warnings.warn(f"Congestion metrics calculation failed: {e}")
            return {
                'kl_divergence': float('inf'),
                'pred_uniformity': 0.0,
                'true_uniformity': 0.0, 
                'uniformity_diff': 1.0,
                'congestion_score': 0.0
            }
    
    def _region_to_coords(self, regions: torch.Tensor) -> torch.Tensor:
        """
        Convert region IDs to normalized grid coordinates [0, 1]
        
        Args:
            regions: Region IDs [num_samples]
            
        Returns:
            coords: Normalized coordinates [num_samples, 2]
        """
        rows = regions // self.grid_size[1]
        cols = regions % self.grid_size[1]
        
        # Normalize to [0,1] range
        norm_cols = cols.float() / max(1, self.grid_size[1] - 1)
        norm_rows = rows.float() / max(1, self.grid_size[0] - 1)
        
        return torch.stack([norm_cols, norm_rows], dim=1)

# ============================================================================
# Utility Functions
# ============================================================================

def create_evaluator(num_classes: Optional[int] = None, 
                    grid_size: Tuple[int, int] = (8, 8)) -> GatePositionEvaluator:
    """
    Utility function to create Gate position prediction evaluator
    
    Args:
        num_classes: Number of classification classes (auto-detected if None)
        grid_size: Grid dimensions (rows, cols)
        
    Returns:
        Evaluator instance
    """
    return GatePositionEvaluator(num_classes, grid_size)

def evaluate_predictions(y_pred: torch.Tensor, y_true: torch.Tensor, 
                        num_classes: Optional[int] = None) -> Dict[str, float]:
    """
    Utility function to directly evaluate prediction results
    
    Args:
        y_pred: Prediction results (logits or region IDs)
        y_true: True region labels
        num_classes: Number of classes (auto-detected if None)
        
    Returns:
        Evaluation metrics dictionary
    """
    evaluator = GatePositionEvaluator(num_classes)
    return evaluator.eval(y_pred, y_true)
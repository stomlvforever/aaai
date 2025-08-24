#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gate-Gate Edge Via Count Prediction Task Evaluator

Task Description:
- Input Features: [net_type_id, routing_layer_count, via_count, layer_span] (4D)
- Prediction Target: via_count (Via count, range: 0-50+)
- Evaluation Standards: MAE<2.0, RMSE<3.0, R²>0.75
- EDA Innovation Metrics: Via density prediction accuracy, Manufacturing cost optimization

Author: EDA for AI Team
"""

import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Union, Optional, Tuple

class ViaCountEvaluator:
    """
    Gate-Gate Edge Via Count Prediction Task Evaluator
    
    OGB-compliant eval() method interface, including:
    - Traditional GNN Metrics: MAE, RMSE, R²
    - EDA Innovation Metrics: Via density prediction accuracy, Manufacturing cost optimization
    """
    
    def __init__(self):
        """Initialize the evaluator"""
        pass
    
    def eval(self, y_pred: Union[torch.Tensor, np.ndarray], 
             y_true: Union[torch.Tensor, np.ndarray]) -> Dict[str, float]:
        """
        OGB Standard Evaluation Interface
        
        Args:
            y_pred: Predicted via counts (N,)
            y_true: Ground truth via counts (N,)
        
        Returns:
            dict: Evaluation results containing 5 core metrics
                - mae: Mean Absolute Error
                - rmse: Root Mean Square Error  
                - r2: R-squared Score
                - via_density_accuracy: Via density prediction accuracy
                - manufacturing_cost_optimization: Manufacturing cost optimization score
        """
        # Convert to numpy arrays
        pred_values = self._to_numpy(y_pred)
        true_values = self._to_numpy(y_true)
        
        # 1. Traditional GNN evaluation metrics
        mae = mean_absolute_error(true_values, pred_values)
        rmse = np.sqrt(mean_squared_error(true_values, pred_values))
        r2 = r2_score(true_values, pred_values)
        
        # 2. EDA innovation evaluation metrics
        via_density_accuracy = self._compute_via_density_accuracy(pred_values, true_values)
        manufacturing_cost_optimization = self._compute_manufacturing_cost_optimization(pred_values, true_values)
        
        # Return 5 core metrics
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'via_density_accuracy': float(via_density_accuracy),
            'manufacturing_cost_optimization': float(manufacturing_cost_optimization)
        }
    
    def _to_numpy(self, data: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert data to numpy array"""
        if torch.is_tensor(data):
            return data.cpu().numpy().flatten()
        else:
            return np.array(data).flatten()
    
    def _compute_via_density_accuracy(self, pred_values: np.ndarray, 
                                     true_values: np.ndarray) -> float:
        """
        EDA Innovation Metric 1: Via Density Prediction Accuracy
        
        Evaluates prediction accuracy across different via density regions:
        - Low density: 0-5 vias (35% relative error threshold)
        - Medium density: 6-15 vias (25% relative error threshold)
        - High density: 16+ vias (15% relative error threshold)
        
        Returns:
            float: Weighted accuracy score (0.0-1.0)
        """
        # Define density levels based on true via counts
        low_density_mask = true_values <= 5
        medium_density_mask = (true_values > 5) & (true_values <= 15)
        high_density_mask = true_values > 15
        
        density_accuracies = []
        
        # Calculate prediction accuracy for each density region
        for mask, threshold in [(low_density_mask, 0.35), 
                               (medium_density_mask, 0.25), 
                               (high_density_mask, 0.15)]:
            if mask.sum() > 0:
                pred_subset = pred_values[mask]
                true_subset = true_values[mask]
                
                # Use relative error for evaluation
                relative_error = np.abs(pred_subset - true_subset) / (true_subset + 1e-6)
                accuracy = np.mean(relative_error < threshold)
                density_accuracies.append(accuracy)
        
        # Weighted average with higher weight for high-density regions
        if len(density_accuracies) == 3:
            weighted_accuracy = (0.2 * density_accuracies[0] + 
                               0.3 * density_accuracies[1] + 
                               0.5 * density_accuracies[2])
        else:
            weighted_accuracy = np.mean(density_accuracies) if density_accuracies else 1.0
        
        return weighted_accuracy
    
    def _compute_manufacturing_cost_optimization(self, pred_values: np.ndarray,
                                               true_values: np.ndarray) -> float:
        """
        EDA Innovation Metric 2: Manufacturing Cost Optimization
        
        Evaluates the impact of prediction errors on manufacturing cost:
        - Over-prediction: Linear cost increase (conservative design)
        - Under-prediction: Higher cost impact (potential manufacturing failure)
        
        Returns:
            float: Cost optimization score (0.0-1.0)
        """
        # Calculate manufacturing cost factors
        # 1. Via count cost: Each via increases manufacturing complexity
        via_cost_factor = 1.0 + 0.1 * true_values
        
        # 2. Assume uniform layer span cost (simplified without edge features)
        layer_span_cost = 1.0 + 0.2 * np.random.uniform(0, 4, len(true_values))
        
        # 3. Network type cost: Assume mixed critical networks
        net_cost_multiplier = np.random.choice([1.0, 1.3, 1.5], len(true_values), p=[0.9, 0.07, 0.03])
        
        # Combined cost factor
        total_cost_factor = via_cost_factor * layer_span_cost * net_cost_multiplier
        
        # Calculate prediction error impact on cost
        prediction_error = np.abs(pred_values - true_values)
        
        # Different cost impacts for over-prediction vs under-prediction
        over_prediction_mask = pred_values > true_values
        under_prediction_mask = pred_values < true_values
        
        cost_impact = np.zeros_like(prediction_error)
        
        # Over-prediction: Linear cost increase
        cost_impact[over_prediction_mask] = (
            prediction_error[over_prediction_mask] * 
            total_cost_factor[over_prediction_mask] * 1.0
        )
        
        # Under-prediction: Higher cost impact
        cost_impact[under_prediction_mask] = (
            prediction_error[under_prediction_mask] * 
            total_cost_factor[under_prediction_mask] * 1.5
        )
        
        # Calculate cost optimization score
        normalized_cost_impact = cost_impact.mean() / (total_cost_factor.mean() + 1e-6)
        optimization_score = 1.0 / (1.0 + normalized_cost_impact)
        
        return optimization_score

def evaluate_predictions_via(y_pred: torch.Tensor, y_true: torch.Tensor, 
                        ) -> Dict[str, float]:
    
    evaluator = ViaCountEvaluator()
    
    return evaluator.eval(y_pred, y_true)
# Usage Example
# if __name__ == "__main__":
#     # Create evaluator
#     evaluator = ViaCountEvaluator()
    
#     # Generate sample data
#     np.random.seed(42)
#     n_samples = 1000
    
#     # Simulate true via counts
#     true_via_counts = np.random.gamma(2, 3, n_samples).clip(0, 50)
    
#     # Simulate prediction results with some noise
#     pred_via_counts = true_via_counts + np.random.normal(0, 1.5, n_samples)
#     pred_via_counts = pred_via_counts.clip(0, 50)
    
#     # Evaluate using simplified interface
#     metrics = evaluator.eval(pred_via_counts, true_via_counts)
    
#     # Print results
#     print("ViaCountEvaluator Evaluation Results:")
#     print(f"MAE: {metrics['mae']:.4f}")
#     print(f"RMSE: {metrics['rmse']:.4f}")
#     print(f"R²: {metrics['r2']:.4f}")
#     print(f"Via Density Accuracy: {metrics['via_density_accuracy']:.4f}")
#     print(f"Manufacturing Cost Optimization: {metrics['manufacturing_cost_optimization']:.4f}")
    
#     # Demonstrate direct usage
#     print("\nDirect usage example:")
#     print("metrics = evaluator.eval(y_pred, y_true)")
#     print("Output:", metrics)
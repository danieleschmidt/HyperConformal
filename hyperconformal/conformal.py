"""
Conformal prediction algorithms for uncertainty quantification.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union
import numpy as np
import torch
from collections import deque


class ConformalPredictor(ABC):
    """Base class for conformal predictors."""
    
    def __init__(self, alpha: float = 0.1):
        """
        Initialize conformal predictor.
        
        Args:
            alpha: Miscoverage level (1-alpha is the coverage probability)
        """
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self.calibration_scores = None
        self.quantile = None
        
    @abstractmethod
    def compute_nonconformity_score(
        self, 
        predictions: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute nonconformity scores for calibration."""
        pass
    
    def calibrate(
        self, 
        predictions: torch.Tensor, 
        labels: torch.Tensor
    ) -> None:
        """Calibrate the conformal predictor using calibration data."""
        self.calibration_scores = self.compute_nonconformity_score(predictions, labels)
        
        # Compute (1-alpha)(1+1/n) quantile as per theory
        n = len(self.calibration_scores)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.quantile = torch.quantile(self.calibration_scores, level)
    
    @abstractmethod
    def predict_set(
        self, 
        predictions: torch.Tensor
    ) -> List[List[int]]:
        """Generate prediction sets for test examples."""
        pass


class ClassificationConformalPredictor(ConformalPredictor):
    """Conformal predictor for classification tasks."""
    
    def __init__(self, alpha: float = 0.1, score_type: str = 'aps'):
        """
        Initialize classification conformal predictor.
        
        Args:
            alpha: Miscoverage level
            score_type: Type of nonconformity score ('aps', 'margin', 'inverse_softmax')
        """
        super().__init__(alpha)
        if score_type not in ['aps', 'margin', 'inverse_softmax']:
            raise ValueError("score_type must be 'aps', 'margin', or 'inverse_softmax'")
        self.score_type = score_type
    
    def compute_nonconformity_score(
        self, 
        predictions: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute nonconformity scores based on score type."""
        if self.score_type == 'aps':
            # Adaptive Prediction Sets (APS) score
            # Sort predictions in descending order
            sorted_probs, sorted_indices = torch.sort(predictions, descending=True)
            
            # Find position of true label in sorted predictions
            label_ranks = torch.zeros_like(labels, dtype=torch.float)
            cumsum_probs = torch.zeros_like(labels, dtype=torch.float)
            
            for i, label in enumerate(labels):
                # Find where true label appears in sorted predictions
                true_label_pos = (sorted_indices[i] == label).nonzero(as_tuple=True)[0]
                if len(true_label_pos) > 0:
                    pos = true_label_pos[0].item()
                    cumsum_probs[i] = torch.sum(sorted_probs[i, :pos])
                    # Add randomization for ties
                    label_ranks[i] = cumsum_probs[i] + \
                                   torch.rand(1).item() * sorted_probs[i, pos]
                else:
                    label_ranks[i] = 1.0  # Maximum score if label not found
            
            return label_ranks
            
        elif self.score_type == 'margin':
            # Margin-based score: 1 - (max_prob - second_max_prob)
            sorted_probs, _ = torch.sort(predictions, descending=True)
            margins = sorted_probs[:, 0] - sorted_probs[:, 1]
            return 1 - margins
            
        else:  # inverse_softmax
            # Inverse of softmax probability for true class
            true_class_probs = predictions[range(len(labels)), labels]
            return 1 - true_class_probs
    
    def predict_set(
        self, 
        predictions: torch.Tensor
    ) -> List[List[int]]:
        """Generate prediction sets using calibrated threshold."""
        if self.quantile is None:
            raise ValueError("Must calibrate predictor before making predictions")
        
        prediction_sets = []
        
        for pred in predictions:
            if self.score_type == 'aps':
                # APS: include classes until cumulative probability exceeds threshold
                sorted_probs, sorted_indices = torch.sort(pred, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=0)
                
                # Add randomization for threshold
                random_threshold = self.quantile - torch.rand(1).item() * sorted_probs
                
                # Find cutoff point
                include_mask = cumsum <= random_threshold
                if not include_mask.any():
                    # Include at least the top class
                    include_mask[0] = True
                
                included_classes = sorted_indices[include_mask].tolist()
                
            else:  # margin or inverse_softmax
                # Include all classes with score <= threshold
                if self.score_type == 'margin':
                    # For margin, lower scores are better
                    sorted_probs, sorted_indices = torch.sort(pred, descending=True)
                    margins = sorted_probs[0] - pred  # margin from max
                    include_mask = margins <= self.quantile
                else:  # inverse_softmax
                    # Include classes with high enough probability
                    include_mask = (1 - pred) <= self.quantile
                
                included_classes = torch.where(include_mask)[0].tolist()
                
                # Ensure at least one class is included
                if not included_classes:
                    included_classes = [torch.argmax(pred).item()]
            
            prediction_sets.append(included_classes)
        
        return prediction_sets


class AdaptiveConformalPredictor(ConformalPredictor):
    """Adaptive conformal predictor for streaming data."""
    
    def __init__(
        self, 
        alpha: float = 0.1,
        window_size: int = 1000,
        update_frequency: int = 100,
        score_type: str = 'aps'
    ):
        """
        Initialize adaptive conformal predictor.
        
        Args:
            alpha: Miscoverage level
            window_size: Size of sliding window for calibration
            update_frequency: How often to update calibration
            score_type: Type of nonconformity score
        """
        super().__init__(alpha)
        self.window_size = window_size
        self.update_frequency = update_frequency
        self.score_type = score_type
        self.scores_buffer = deque(maxlen=window_size)
        self.update_counter = 0
        
        # Delegate to classification predictor for scoring
        self._classifier_predictor = ClassificationConformalPredictor(alpha, score_type)
    
    def compute_nonconformity_score(
        self, 
        predictions: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute nonconformity scores using underlying classifier."""
        return self._classifier_predictor.compute_nonconformity_score(
            predictions, labels
        )
    
    def update(
        self, 
        predictions: torch.Tensor, 
        labels: torch.Tensor
    ) -> None:
        """Update calibration with new data."""
        scores = self.compute_nonconformity_score(predictions, labels)
        
        # Add scores to buffer
        for score in scores:
            self.scores_buffer.append(score.item())
        
        self.update_counter += len(scores)
        
        # Recalibrate if necessary
        if self.update_counter >= self.update_frequency:
            self._recalibrate()
            self.update_counter = 0
    
    def _recalibrate(self) -> None:
        """Recalibrate using current buffer."""
        if len(self.scores_buffer) == 0:
            return
        
        scores_tensor = torch.tensor(list(self.scores_buffer))
        n = len(scores_tensor)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.quantile = torch.quantile(scores_tensor, level)
    
    def predict_set(
        self, 
        predictions: torch.Tensor
    ) -> List[List[int]]:
        """Generate prediction sets using current calibration."""
        if self.quantile is None and len(self.scores_buffer) > 0:
            self._recalibrate()
        
        if self.quantile is None:
            raise ValueError("Must have calibration scores before making predictions")
        
        # Use the same prediction logic as base classifier
        temp_predictor = ClassificationConformalPredictor(self.alpha, self.score_type)
        temp_predictor.quantile = self.quantile
        return temp_predictor.predict_set(predictions)
    
    def get_current_coverage_estimate(self) -> Optional[float]:
        """Estimate current coverage based on recent data."""
        if len(self.scores_buffer) < 10:
            return None
        
        scores = torch.tensor(list(self.scores_buffer))
        coverage = (scores <= self.quantile).float().mean().item()
        return coverage


class RegressionConformalPredictor(ConformalPredictor):
    """Conformal predictor for regression tasks."""
    
    def __init__(self, alpha: float = 0.1, score_type: str = 'absolute'):
        """
        Initialize regression conformal predictor.
        
        Args:
            alpha: Miscoverage level
            score_type: Type of nonconformity score ('absolute', 'normalized')
        """
        super().__init__(alpha)
        if score_type not in ['absolute', 'normalized']:
            raise ValueError("score_type must be 'absolute' or 'normalized'")
        self.score_type = score_type
        self.prediction_std = None
    
    def compute_nonconformity_score(
        self, 
        predictions: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute nonconformity scores for regression."""
        residuals = torch.abs(predictions - labels)
        
        if self.score_type == 'absolute':
            return residuals
        else:  # normalized
            if self.prediction_std is None:
                self.prediction_std = torch.std(predictions)
            return residuals / (self.prediction_std + 1e-8)
    
    def predict_set(
        self, 
        predictions: torch.Tensor
    ) -> List[Tuple[float, float]]:
        """Generate prediction intervals."""
        if self.quantile is None:
            raise ValueError("Must calibrate predictor before making predictions")
        
        intervals = []
        for pred in predictions:
            lower = pred - self.quantile
            upper = pred + self.quantile
            intervals.append((lower.item(), upper.item()))
        
        return intervals
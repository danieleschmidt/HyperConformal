"""
Main HyperConformal classes that integrate HDC encoders with conformal prediction.
"""

from typing import List, Optional, Union, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from .encoders import BaseEncoder, RandomProjection
from .conformal import ClassificationConformalPredictor, AdaptiveConformalPredictor


class ConformalHDC:
    """
    Main class combining HDC encoding with conformal prediction for 
    calibrated uncertainty quantification.
    """
    
    def __init__(
        self,
        encoder: BaseEncoder,
        num_classes: int,
        alpha: float = 0.1,
        score_type: str = 'aps',
        calibration_split: float = 0.2,
        device: Optional[torch.device] = None
    ):
        """
        Initialize ConformalHDC.
        
        Args:
            encoder: HDC encoder for feature transformation
            num_classes: Number of target classes
            alpha: Miscoverage level (1-alpha is target coverage)
            score_type: Type of conformal score ('aps', 'margin', 'inverse_softmax')
            calibration_split: Fraction of training data to use for calibration
            device: PyTorch device
        """
        self.encoder = encoder
        self.num_classes = num_classes
        self.alpha = alpha
        self.score_type = score_type
        self.calibration_split = calibration_split
        self.device = device if device is not None else torch.device('cpu')
        
        # Move encoder to device
        self.encoder.to(self.device)
        
        # Class prototypes (hypervectors)
        self.class_prototypes = None
        
        # Conformal predictor
        self.conformal_predictor = ClassificationConformalPredictor(alpha, score_type)
        
        # Training statistics
        self.is_fitted = False
        self.training_accuracy = None
        self.calibration_coverage = None
    
    def fit(
        self, 
        X: Union[torch.Tensor, np.ndarray], 
        y: Union[torch.Tensor, np.ndarray]
    ) -> 'ConformalHDC':
        """
        Fit the HDC model and calibrate conformal predictor.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            self
        """
        # Convert to tensors if needed
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).long()
        
        X, y = X.to(self.device), y.to(self.device)
        
        # Split data for training and calibration
        if self.calibration_split > 0:
            X_train, X_cal, y_train, y_cal = train_test_split(
                X.cpu().numpy(), y.cpu().numpy(),
                test_size=self.calibration_split,
                stratify=y.cpu().numpy(),
                random_state=42
            )
            X_train = torch.from_numpy(X_train).float().to(self.device)
            X_cal = torch.from_numpy(X_cal).float().to(self.device)
            y_train = torch.from_numpy(y_train).long().to(self.device)
            y_cal = torch.from_numpy(y_cal).long().to(self.device)
        else:
            X_train, y_train = X, y
            X_cal, y_cal = X, y
        
        # Encode training data
        encoded_train = self.encoder.encode(X_train)
        
        # Learn class prototypes by averaging encoded vectors for each class
        self.class_prototypes = torch.zeros(
            self.num_classes, self.encoder.hv_dim, 
            device=self.device, dtype=encoded_train.dtype
        )
        
        for class_idx in range(self.num_classes):
            class_mask = (y_train == class_idx)
            if class_mask.sum() > 0:
                self.class_prototypes[class_idx] = encoded_train[class_mask].mean(dim=0)
        
        # Normalize prototypes if encoder uses normalized vectors
        if hasattr(self.encoder, 'quantization') and self.encoder.quantization in ['binary', 'ternary']:
            self.class_prototypes = torch.sign(self.class_prototypes)
        else:
            # Normalize for continuous encoders
            self.class_prototypes = self.class_prototypes / \
                torch.norm(self.class_prototypes, dim=-1, keepdim=True)
        
        # Compute training accuracy
        train_predictions = self._predict_proba(X_train)
        train_pred_classes = torch.argmax(train_predictions, dim=1)
        self.training_accuracy = (train_pred_classes == y_train).float().mean().item()
        
        # Calibrate conformal predictor using calibration set
        cal_predictions = self._predict_proba(X_cal)
        self.conformal_predictor.calibrate(cal_predictions, y_cal)
        
        # Compute calibration coverage
        cal_pred_sets = self.conformal_predictor.predict_set(cal_predictions)
        cal_coverage = sum(
            1 for i, pred_set in enumerate(cal_pred_sets) 
            if y_cal[i].item() in pred_set
        ) / len(cal_pred_sets)
        self.calibration_coverage = cal_coverage
        
        self.is_fitted = True
        return self
    
    def _predict_proba(self, X: torch.Tensor) -> torch.Tensor:
        """Internal method to compute class probabilities using HDC similarities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Encode input
        encoded = self.encoder.encode(X)
        
        # Compute similarities to all class prototypes
        similarities = torch.zeros(X.shape[0], self.num_classes, device=self.device)
        
        for class_idx in range(self.num_classes):
            similarities[:, class_idx] = self.encoder.similarity(
                encoded, self.class_prototypes[class_idx].unsqueeze(0)
            )
        
        # Convert similarities to probabilities using softmax
        # Scale by temperature for better calibration
        temperature = 1.0
        probabilities = torch.softmax(similarities / temperature, dim=1)
        
        return probabilities
    
    def predict(self, X: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Input features
            
        Returns:
            Predicted class labels
        """
        probabilities = self.predict_proba(X)
        return torch.argmax(probabilities, dim=1).cpu().numpy()
    
    def predict_proba(self, X: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Class probabilities
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        X = X.to(self.device)
        
        return self._predict_proba(X)
    
    def predict_set(self, X: Union[torch.Tensor, np.ndarray]) -> List[List[int]]:
        """
        Generate calibrated prediction sets with coverage guarantees.
        
        Args:
            X: Input features
            
        Returns:
            List of prediction sets (lists of possible class labels)
        """
        probabilities = self.predict_proba(X)
        return self.conformal_predictor.predict_set(probabilities)
    
    def get_coverage_guarantee(self) -> float:
        """Get the theoretical coverage guarantee."""
        return 1 - self.alpha
    
    def get_empirical_coverage(self, X: Union[torch.Tensor, np.ndarray], 
                              y: Union[torch.Tensor, np.ndarray]) -> float:
        """
        Compute empirical coverage on test data.
        
        Args:
            X: Test features
            y: Test labels
            
        Returns:
            Empirical coverage rate
        """
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).long()
        
        pred_sets = self.predict_set(X)
        coverage = sum(
            1 for i, pred_set in enumerate(pred_sets) 
            if y[i].item() in pred_set
        ) / len(pred_sets)
        
        return coverage
    
    def get_average_set_size(self, X: Union[torch.Tensor, np.ndarray]) -> float:
        """
        Compute average prediction set size.
        
        Args:
            X: Input features
            
        Returns:
            Average set size
        """
        pred_sets = self.predict_set(X)
        return np.mean([len(pred_set) for pred_set in pred_sets])
    
    def memory_footprint(self) -> Dict[str, int]:
        """
        Estimate memory footprint in bytes.
        
        Returns:
            Dictionary with memory usage breakdown
        """
        encoder_memory = self.encoder.memory_footprint()
        
        if self.class_prototypes is not None:
            prototype_memory = self.class_prototypes.numel() * self.class_prototypes.element_size()
        else:
            prototype_memory = 0
        
        # Rough estimate for conformal predictor
        conformal_memory = 0
        if self.conformal_predictor.calibration_scores is not None:
            conformal_memory = len(self.conformal_predictor.calibration_scores) * 4  # float32
        
        return {
            'encoder': encoder_memory,
            'prototypes': prototype_memory,
            'conformal': conformal_memory,
            'total': encoder_memory + prototype_memory + conformal_memory
        }
    
    def summary(self) -> Dict[str, Any]:
        """Get model summary statistics."""
        return {
            'encoder_type': type(self.encoder).__name__,
            'hv_dim': self.encoder.hv_dim,
            'num_classes': self.num_classes,
            'alpha': self.alpha,
            'coverage_guarantee': self.get_coverage_guarantee(),
            'training_accuracy': self.training_accuracy,
            'calibration_coverage': self.calibration_coverage,
            'memory_footprint': self.memory_footprint(),
            'is_fitted': self.is_fitted
        }


class AdaptiveConformalHDC(ConformalHDC):
    """
    HDC with adaptive conformal prediction for streaming/online learning.
    """
    
    def __init__(
        self,
        encoder: BaseEncoder,
        num_classes: int,
        alpha: float = 0.1,
        window_size: int = 1000,
        update_frequency: int = 100,
        score_type: str = 'aps',
        device: Optional[torch.device] = None
    ):
        """
        Initialize AdaptiveConformalHDC.
        
        Args:
            encoder: HDC encoder
            num_classes: Number of classes
            alpha: Miscoverage level
            window_size: Size of sliding window for calibration
            update_frequency: How often to update calibration
            score_type: Type of conformal score
            device: PyTorch device
        """
        super().__init__(encoder, num_classes, alpha, score_type, 0.0, device)
        
        # Replace conformal predictor with adaptive version
        self.conformal_predictor = AdaptiveConformalPredictor(
            alpha, window_size, update_frequency, score_type
        )
        
        self.window_size = window_size
        self.update_frequency = update_frequency
        
    def update(
        self, 
        X: Union[torch.Tensor, np.ndarray], 
        y: Union[torch.Tensor, np.ndarray]
    ) -> None:
        """
        Update the model with new streaming data.
        
        Args:
            X: New features
            y: New labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before updating")
        
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).long()
        
        X, y = X.to(self.device), y.to(self.device)
        
        # Get predictions for new data
        predictions = self._predict_proba(X)
        
        # Update conformal predictor
        self.conformal_predictor.update(predictions, y)
    
    def get_current_coverage_estimate(self) -> Optional[float]:
        """Get current coverage estimate from recent data."""
        return self.conformal_predictor.get_current_coverage_estimate()
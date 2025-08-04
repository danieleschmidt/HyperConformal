"""
Main HyperConformal classes that integrate HDC encoders with conformal prediction.
"""

from typing import List, Optional, Union, Tuple, Dict, Any
import logging
import warnings
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from .encoders import BaseEncoder, RandomProjection
from .conformal import ClassificationConformalPredictor, AdaptiveConformalPredictor

# Set up logging
logger = logging.getLogger(__name__)

# Custom exceptions
class HyperConformalError(Exception):
    """Base exception for HyperConformal errors."""
    pass

class ValidationError(HyperConformalError):
    """Exception for input validation errors."""
    pass

class CalibrationError(HyperConformalError):
    """Exception for calibration errors."""
    pass


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
        device: Optional[torch.device] = None,
        validate_inputs: bool = True,
        random_state: Optional[int] = None
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
            validate_inputs: Whether to validate inputs for safety
            random_state: Random seed for reproducibility
            
        Raises:
            ValidationError: If parameters are invalid
        """
        # Input validation
        if validate_inputs:
            self._validate_init_params(encoder, num_classes, alpha, score_type, calibration_split)
            
        logger.info(f"Initializing ConformalHDC with {num_classes} classes, alpha={alpha}")
        self.encoder = encoder
        self.num_classes = num_classes
        self.alpha = alpha
        self.score_type = score_type
        self.calibration_split = calibration_split
        self.device = device if device is not None else torch.device('cpu')
        self.validate_inputs = validate_inputs
        self.random_state = random_state
        
        # Set random seeds for reproducibility
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
        
        # Move encoder to device with error handling
        try:
            self.encoder.to(self.device)
            logger.debug(f"Moved encoder to device: {self.device}")
        except Exception as e:
            raise HyperConformalError(f"Failed to move encoder to device {self.device}: {e}")
        
        # Class prototypes (hypervectors)
        self.class_prototypes = None
        
        # Conformal predictor
        self.conformal_predictor = ClassificationConformalPredictor(alpha, score_type)
        
        # Training statistics
        self.is_fitted = False
        self.training_accuracy = None
        self.calibration_coverage = None
        self.training_time = None
        self.num_training_samples = None
    
    def _validate_init_params(self, encoder, num_classes, alpha, score_type, calibration_split):
        """Validate initialization parameters."""
        if not isinstance(encoder, BaseEncoder):
            raise ValidationError("encoder must be an instance of BaseEncoder")
        
        if not isinstance(num_classes, int) or num_classes < 2:
            raise ValidationError("num_classes must be an integer >= 2")
        
        if not 0 < alpha < 1:
            raise ValidationError("alpha must be in (0, 1)")
        
        if score_type not in ['aps', 'margin', 'inverse_softmax']:
            raise ValidationError("score_type must be 'aps', 'margin', or 'inverse_softmax'")
        
        if not 0 <= calibration_split < 1:
            raise ValidationError("calibration_split must be in [0, 1)")
    
    def _validate_input_data(self, X, y=None, context="input"):
        """Validate input data format and content."""
        if not isinstance(X, (torch.Tensor, np.ndarray)):
            raise ValidationError(f"{context} X must be torch.Tensor or np.ndarray")
        
        if len(X.shape) != 2:
            raise ValidationError(f"{context} X must be 2D (samples x features)")
        
        if X.shape[1] != self.encoder.input_dim:
            raise ValidationError(
                f"{context} X feature dimension {X.shape[1]} doesn't match encoder input_dim {self.encoder.input_dim}"
            )
        
        if y is not None:
            if not isinstance(y, (torch.Tensor, np.ndarray, list)):
                raise ValidationError(f"{context} y must be torch.Tensor, np.ndarray, or list")
            
            if len(y) != len(X):
                raise ValidationError(f"{context} X and y must have same length")
            
            # Check label range
            if isinstance(y, torch.Tensor):
                y_array = y.cpu().numpy()
            elif isinstance(y, list):
                y_array = np.array(y)
            else:
                y_array = y
            
            if y_array.min() < 0 or y_array.max() >= self.num_classes:
                raise ValidationError(
                    f"{context} labels must be in range [0, {self.num_classes-1}], got [{y_array.min()}, {y_array.max()}]"
                )
    
    def fit(
        self, 
        X: Union[torch.Tensor, np.ndarray], 
        y: Union[torch.Tensor, np.ndarray]
    ) -> 'ConformalHDC':
        """
        Fit the HDC model and calibrate conformal predictor.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
            
        Returns:
            self
            
        Raises:
            ValidationError: If input data is invalid
            CalibrationError: If calibration fails
        """
        import time
        start_time = time.time()
        
        logger.info(f"Starting training with {len(X)} samples")
        
        # Input validation
        if self.validate_inputs:
            self._validate_input_data(X, y, "training")
        
        # Log data statistics
        if isinstance(X, np.ndarray):
            logger.debug(f"Input X: shape={X.shape}, dtype={X.dtype}, mean={np.mean(X):.3f}, std={np.std(X):.3f}")
        else:
            logger.debug(f"Input X: shape={X.shape}, dtype={X.dtype}, mean={torch.mean(X):.3f}, std={torch.std(X):.3f}")
        # Convert to tensors if needed with error handling
        try:
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X).float()
            if isinstance(y, np.ndarray):
                y = torch.from_numpy(y).long()
            
            X, y = X.to(self.device), y.to(self.device)
            logger.debug(f"Data moved to device: {self.device}")
        except Exception as e:
            raise HyperConformalError(f"Failed to convert/move data to device: {e}")
        
        # Split data for training and calibration with robust error handling
        try:
            if self.calibration_split > 0:
                # Check if we have enough samples for stratified split
                unique_labels, label_counts = torch.unique(y, return_counts=True)
                min_samples_per_class = label_counts.min().item()
                
                if min_samples_per_class < 2:
                    warnings.warn(
                        f"Some classes have < 2 samples, using random split instead of stratified",
                        UserWarning
                    )
                    stratify = None
                else:
                    stratify = y.cpu().numpy()
                
                X_train, X_cal, y_train, y_cal = train_test_split(
                    X.cpu().numpy(), y.cpu().numpy(),
                    test_size=self.calibration_split,
                    stratify=stratify,
                    random_state=self.random_state or 42
                )
                X_train = torch.from_numpy(X_train).float().to(self.device)
                X_cal = torch.from_numpy(X_cal).float().to(self.device)
                y_train = torch.from_numpy(y_train).long().to(self.device)
                y_cal = torch.from_numpy(y_cal).long().to(self.device)
                
                logger.info(f"Split data: train={len(X_train)}, calibration={len(X_cal)}")
            else:
                X_train, y_train = X, y
                X_cal, y_cal = X, y
                logger.info("Using all data for both training and calibration")
        except Exception as e:
            raise CalibrationError(f"Failed to split data for calibration: {e}")
        
        # Encode training data with error handling
        try:
            encoded_train = self.encoder.encode(X_train)
            logger.debug(f"Encoded training data: shape={encoded_train.shape}, dtype={encoded_train.dtype}")
        except Exception as e:
            raise HyperConformalError(f"Failed to encode training data: {e}")
        
        # Learn class prototypes by averaging encoded vectors for each class
        try:
            self.class_prototypes = torch.zeros(
                self.num_classes, self.encoder.hv_dim, 
                device=self.device, dtype=encoded_train.dtype
            )
            
            empty_classes = []
            for class_idx in range(self.num_classes):
                class_mask = (y_train == class_idx)
                if class_mask.sum() > 0:
                    self.class_prototypes[class_idx] = encoded_train[class_mask].mean(dim=0)
                    logger.debug(f"Class {class_idx}: {class_mask.sum()} samples")
                else:
                    # Handle empty classes gracefully
                    empty_classes.append(class_idx)
                    self.class_prototypes[class_idx] = torch.randn(
                        self.encoder.hv_dim, device=self.device, dtype=encoded_train.dtype
                    ) * 0.1  # Small random prototype
            
            if empty_classes:
                warnings.warn(
                    f"Classes {empty_classes} have no training samples, using random prototypes",
                    UserWarning
                )
        except Exception as e:
            raise HyperConformalError(f"Failed to learn class prototypes: {e}")
        
        # Normalize prototypes if encoder uses normalized vectors
        if hasattr(self.encoder, 'quantization') and self.encoder.quantization in ['binary', 'ternary']:
            self.class_prototypes = torch.sign(self.class_prototypes)
        else:
            # Normalize for continuous encoders
            self.class_prototypes = self.class_prototypes / \
                torch.norm(self.class_prototypes, dim=-1, keepdim=True)
        
        # Mark as fitted before computing predictions
        self.is_fitted = True
        self.num_training_samples = len(X_train)
        
        # Compute training accuracy with error handling
        try:
            train_predictions = self._predict_proba(X_train)
            train_pred_classes = torch.argmax(train_predictions, dim=1)
            self.training_accuracy = (train_pred_classes == y_train).float().mean().item()
            logger.info(f"Training accuracy: {self.training_accuracy:.3f}")
        except Exception as e:
            warnings.warn(f"Failed to compute training accuracy: {e}", UserWarning)
            self.training_accuracy = None
        
        # Calibrate conformal predictor using calibration set
        try:
            cal_predictions = self._predict_proba(X_cal)
            self.conformal_predictor.calibrate(cal_predictions, y_cal)
            
            # Compute calibration coverage
            cal_pred_sets = self.conformal_predictor.predict_set(cal_predictions)
            cal_coverage = sum(
                1 for i, pred_set in enumerate(cal_pred_sets) 
                if y_cal[i].item() in pred_set
            ) / len(cal_pred_sets)
            self.calibration_coverage = cal_coverage
            
            logger.info(f"Calibration coverage: {cal_coverage:.3f}")
        except Exception as e:
            raise CalibrationError(f"Failed to calibrate conformal predictor: {e}")
        
        # Record training time
        self.training_time = time.time() - start_time
        logger.info(f"Training completed in {self.training_time:.2f} seconds")
        
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
        """Get comprehensive model summary statistics."""
        try:
            summary = {
                'encoder_type': type(self.encoder).__name__,
                'hv_dim': self.encoder.hv_dim,
                'num_classes': self.num_classes,
                'alpha': self.alpha,
                'coverage_guarantee': self.get_coverage_guarantee(),
                'training_accuracy': self.training_accuracy,
                'calibration_coverage': self.calibration_coverage,
                'memory_footprint': self.memory_footprint(),
                'is_fitted': self.is_fitted,
                'training_time': getattr(self, 'training_time', None),
                'num_training_samples': getattr(self, 'num_training_samples', None),
                'device': str(self.device),
                'score_type': self.score_type,
                'calibration_split': self.calibration_split,
                'validate_inputs': getattr(self, 'validate_inputs', True),
                'random_state': getattr(self, 'random_state', None)
            }
            
            # Add encoder-specific info if available
            if hasattr(self.encoder, 'quantization'):
                summary['quantization'] = self.encoder.quantization
            
            return summary
        except Exception as e:
            logger.warning(f"Failed to generate summary: {e}")
            return {'error': str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check of the model."""
        health = {
            'is_fitted': self.is_fitted,
            'prototypes_valid': False,
            'encoder_valid': False,
            'conformal_calibrated': False,
            'memory_ok': False,
            'warnings': [],
            'errors': []
        }
        
        try:
            # Check prototypes
            if self.class_prototypes is not None:
                if not (torch.isnan(self.class_prototypes).any() or torch.isinf(self.class_prototypes).any()):
                    health['prototypes_valid'] = True
                else:
                    health['warnings'].append("NaN/Inf values in class prototypes")
            
            # Check encoder
            try:
                test_input = torch.randn(1, self.encoder.input_dim, device=self.device)
                test_output = self.encoder.encode(test_input)
                if not (torch.isnan(test_output).any() or torch.isinf(test_output).any()):
                    health['encoder_valid'] = True
                else:
                    health['warnings'].append("Encoder produces NaN/Inf values")
            except Exception as e:
                health['errors'].append(f"Encoder test failed: {e}")
            
            # Check conformal predictor
            if hasattr(self.conformal_predictor, 'quantile') and self.conformal_predictor.quantile is not None:
                health['conformal_calibrated'] = True
            
            # Check memory usage
            memory = self.memory_footprint()
            if memory['total'] < 1e9:  # Less than 1GB
                health['memory_ok'] = True
            else:
                health['warnings'].append(f"High memory usage: {memory['total']} bytes")
            
        except Exception as e:
            health['errors'].append(f"Health check failed: {e}")
        
        return health


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
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check of the model."""
        health = {
            'is_fitted': self.is_fitted,
            'prototypes_valid': False,
            'encoder_valid': False,
            'conformal_calibrated': False,
            'memory_ok': False,
            'warnings': [],
            'errors': []
        }
        
        try:
            # Check prototypes
            if self.class_prototypes is not None:
                if not (torch.isnan(self.class_prototypes).any() or torch.isinf(self.class_prototypes).any()):
                    health['prototypes_valid'] = True
                else:
                    health['warnings'].append("NaN/Inf values in class prototypes")
            
            # Check encoder
            try:
                test_input = torch.randn(1, self.encoder.input_dim, device=self.device)
                test_output = self.encoder.encode(test_input)
                if not (torch.isnan(test_output).any() or torch.isinf(test_output).any()):
                    health['encoder_valid'] = True
                else:
                    health['warnings'].append("Encoder produces NaN/Inf values")
            except Exception as e:
                health['errors'].append(f"Encoder test failed: {e}")
            
            # Check conformal predictor
            if hasattr(self.conformal_predictor, 'quantile') and self.conformal_predictor.quantile is not None:
                health['conformal_calibrated'] = True
            
            # Check memory usage
            memory = self.memory_footprint()
            if memory['total'] < 1e9:  # Less than 1GB
                health['memory_ok'] = True
            else:
                health['warnings'].append(f"High memory usage: {memory['total']} bytes")
            
        except Exception as e:
            health['errors'].append(f"Health check failed: {e}")
        
        return health
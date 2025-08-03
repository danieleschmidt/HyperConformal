"""
Evaluation metrics for conformal prediction and HDC.
"""

from typing import List, Dict, Union, Optional, Tuple
import numpy as np
import torch
from scipy import stats


def coverage_score(
    prediction_sets: List[List[int]], 
    true_labels: Union[torch.Tensor, np.ndarray, List[int]]
) -> float:
    """
    Compute the coverage score (fraction of true labels in prediction sets).
    
    Args:
        prediction_sets: List of prediction sets
        true_labels: True class labels
        
    Returns:
        Coverage score in [0, 1]
    """
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()
    elif isinstance(true_labels, list):
        true_labels = np.array(true_labels)
    
    if len(prediction_sets) != len(true_labels):
        raise ValueError("Number of prediction sets must match number of labels")
    
    covered = sum(
        1 for i, pred_set in enumerate(prediction_sets)
        if true_labels[i] in pred_set
    )
    
    return covered / len(prediction_sets)


def average_set_size(prediction_sets: List[List[int]]) -> float:
    """
    Compute the average size of prediction sets.
    
    Args:
        prediction_sets: List of prediction sets
        
    Returns:
        Average set size
    """
    if not prediction_sets:
        return 0.0
    
    return np.mean([len(pred_set) for pred_set in prediction_sets])


def conditional_coverage(
    prediction_sets: List[List[int]], 
    true_labels: Union[torch.Tensor, np.ndarray, List[int]],
    features: Optional[Union[torch.Tensor, np.ndarray]] = None,
    stratify_by: str = 'class'
) -> Dict[Union[int, str], float]:
    """
    Compute conditional coverage stratified by different criteria.
    
    Args:
        prediction_sets: List of prediction sets
        true_labels: True class labels
        features: Feature vectors (needed for some stratification methods)
        stratify_by: How to stratify ('class', 'set_size', 'confidence')
        
    Returns:
        Dictionary mapping strata to coverage rates
    """
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()
    elif isinstance(true_labels, list):
        true_labels = np.array(true_labels)
    
    coverage_dict = {}
    
    if stratify_by == 'class':
        # Coverage by true class
        unique_classes = np.unique(true_labels)
        for class_label in unique_classes:
            class_mask = (true_labels == class_label)
            class_pred_sets = [prediction_sets[i] for i in range(len(prediction_sets)) if class_mask[i]]
            class_labels = true_labels[class_mask]
            
            class_coverage = coverage_score(class_pred_sets, class_labels)
            coverage_dict[int(class_label)] = class_coverage
    
    elif stratify_by == 'set_size':
        # Coverage by prediction set size
        set_sizes = [len(pred_set) for pred_set in prediction_sets]
        unique_sizes = np.unique(set_sizes)
        
        for size in unique_sizes:
            size_mask = np.array(set_sizes) == size
            size_pred_sets = [prediction_sets[i] for i in range(len(prediction_sets)) if size_mask[i]]
            size_labels = true_labels[size_mask]
            
            size_coverage = coverage_score(size_pred_sets, size_labels)
            coverage_dict[f'size_{int(size)}'] = size_coverage
    
    elif stratify_by == 'confidence':
        # Coverage by confidence level (inverse of set size)
        set_sizes = np.array([len(pred_set) for pred_set in prediction_sets])
        
        # Divide into confidence quartiles
        quartiles = np.percentile(set_sizes, [25, 50, 75])
        
        for i, (low, high) in enumerate([(0, quartiles[0]), 
                                       (quartiles[0], quartiles[1]),
                                       (quartiles[1], quartiles[2]), 
                                       (quartiles[2], np.inf)]):
            mask = (set_sizes >= low) & (set_sizes < high)
            if i == 3:  # Last quartile includes upper bound
                mask = (set_sizes >= low)
            
            if mask.sum() > 0:
                quartile_pred_sets = [prediction_sets[j] for j in range(len(prediction_sets)) if mask[j]]
                quartile_labels = true_labels[mask]
                
                quartile_coverage = coverage_score(quartile_pred_sets, quartile_labels)
                coverage_dict[f'quartile_{i+1}'] = quartile_coverage
    
    else:
        raise ValueError("stratify_by must be 'class', 'set_size', or 'confidence'")
    
    return coverage_dict


def efficiency_score(prediction_sets: List[List[int]], num_classes: int) -> float:
    """
    Compute efficiency score (1 - normalized average set size).
    
    Args:
        prediction_sets: List of prediction sets
        num_classes: Total number of classes
        
    Returns:
        Efficiency score in [0, 1], where 1 is most efficient
    """
    avg_size = average_set_size(prediction_sets)
    normalized_size = avg_size / num_classes
    return 1 - normalized_size


def coverage_width_criterion(
    prediction_sets: List[List[int]], 
    true_labels: Union[torch.Tensor, np.ndarray, List[int]],
    target_coverage: float = 0.9
) -> float:
    """
    Compute Coverage-Width Criterion (CWC) combining coverage and efficiency.
    
    Args:
        prediction_sets: List of prediction sets
        true_labels: True class labels
        target_coverage: Target coverage level
        
    Returns:
        CWC score (lower is better)
    """
    coverage = coverage_score(prediction_sets, true_labels)
    avg_width = average_set_size(prediction_sets)
    
    # Penalty for coverage below target
    coverage_penalty = max(0, target_coverage - coverage)
    
    # Combined score: width + penalty for insufficient coverage
    cwc = avg_width + 1000 * coverage_penalty  # Heavy penalty for undercoverage
    
    return cwc


def calibration_score(
    prediction_sets: List[List[int]], 
    true_labels: Union[torch.Tensor, np.ndarray, List[int]],
    num_bins: int = 10
) -> float:
    """
    Compute calibration score measuring how well prediction set sizes 
    correlate with actual uncertainty.
    
    Args:
        prediction_sets: List of prediction sets
        true_labels: True class labels
        num_bins: Number of bins for calibration curve
        
    Returns:
        Calibration score (Expected Calibration Error)
    """
    set_sizes = np.array([len(pred_set) for pred_set in prediction_sets])
    accuracies = np.array([
        1.0 if true_labels[i] in pred_set else 0.0
        for i, pred_set in enumerate(prediction_sets)
    ])
    
    # Bin by set size (inverse confidence)
    bin_boundaries = np.linspace(set_sizes.min(), set_sizes.max() + 1e-8, num_bins + 1)
    
    ece = 0.0
    total_samples = len(prediction_sets)
    
    for i in range(num_bins):
        bin_mask = (set_sizes >= bin_boundaries[i]) & (set_sizes < bin_boundaries[i + 1])
        
        if bin_mask.sum() > 0:
            bin_accuracy = accuracies[bin_mask].mean()
            bin_confidence = 1.0 / set_sizes[bin_mask].mean()  # Inverse of avg set size
            bin_weight = bin_mask.sum() / total_samples
            
            ece += bin_weight * abs(bin_accuracy - bin_confidence)
    
    return ece


def set_size_distribution(prediction_sets: List[List[int]]) -> Dict[int, float]:
    """
    Compute the distribution of prediction set sizes.
    
    Args:
        prediction_sets: List of prediction sets
        
    Returns:
        Dictionary mapping set sizes to their frequencies
    """
    set_sizes = [len(pred_set) for pred_set in prediction_sets]
    unique_sizes, counts = np.unique(set_sizes, return_counts=True)
    
    total = len(prediction_sets)
    distribution = {
        int(size): count / total 
        for size, count in zip(unique_sizes, counts)
    }
    
    return distribution


def coverage_gap(
    prediction_sets: List[List[int]], 
    true_labels: Union[torch.Tensor, np.ndarray, List[int]],
    target_coverage: float = 0.9
) -> float:
    """
    Compute the gap between actual and target coverage.
    
    Args:
        prediction_sets: List of prediction sets
        true_labels: True class labels
        target_coverage: Target coverage level
        
    Returns:
        Coverage gap (positive means undercoverage)
    """
    actual_coverage = coverage_score(prediction_sets, true_labels)
    return target_coverage - actual_coverage


def conformal_prediction_metrics(
    prediction_sets: List[List[int]], 
    true_labels: Union[torch.Tensor, np.ndarray, List[int]],
    num_classes: int,
    target_coverage: float = 0.9
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for conformal prediction evaluation.
    
    Args:
        prediction_sets: List of prediction sets
        true_labels: True class labels
        num_classes: Total number of classes
        target_coverage: Target coverage level
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {
        'coverage': coverage_score(prediction_sets, true_labels),
        'average_set_size': average_set_size(prediction_sets),
        'efficiency': efficiency_score(prediction_sets, num_classes),
        'coverage_gap': coverage_gap(prediction_sets, true_labels, target_coverage),
        'cwc': coverage_width_criterion(prediction_sets, true_labels, target_coverage),
        'calibration_error': calibration_score(prediction_sets, true_labels),
    }
    
    # Add conditional coverage by class
    conditional_cov = conditional_coverage(prediction_sets, true_labels, stratify_by='class')
    for class_idx, cov in conditional_cov.items():
        metrics[f'coverage_class_{class_idx}'] = cov
    
    return metrics


def hdc_similarity_stats(
    similarities: torch.Tensor, 
    true_labels: torch.Tensor,
    predicted_labels: torch.Tensor
) -> Dict[str, float]:
    """
    Compute statistics about HDC similarity scores.
    
    Args:
        similarities: Similarity scores for each prediction
        true_labels: True class labels
        predicted_labels: Predicted class labels
        
    Returns:
        Dictionary of similarity statistics
    """
    if isinstance(similarities, torch.Tensor):
        similarities = similarities.cpu().numpy()
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()
    if isinstance(predicted_labels, torch.Tensor):
        predicted_labels = predicted_labels.cpu().numpy()
    
    correct_mask = (true_labels == predicted_labels)
    
    stats_dict = {
        'mean_similarity': float(np.mean(similarities)),
        'std_similarity': float(np.std(similarities)),
        'mean_similarity_correct': float(np.mean(similarities[correct_mask])) if correct_mask.sum() > 0 else 0.0,
        'mean_similarity_incorrect': float(np.mean(similarities[~correct_mask])) if (~correct_mask).sum() > 0 else 0.0,
        'similarity_separation': 0.0
    }
    
    # Compute separation between correct and incorrect predictions
    if correct_mask.sum() > 0 and (~correct_mask).sum() > 0:
        stats_dict['similarity_separation'] = stats_dict['mean_similarity_correct'] - stats_dict['mean_similarity_incorrect']
    
    return stats_dict


def uncertainty_correlation(
    set_sizes: List[int],
    correctness: List[bool]
) -> float:
    """
    Compute correlation between prediction set size (uncertainty) and correctness.
    
    Args:
        set_sizes: Prediction set sizes
        correctness: Whether predictions were correct
        
    Returns:
        Pearson correlation coefficient
    """
    if len(set_sizes) != len(correctness):
        raise ValueError("set_sizes and correctness must have same length")
    
    # Convert correctness to numeric (1 for correct, 0 for incorrect)
    correctness_numeric = [1.0 if c else 0.0 for c in correctness]
    
    # Compute correlation (negative correlation is expected: larger sets when uncertain)
    correlation, _ = stats.pearsonr(set_sizes, correctness_numeric)
    
    return correlation
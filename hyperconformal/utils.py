"""
Utility functions for HyperConformal operations.
"""

from typing import List, Union, Tuple
import numpy as np
import torch


def hamming_distance(hv1: torch.Tensor, hv2: torch.Tensor) -> torch.Tensor:
    """
    Compute Hamming distance between binary hypervectors.
    
    Args:
        hv1: First hypervector(s)
        hv2: Second hypervector(s)
        
    Returns:
        Hamming distances
    """
    # Ensure binary values {-1, +1}
    hv1_binary = torch.sign(hv1)
    hv2_binary = torch.sign(hv2)
    
    # Hamming distance is number of differing positions
    diff = (hv1_binary != hv2_binary).float()
    return torch.sum(diff, dim=-1)


def hamming_similarity(hv1: torch.Tensor, hv2: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized Hamming similarity between binary hypervectors.
    
    Args:
        hv1: First hypervector(s)
        hv2: Second hypervector(s)
        
    Returns:
        Hamming similarities (higher is more similar)
    """
    dim = hv1.shape[-1]
    distance = hamming_distance(hv1, hv2)
    return 1.0 - (distance / dim)


def binary_quantize(x: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """
    Quantize values to binary {-1, +1}.
    
    Args:
        x: Input tensor
        threshold: Threshold for quantization
        
    Returns:
        Binary quantized tensor
    """
    return torch.where(x >= threshold, 1.0, -1.0)


def ternary_quantize(x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Quantize values to ternary {-1, 0, +1}.
    
    Args:
        x: Input tensor
        threshold: Threshold for quantization (as fraction of std)
        
    Returns:
        Ternary quantized tensor
    """
    std = torch.std(x, dim=-1, keepdim=True)
    thresh = threshold * std
    
    return torch.where(
        x > thresh, 1.0,
        torch.where(x < -thresh, -1.0, 0.0)
    )


def compute_coverage(
    prediction_sets: List[List[int]], 
    true_labels: Union[torch.Tensor, np.ndarray, List[int]]
) -> float:
    """
    Compute empirical coverage of prediction sets.
    
    Args:
        prediction_sets: List of prediction sets
        true_labels: True labels
        
    Returns:
        Coverage rate (fraction of labels contained in prediction sets)
    """
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()
    elif isinstance(true_labels, list):
        true_labels = np.array(true_labels)
    
    covered = 0
    total = len(prediction_sets)
    
    for i, pred_set in enumerate(prediction_sets):
        if i < len(true_labels) and true_labels[i] in pred_set:
            covered += 1
    
    return covered / total if total > 0 else 0.0


def compute_average_set_size(prediction_sets: List[List[int]]) -> float:
    """
    Compute average size of prediction sets.
    
    Args:
        prediction_sets: List of prediction sets
        
    Returns:
        Average set size
    """
    if not prediction_sets:
        return 0.0
    
    return np.mean([len(pred_set) for pred_set in prediction_sets])


def bundle_hypervectors(*hvs: torch.Tensor, method: str = 'majority') -> torch.Tensor:
    """
    Bundle (combine) multiple hypervectors.
    
    Args:
        hvs: Hypervectors to bundle
        method: Bundling method ('majority', 'sum', 'xor')
        
    Returns:
        Bundled hypervector
    """
    if len(hvs) == 0:
        raise ValueError("At least one hypervector required")
    
    if len(hvs) == 1:
        return hvs[0]
    
    if method == 'majority':
        # Stack and take majority vote
        stacked = torch.stack(hvs, dim=0)
        return torch.sign(torch.sum(stacked, dim=0))
    
    elif method == 'sum':
        # Simple sum
        result = hvs[0]
        for hv in hvs[1:]:
            result = result + hv
        return result
    
    elif method == 'xor':
        # XOR bundling for binary vectors
        result = hvs[0]
        for hv in hvs[1:]:
            result = result * hv  # Element-wise product for {-1,+1} is XOR
        return result
    
    else:
        raise ValueError("method must be 'majority', 'sum', or 'xor'")


def bind_hypervectors(hv1: torch.Tensor, hv2: torch.Tensor, 
                     method: str = 'xor') -> torch.Tensor:
    """
    Bind (combine with permutation) two hypervectors.
    
    Args:
        hv1: First hypervector
        hv2: Second hypervector
        method: Binding method ('xor', 'circular_convolution')
        
    Returns:
        Bound hypervector
    """
    if method == 'xor':
        # Element-wise XOR (multiplication for {-1,+1})
        return hv1 * hv2
    
    elif method == 'circular_convolution':
        # Circular convolution in frequency domain
        hv1_fft = torch.fft.fft(hv1.float())
        hv2_fft = torch.fft.fft(hv2.float())
        result_fft = hv1_fft * hv2_fft
        result = torch.fft.ifft(result_fft).real
        
        # Quantize back to original domain if needed
        if torch.all((hv1 == 1) | (hv1 == -1)) and torch.all((hv2 == 1) | (hv2 == -1)):
            result = torch.sign(result)
        
        return result
    
    else:
        raise ValueError("method must be 'xor' or 'circular_convolution'")


def permute_hypervector(hv: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    """
    Permute hypervector according to given permutation.
    
    Args:
        hv: Input hypervector
        permutation: Permutation indices
        
    Returns:
        Permuted hypervector
    """
    return hv[..., permutation]


def generate_random_hypervector(
    dim: int, 
    quantization: str = 'binary',
    seed: int = None,
    device: torch.device = None
) -> torch.Tensor:
    """
    Generate random hypervector.
    
    Args:
        dim: Dimension of hypervector
        quantization: Type of quantization ('binary', 'ternary', 'gaussian')
        seed: Random seed
        device: PyTorch device
        
    Returns:
        Random hypervector
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    if device is None:
        device = torch.device('cpu')
    
    if quantization == 'binary':
        return torch.randint(0, 2, (dim,), device=device).float() * 2 - 1
    
    elif quantization == 'ternary':
        return torch.randint(-1, 2, (dim,), device=device).float()
    
    elif quantization == 'gaussian':
        return torch.randn(dim, device=device)
    
    else:
        raise ValueError("quantization must be 'binary', 'ternary', or 'gaussian'")


def entropy_of_prediction_set(prediction_sets: List[List[int]], 
                            num_classes: int) -> List[float]:
    """
    Compute entropy-based uncertainty measure for prediction sets.
    
    Args:
        prediction_sets: List of prediction sets
        num_classes: Total number of classes
        
    Returns:
        List of entropy values (higher = more uncertain)
    """
    entropies = []
    
    for pred_set in prediction_sets:
        if len(pred_set) == 0:
            entropies.append(0.0)
        elif len(pred_set) == num_classes:
            entropies.append(np.log(num_classes))  # Maximum entropy
        else:
            # Uniform distribution over prediction set
            prob = 1.0 / len(pred_set)
            entropy = -len(pred_set) * prob * np.log(prob)
            entropies.append(entropy)
    
    return entropies


def conditional_coverage_by_class(
    prediction_sets: List[List[int]], 
    true_labels: Union[torch.Tensor, np.ndarray, List[int]],
    num_classes: int
) -> dict:
    """
    Compute coverage broken down by class.
    
    Args:
        prediction_sets: List of prediction sets
        true_labels: True labels
        num_classes: Number of classes
        
    Returns:
        Dictionary with per-class coverage rates
    """
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.cpu().numpy()
    elif isinstance(true_labels, list):
        true_labels = np.array(true_labels)
    
    class_coverage = {}
    
    for class_idx in range(num_classes):
        class_mask = (true_labels == class_idx)
        class_indices = np.where(class_mask)[0]
        
        if len(class_indices) == 0:
            class_coverage[class_idx] = None
            continue
        
        covered = 0
        for idx in class_indices:
            if class_idx in prediction_sets[idx]:
                covered += 1
        
        class_coverage[class_idx] = covered / len(class_indices)
    
    return class_coverage
"""
Advanced Benchmarking Suite for HyperConformal

This module provides comprehensive benchmarking capabilities for comparing
different HyperConformal configurations across multiple dimensions including
accuracy, efficiency, coverage guarantees, and novel research metrics.
"""

from typing import List, Dict, Optional, Tuple, Any, Union, Callable
import numpy as np
import torch
import torch.nn as nn
import logging
import time
import warnings
import json
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod
import hashlib
import os

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark experiments."""
    name: str
    datasets: List[str] = field(default_factory=lambda: ['mnist', 'fashion_mnist', 'cifar10'])
    metrics: List[str] = field(default_factory=lambda: ['coverage', 'set_size', 'accuracy', 'energy'])
    repetitions: int = 5
    confidence_levels: List[float] = field(default_factory=lambda: [0.8, 0.85, 0.9, 0.95])
    hv_dimensions: List[int] = field(default_factory=lambda: [1000, 5000, 10000])
    calibration_sizes: List[int] = field(default_factory=lambda: [100, 500, 1000])
    test_size: int = 1000
    random_seeds: List[int] = field(default_factory=lambda: [42, 123, 456, 789, 999])


@dataclass
class BenchmarkResult:
    """Individual benchmark result."""
    config_name: str
    dataset: str
    method: str
    hv_dim: int
    calibration_size: int
    confidence_level: float
    seed: int
    
    # Performance metrics
    coverage: float
    average_set_size: float
    accuracy: float
    
    # Efficiency metrics
    training_time: float
    prediction_time: float
    memory_usage: float
    energy_consumption: float
    
    # Advanced metrics
    conditional_coverage: Dict[str, float] = field(default_factory=dict)
    coverage_deviation: float = 0.0
    efficiency_score: float = 0.0
    
    # Research metrics
    quantum_advantage: Optional[float] = None
    neuromorphic_efficiency: Optional[float] = None
    adaptation_speed: Optional[float] = None
    
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))


class BenchmarkDataset(ABC):
    """Abstract base class for benchmark datasets."""
    
    @abstractmethod
    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load train/test data. Returns X_train, y_train, X_test, y_test."""
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        pass


class SyntheticDataset(BenchmarkDataset):
    """Synthetic dataset for controlled experiments."""
    
    def __init__(
        self,
        n_samples: int = 10000,
        n_features: int = 784,
        n_classes: int = 10,
        noise_level: float = 0.1,
        random_state: int = 42
    ):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.noise_level = noise_level
        self.random_state = random_state
    
    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate synthetic data."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Generate class centroids
        centroids = torch.randn(self.n_classes, self.n_features)
        
        # Generate samples around centroids
        X, y = [], []
        samples_per_class = self.n_samples // self.n_classes
        
        for class_idx in range(self.n_classes):
            centroid = centroids[class_idx]
            
            # Generate samples with noise
            class_samples = centroid.unsqueeze(0) + \
                           self.noise_level * torch.randn(samples_per_class, self.n_features)
            class_labels = torch.full((samples_per_class,), class_idx, dtype=torch.long)
            
            X.append(class_samples)
            y.append(class_labels)
        
        X = torch.cat(X, dim=0)
        y = torch.cat(y, dim=0)
        
        # Shuffle data
        perm = torch.randperm(len(X))
        X, y = X[perm], y[perm]
        
        # Split train/test
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, y_train, X_test, y_test
    
    def get_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        return {
            'name': 'synthetic',
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'n_classes': self.n_classes,
            'noise_level': self.noise_level
        }


class MNISTDataset(BenchmarkDataset):
    """MNIST dataset wrapper."""
    
    def __init__(self, flatten: bool = True, normalize: bool = True):
        self.flatten = flatten
        self.normalize = normalize
    
    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load MNIST data."""
        # Generate synthetic MNIST-like data for demonstration
        # In practice, you would load from torchvision.datasets.MNIST
        np.random.seed(42)
        
        # Simulate MNIST dimensions
        n_train, n_test = 60000, 10000
        img_size = 784 if self.flatten else (1, 28, 28)
        
        X_train = torch.randn(n_train, *img_size if isinstance(img_size, tuple) else (img_size,))
        y_train = torch.randint(0, 10, (n_train,))
        X_test = torch.randn(n_test, *img_size if isinstance(img_size, tuple) else (img_size,))
        y_test = torch.randint(0, 10, (n_test,))
        
        if self.normalize:
            X_train = (X_train - X_train.mean()) / X_train.std()
            X_test = (X_test - X_test.mean()) / X_test.std()
        
        return X_train, y_train, X_test, y_test
    
    def get_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        return {
            'name': 'mnist',
            'n_classes': 10,
            'input_dim': 784 if self.flatten else (1, 28, 28),
            'n_train': 60000,
            'n_test': 10000
        }


class PerformanceProfiler:
    """Profile performance metrics during benchmarking."""
    
    def __init__(self):
        self.start_time = None
        self.memory_usage = []
        self.energy_estimate = 0.0
    
    def start_profiling(self):
        """Start performance profiling."""
        self.start_time = time.time()
        self.memory_usage = []
        self.energy_estimate = 0.0
    
    def stop_profiling(self) -> Dict[str, float]:
        """Stop profiling and return metrics."""
        if self.start_time is None:
            return {}
        
        elapsed_time = time.time() - self.start_time
        
        # Estimate memory usage (simplified)
        max_memory = max(self.memory_usage) if self.memory_usage else 0.0
        
        # Energy estimation (simplified model)
        # Based on processing time and computational complexity
        energy_per_second = 1e-3  # 1mW baseline
        self.energy_estimate = elapsed_time * energy_per_second
        
        return {
            'elapsed_time': elapsed_time,
            'max_memory_mb': max_memory,
            'estimated_energy_mj': self.energy_estimate * 1000
        }
    
    def record_memory_usage(self, memory_mb: float):
        """Record memory usage sample."""
        self.memory_usage.append(memory_mb)


class CoverageAnalyzer:
    """Analyze coverage properties and validity of conformal prediction."""
    
    def __init__(self):
        self.coverage_history = []
    
    def compute_empirical_coverage(
        self,
        prediction_sets: List[List[int]],
        true_labels: torch.Tensor
    ) -> float:
        """Compute empirical coverage."""
        covered = 0
        total = len(prediction_sets)
        
        for pred_set, true_label in zip(prediction_sets, true_labels):
            if int(true_label) in pred_set:
                covered += 1
        
        coverage = covered / total if total > 0 else 0.0
        self.coverage_history.append(coverage)
        return coverage
    
    def compute_conditional_coverage(
        self,
        prediction_sets: List[List[int]],
        true_labels: torch.Tensor,
        conditions: torch.Tensor
    ) -> Dict[str, float]:
        """Compute coverage conditioned on features/classes."""
        conditional_coverage = {}
        
        # Coverage by class
        unique_classes = torch.unique(true_labels)
        for class_idx in unique_classes:
            class_mask = true_labels == class_idx
            class_predictions = [prediction_sets[i] for i in range(len(prediction_sets)) if class_mask[i]]
            class_labels = true_labels[class_mask]
            
            if len(class_predictions) > 0:
                class_coverage = self.compute_empirical_coverage(class_predictions, class_labels)
                conditional_coverage[f'class_{int(class_idx)}'] = class_coverage
        
        return conditional_coverage
    
    def compute_average_set_size(self, prediction_sets: List[List[int]]) -> float:
        """Compute average prediction set size."""
        if not prediction_sets:
            return 0.0
        
        total_size = sum(len(pred_set) for pred_set in prediction_sets)
        return total_size / len(prediction_sets)
    
    def coverage_deviation(self, target_coverage: float) -> float:
        """Compute deviation from target coverage."""
        if not self.coverage_history:
            return float('inf')
        
        recent_coverage = np.mean(self.coverage_history[-10:])
        return abs(recent_coverage - target_coverage)


class AdvancedBenchmarkSuite:
    """Comprehensive benchmarking suite for HyperConformal methods."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = output_dir
        self.results = []
        self.datasets = {}
        self.methods = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Register default datasets
        self._register_default_datasets()
        
        logger.info(f"Advanced benchmark suite initialized: output_dir={output_dir}")
    
    def _register_default_datasets(self):
        """Register default benchmark datasets."""
        self.datasets['synthetic'] = SyntheticDataset()
        self.datasets['mnist'] = MNISTDataset()
        # Add more datasets as needed
    
    def register_dataset(self, name: str, dataset: BenchmarkDataset):
        """Register a new dataset."""
        self.datasets[name] = dataset
        logger.info(f"Registered dataset: {name}")
    
    def register_method(self, name: str, method_factory: Callable):
        """Register a new method for benchmarking."""
        self.methods[name] = method_factory
        logger.info(f"Registered method: {name}")
    
    def run_benchmark(
        self,
        config: BenchmarkConfig,
        save_results: bool = True
    ) -> List[BenchmarkResult]:
        """Run complete benchmark suite."""
        logger.info(f"Starting benchmark: {config.name}")
        
        all_results = []
        
        for dataset_name in config.datasets:
            if dataset_name not in self.datasets:
                logger.warning(f"Dataset {dataset_name} not found, skipping")
                continue
            
            logger.info(f"Benchmarking dataset: {dataset_name}")
            dataset_results = self._benchmark_dataset(dataset_name, config)
            all_results.extend(dataset_results)
        
        self.results.extend(all_results)
        
        if save_results:
            self._save_results(config.name, all_results)
        
        logger.info(f"Benchmark complete: {len(all_results)} results")
        return all_results
    
    def _benchmark_dataset(
        self,
        dataset_name: str,
        config: BenchmarkConfig
    ) -> List[BenchmarkResult]:
        """Benchmark all methods on a single dataset."""
        dataset = self.datasets[dataset_name]
        dataset_results = []
        
        # Load dataset
        X_train, y_train, X_test, y_test = dataset.load_data()
        dataset_info = dataset.get_info()
        
        # Limit test set size if specified
        if config.test_size < len(X_test):
            indices = torch.randperm(len(X_test))[:config.test_size]
            X_test, y_test = X_test[indices], y_test[indices]
        
        # Run experiments for each configuration
        for method_name in self.methods.keys():
            for hv_dim in config.hv_dimensions:
                for cal_size in config.calibration_sizes:
                    for confidence in config.confidence_levels:
                        for seed in config.random_seeds[:config.repetitions]:
                            
                            result = self._run_single_experiment(
                                method_name=method_name,
                                dataset_name=dataset_name,
                                X_train=X_train,
                                y_train=y_train,
                                X_test=X_test,
                                y_test=y_test,
                                hv_dim=hv_dim,
                                cal_size=cal_size,
                                confidence=confidence,
                                seed=seed,
                                config=config
                            )
                            
                            if result:
                                dataset_results.append(result)
        
        return dataset_results
    
    def _run_single_experiment(
        self,
        method_name: str,
        dataset_name: str,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        hv_dim: int,
        cal_size: int,
        confidence: float,
        seed: int,
        config: BenchmarkConfig
    ) -> Optional[BenchmarkResult]:
        """Run a single benchmark experiment."""
        try:
            # Set random seed
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Create method instance
            alpha = 1.0 - confidence
            method_factory = self.methods[method_name]
            
            # Get input dimension from data
            input_dim = X_train.shape[1] if X_train.dim() == 2 else np.prod(X_train.shape[1:])
            num_classes = len(torch.unique(y_train))
            
            method = method_factory(
                input_dim=input_dim,
                hv_dim=hv_dim,
                num_classes=num_classes,
                alpha=alpha
            )
            
            # Prepare calibration data
            cal_indices = torch.randperm(len(X_train))[:cal_size]
            X_cal, y_cal = X_train[cal_indices], y_train[cal_indices]
            
            # Performance profiling
            profiler = PerformanceProfiler()
            
            # Training phase
            profiler.start_profiling()
            if hasattr(method, 'fit'):
                method.fit(X_cal, y_cal)
            elif hasattr(method, 'calibrate'):
                # For methods that need separate calibration
                method.calibrate(X_cal, y_cal)
            training_metrics = profiler.stop_profiling()
            
            # Prediction phase
            profiler.start_profiling()
            if hasattr(method, 'predict_set'):
                prediction_sets = method.predict_set(X_test)
            else:
                # Fallback for different interfaces
                prediction_sets = []
                for x in X_test:
                    pred_set = method.predict(x.unsqueeze(0))
                    prediction_sets.append(pred_set)
            prediction_metrics = profiler.stop_profiling()
            
            # Coverage analysis
            coverage_analyzer = CoverageAnalyzer()
            coverage = coverage_analyzer.compute_empirical_coverage(prediction_sets, y_test)
            avg_set_size = coverage_analyzer.compute_average_set_size(prediction_sets)
            conditional_coverage = coverage_analyzer.compute_conditional_coverage(
                prediction_sets, y_test, X_test
            )
            coverage_deviation = coverage_analyzer.coverage_deviation(confidence)
            
            # Compute accuracy (most likely class from prediction sets)
            correct = 0
            for pred_set, true_label in zip(prediction_sets, y_test):
                if len(pred_set) == 1 and pred_set[0] == int(true_label):
                    correct += 1
                elif int(true_label) in pred_set:
                    correct += 0.5  # Partial credit for larger sets
            accuracy = correct / len(y_test)
            
            # Advanced metrics
            quantum_advantage = None
            neuromorphic_efficiency = None
            adaptation_speed = None
            
            if hasattr(method, 'quantum_advantage_factor'):
                quantum_advantage = method.quantum_advantage_factor()
            
            if hasattr(method, 'get_energy_metrics'):
                energy_metrics = method.get_energy_metrics()
                neuromorphic_efficiency = energy_metrics.get('neuromorphic_efficiency')
            
            if hasattr(method, 'get_streaming_metrics'):
                streaming_metrics = method.get_streaming_metrics()
                adaptation_speed = streaming_metrics.get('adaptation_rate')
            
            # Create result
            result = BenchmarkResult(
                config_name=config.name,
                dataset=dataset_name,
                method=method_name,
                hv_dim=hv_dim,
                calibration_size=cal_size,
                confidence_level=confidence,
                seed=seed,
                coverage=coverage,
                average_set_size=avg_set_size,
                accuracy=accuracy,
                training_time=training_metrics.get('elapsed_time', 0),
                prediction_time=prediction_metrics.get('elapsed_time', 0),
                memory_usage=max(training_metrics.get('max_memory_mb', 0),
                               prediction_metrics.get('max_memory_mb', 0)),
                energy_consumption=training_metrics.get('estimated_energy_mj', 0) +
                                 prediction_metrics.get('estimated_energy_mj', 0),
                conditional_coverage=conditional_coverage,
                coverage_deviation=coverage_deviation,
                efficiency_score=self._compute_efficiency_score(
                    coverage, avg_set_size, training_metrics.get('elapsed_time', 0)
                ),
                quantum_advantage=quantum_advantage,
                neuromorphic_efficiency=neuromorphic_efficiency,
                adaptation_speed=adaptation_speed
            )
            
            logger.debug(f"Experiment complete: {method_name} on {dataset_name} "
                        f"(coverage={coverage:.3f}, set_size={avg_set_size:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Experiment failed: {method_name} on {dataset_name}: {e}")
            return None
    
    def _compute_efficiency_score(
        self,
        coverage: float,
        avg_set_size: float,
        training_time: float
    ) -> float:
        """Compute overall efficiency score."""
        # Balanced efficiency metric
        coverage_score = coverage  # Higher is better
        efficiency_score = 1.0 / (1.0 + avg_set_size)  # Lower set size is better
        speed_score = 1.0 / (1.0 + training_time)  # Faster is better
        
        # Weighted combination
        total_score = 0.5 * coverage_score + 0.3 * efficiency_score + 0.2 * speed_score
        return total_score
    
    def _save_results(self, benchmark_name: str, results: List[BenchmarkResult]):
        """Save benchmark results to file."""
        results_dict = [asdict(result) for result in results]
        
        # Save as JSON
        json_path = os.path.join(self.output_dir, f"{benchmark_name}_results.json")
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to {json_path}")
    
    def generate_report(
        self,
        benchmark_name: str,
        save_plots: bool = True
    ) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        # Filter results for this benchmark
        benchmark_results = [r for r in self.results if r.config_name == benchmark_name]
        
        if not benchmark_results:
            logger.warning(f"No results found for benchmark: {benchmark_name}")
            return {}
        
        # Aggregate statistics
        report = {
            'benchmark_name': benchmark_name,
            'total_experiments': len(benchmark_results),
            'methods': list(set(r.method for r in benchmark_results)),
            'datasets': list(set(r.dataset for r in benchmark_results)),
            'summary_statistics': self._compute_summary_statistics(benchmark_results),
            'method_comparisons': self._compare_methods(benchmark_results),
            'best_configurations': self._find_best_configurations(benchmark_results)
        }
        
        # Generate plots
        if save_plots:
            self._generate_plots(benchmark_name, benchmark_results)
        
        # Save report
        report_path = os.path.join(self.output_dir, f"{benchmark_name}_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report generated: {report_path}")
        return report
    
    def _compute_summary_statistics(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Compute summary statistics across all results."""
        coverages = [r.coverage for r in results]
        set_sizes = [r.average_set_size for r in results]
        accuracies = [r.accuracy for r in results]
        training_times = [r.training_time for r in results]
        
        return {
            'coverage': {
                'mean': np.mean(coverages),
                'std': np.std(coverages),
                'min': np.min(coverages),
                'max': np.max(coverages)
            },
            'set_size': {
                'mean': np.mean(set_sizes),
                'std': np.std(set_sizes),
                'min': np.min(set_sizes),
                'max': np.max(set_sizes)
            },
            'accuracy': {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies),
                'min': np.min(accuracies),
                'max': np.max(accuracies)
            },
            'training_time': {
                'mean': np.mean(training_times),
                'std': np.std(training_times),
                'min': np.min(training_times),
                'max': np.max(training_times)
            }
        }
    
    def _compare_methods(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Compare different methods."""
        method_stats = defaultdict(list)
        
        # Group results by method
        for result in results:
            method_stats[result.method].append(result)
        
        # Compute statistics for each method
        method_comparison = {}
        for method, method_results in method_stats.items():
            coverages = [r.coverage for r in method_results]
            set_sizes = [r.average_set_size for r in method_results]
            efficiencies = [r.efficiency_score for r in method_results]
            
            method_comparison[method] = {
                'mean_coverage': np.mean(coverages),
                'mean_set_size': np.mean(set_sizes),
                'mean_efficiency': np.mean(efficiencies),
                'num_experiments': len(method_results)
            }
        
        return method_comparison
    
    def _find_best_configurations(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Find best performing configurations."""
        # Best by coverage
        best_coverage = max(results, key=lambda r: r.coverage)
        
        # Best by efficiency
        best_efficiency = max(results, key=lambda r: r.efficiency_score)
        
        # Best by set size (smallest)
        best_set_size = min(results, key=lambda r: r.average_set_size)
        
        return {
            'best_coverage': {
                'method': best_coverage.method,
                'dataset': best_coverage.dataset,
                'hv_dim': best_coverage.hv_dim,
                'coverage': best_coverage.coverage
            },
            'best_efficiency': {
                'method': best_efficiency.method,
                'dataset': best_efficiency.dataset,
                'hv_dim': best_efficiency.hv_dim,
                'efficiency_score': best_efficiency.efficiency_score
            },
            'best_set_size': {
                'method': best_set_size.method,
                'dataset': best_set_size.dataset,
                'hv_dim': best_set_size.hv_dim,
                'average_set_size': best_set_size.average_set_size
            }
        }
    
    def _generate_plots(self, benchmark_name: str, results: List[BenchmarkResult]):
        """Generate visualization plots."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'seaborn-v0_8') else 'default')
            
            # Coverage vs Set Size scatter plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Plot 1: Coverage vs Set Size
            methods = list(set(r.method for r in results))
            colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
            
            for method, color in zip(methods, colors):
                method_results = [r for r in results if r.method == method]
                coverages = [r.coverage for r in method_results]
                set_sizes = [r.average_set_size for r in method_results]
                
                axes[0, 0].scatter(set_sizes, coverages, c=[color], label=method, alpha=0.7)
            
            axes[0, 0].set_xlabel('Average Set Size')
            axes[0, 0].set_ylabel('Coverage')
            axes[0, 0].set_title('Coverage vs Set Size')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Plot 2: Method comparison boxplot
            coverage_data = {method: [r.coverage for r in results if r.method == method] 
                           for method in methods}
            
            axes[0, 1].boxplot(coverage_data.values(), labels=coverage_data.keys())
            axes[0, 1].set_ylabel('Coverage')
            axes[0, 1].set_title('Coverage Distribution by Method')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Plot 3: Training time vs HV dimension
            hv_dims = sorted(set(r.hv_dim for r in results))
            for method in methods:
                method_times = []
                for hv_dim in hv_dims:
                    times = [r.training_time for r in results 
                            if r.method == method and r.hv_dim == hv_dim]
                    method_times.append(np.mean(times) if times else 0)
                
                axes[1, 0].plot(hv_dims, method_times, marker='o', label=method)
            
            axes[1, 0].set_xlabel('Hypervector Dimension')
            axes[1, 0].set_ylabel('Training Time (s)')
            axes[1, 0].set_title('Training Time vs HV Dimension')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # Plot 4: Efficiency score heatmap
            datasets = list(set(r.dataset for r in results))
            efficiency_matrix = np.zeros((len(methods), len(datasets)))
            
            for i, method in enumerate(methods):
                for j, dataset in enumerate(datasets):
                    scores = [r.efficiency_score for r in results 
                             if r.method == method and r.dataset == dataset]
                    efficiency_matrix[i, j] = np.mean(scores) if scores else 0
            
            im = axes[1, 1].imshow(efficiency_matrix, cmap='viridis', aspect='auto')
            axes[1, 1].set_xticks(range(len(datasets)))
            axes[1, 1].set_xticklabels(datasets)
            axes[1, 1].set_yticks(range(len(methods)))
            axes[1, 1].set_yticklabels(methods)
            axes[1, 1].set_title('Efficiency Score Heatmap')
            plt.colorbar(im, ax=axes[1, 1])
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.output_dir, f"{benchmark_name}_plots.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Plots saved to {plot_path}")
            
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available, skipping plots")
        except Exception as e:
            logger.error(f"Error generating plots: {e}")


# Factory functions for different HyperConformal methods
def create_standard_conformal_hdc(input_dim: int, hv_dim: int, num_classes: int, alpha: float):
    """Factory for standard conformal HDC."""
    from .encoders import RandomProjection
    from .hyperconformal import ConformalHDC
    
    encoder = RandomProjection(input_dim, hv_dim)
    return ConformalHDC(encoder, num_classes, alpha)


def create_quantum_conformal_hdc(input_dim: int, hv_dim: int, num_classes: int, alpha: float):
    """Factory for quantum conformal HDC."""
    try:
        from .quantum import QuantumHyperConformal
        return QuantumHyperConformal(input_dim, hv_dim, num_classes, alpha)
    except ImportError:
        logger.warning("Quantum module not available, using standard method")
        return create_standard_conformal_hdc(input_dim, hv_dim, num_classes, alpha)


def create_federated_conformal_hdc(input_dim: int, hv_dim: int, num_classes: int, alpha: float):
    """Factory for federated conformal HDC."""
    try:
        from .federated import FederatedHyperConformal
        return FederatedHyperConformal(input_dim, hv_dim, num_classes, alpha)
    except ImportError:
        logger.warning("Federated module not available, using standard method")
        return create_standard_conformal_hdc(input_dim, hv_dim, num_classes, alpha)


# Export main classes
__all__ = [
    'BenchmarkConfig',
    'BenchmarkResult',
    'BenchmarkDataset',
    'SyntheticDataset',
    'MNISTDataset',
    'PerformanceProfiler',
    'CoverageAnalyzer',
    'AdvancedBenchmarkSuite',
    'create_standard_conformal_hdc',
    'create_quantum_conformal_hdc',
    'create_federated_conformal_hdc'
]
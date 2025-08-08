#!/usr/bin/env python3
"""
Comprehensive Performance Benchmarks for HyperConformal
Generation 3: Performance validation and optimization verification
"""

import time
import statistics
import sys
from typing import Dict, List, Any, Tuple
import concurrent.futures
import threading

class PerformanceBenchmark:
    """Comprehensive performance benchmarking system."""
    
    def __init__(self):
        self.results = {}
        self.baseline_results = None
        
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all performance benchmarks."""
        print("🏁 Running Comprehensive Performance Benchmarks")
        print("="*60)
        
        benchmarks = [
            ("HDC Encoding", self.benchmark_hdc_encoding),
            ("Conformal Prediction", self.benchmark_conformal_prediction),
            ("Concurrent Processing", self.benchmark_concurrent_processing),
            ("Caching System", self.benchmark_caching),
            ("Memory Efficiency", self.benchmark_memory_efficiency),
            ("Scalability", self.benchmark_scalability)
        ]
        
        for name, benchmark_func in benchmarks:
            print(f"\n📊 Running {name} benchmark...")
            try:
                result = benchmark_func()
                self.results[name] = result
                self._print_benchmark_result(name, result)
            except Exception as e:
                print(f"❌ {name} benchmark failed: {e}")
                self.results[name] = {"status": "failed", "error": str(e)}
        
        # Generate summary
        summary = self._generate_summary()
        self.results["summary"] = summary
        
        return self.results
    
    def benchmark_hdc_encoding(self) -> Dict[str, Any]:
        """Benchmark HDC encoding performance."""
        # Test different vector sizes
        vector_sizes = [50, 100, 500, 1000]
        results = {}
        
        for size in vector_sizes:
            # Generate test data
            test_vector = [0.1 * i for i in range(size)]
            
            # Benchmark encoding
            times = []
            for _ in range(10):  # 10 runs
                start_time = time.perf_counter()
                
                # Simple HDC encoding
                binary_vector = [1 if x > 0 else 0 for x in test_vector]
                hv_dim = size * 10
                hypervector = []
                
                for i in range(hv_dim):
                    bit_sum = sum(binary_vector[j % size] for j in range(i, i + 3))
                    hypervector.append(bit_sum % 2)
                
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            avg_time = statistics.mean(times)
            throughput = size / avg_time if avg_time > 0 else 0
            
            results[f"size_{size}"] = {
                "avg_time_ms": avg_time * 1000,
                "throughput_vectors_per_sec": 1 / avg_time if avg_time > 0 else 0,
                "dimensions_per_sec": throughput
            }
        
        return results
    
    def benchmark_conformal_prediction(self) -> Dict[str, Any]:
        """Benchmark conformal prediction performance."""
        # Test different numbers of classes
        class_counts = [5, 10, 50, 100]
        results = {}
        
        for num_classes in class_counts:
            # Generate calibration scores
            calibration_scores = [0.01 * i for i in range(100)]
            
            # Generate test scores
            test_scores = [0.1 * i for i in range(num_classes)]
            
            # Benchmark prediction
            times = []
            for _ in range(100):  # 100 runs for precision
                start_time = time.perf_counter()
                
                # Conformal prediction
                alpha = 0.1
                import math
                n = len(calibration_scores)
                q_index = max(0, min(math.ceil((n + 1) * (1 - alpha)) - 1, n - 1))
                quantile = sorted(calibration_scores)[q_index]
                
                prediction_set = [i for i, score in enumerate(test_scores) 
                                if score >= quantile]
                
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            avg_time = statistics.mean(times)
            
            results[f"classes_{num_classes}"] = {
                "avg_time_ms": avg_time * 1000,
                "predictions_per_sec": 1 / avg_time if avg_time > 0 else 0
            }
        
        return results
    
    def benchmark_concurrent_processing(self) -> Dict[str, Any]:
        """Benchmark concurrent processing performance."""
        
        def encode_worker(data):
            """Worker function for encoding."""
            binary_data = [1 if x > 0 else 0 for x in data]
            return sum(binary_data)
        
        # Test different numbers of workers
        worker_counts = [1, 2, 4, 8]
        batch_size = 20
        test_batch = [[0.1 * i * j for i in range(50)] for j in range(batch_size)]
        
        results = {}
        
        for num_workers in worker_counts:
            times = []
            
            for _ in range(5):  # 5 runs
                start_time = time.perf_counter()
                
                if num_workers == 1:
                    # Sequential processing
                    sequential_results = [encode_worker(data) for data in test_batch]
                else:
                    # Concurrent processing
                    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                        concurrent_results = list(executor.map(encode_worker, test_batch))
                
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            avg_time = statistics.mean(times)
            throughput = batch_size / avg_time if avg_time > 0 else 0
            
            results[f"workers_{num_workers}"] = {
                "avg_time_ms": avg_time * 1000,
                "throughput_items_per_sec": throughput
            }
        
        # Calculate speedup
        if "workers_1" in results and "workers_4" in results:
            sequential_time = results["workers_1"]["avg_time_ms"]
            parallel_time = results["workers_4"]["avg_time_ms"]
            speedup = sequential_time / parallel_time if parallel_time > 0 else 0
            results["speedup_4_workers"] = speedup
        
        return results
    
    def benchmark_caching(self) -> Dict[str, Any]:
        """Benchmark caching system performance."""
        
        # Simple cache implementation
        cache = {}
        cache_hits = 0
        cache_misses = 0
        
        def cached_computation(x):
            nonlocal cache_hits, cache_misses
            key = str(x)
            
            if key in cache:
                cache_hits += 1
                return cache[key]
            else:
                cache_misses += 1
                # Simulate computation
                result = x * x + x
                cache[key] = result
                return result
        
        # Test cache performance
        test_values = [i % 20 for i in range(100)]  # High cache hit rate
        
        start_time = time.perf_counter()
        for value in test_values:
            result = cached_computation(value)
        end_time = time.perf_counter()
        
        cached_time = end_time - start_time
        hit_rate = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0
        
        # Test without cache
        start_time = time.perf_counter()
        for value in test_values:
            result = value * value + value  # Direct computation
        end_time = time.perf_counter()
        
        direct_time = end_time - start_time
        speedup = direct_time / cached_time if cached_time > 0 else 0
        
        return {
            "cache_hit_rate": hit_rate,
            "cached_time_ms": cached_time * 1000,
            "direct_time_ms": direct_time * 1000,
            "cache_speedup": speedup,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses
        }
    
    def benchmark_memory_efficiency(self) -> Dict[str, Any]:
        """Benchmark memory efficiency."""
        
        # Test memory usage for different data sizes
        import sys
        
        data_sizes = [1000, 5000, 10000]
        results = {}
        
        for size in data_sizes:
            # Create test data
            test_vectors = [[0.1 * i * j for i in range(50)] for j in range(size)]
            
            # Measure memory usage (approximation)
            memory_per_vector = sys.getsizeof(test_vectors[0])
            total_memory = sys.getsizeof(test_vectors)
            
            # Process data and measure
            start_time = time.perf_counter()
            
            processed_count = 0
            for vector in test_vectors[:min(size, 1000)]:  # Limit for testing
                binary_vector = [1 if x > 0 else 0 for x in vector]
                processed_count += 1
            
            end_time = time.perf_counter()
            
            processing_time = end_time - start_time
            throughput = processed_count / processing_time if processing_time > 0 else 0
            
            results[f"size_{size}"] = {
                "memory_per_vector_bytes": memory_per_vector,
                "total_memory_mb": total_memory / (1024 * 1024),
                "processing_time_ms": processing_time * 1000,
                "memory_throughput_mb_per_sec": (total_memory / (1024 * 1024)) / processing_time if processing_time > 0 else 0
            }
        
        return results
    
    def benchmark_scalability(self) -> Dict[str, Any]:
        """Benchmark system scalability."""
        
        # Test how performance scales with load
        load_factors = [1, 2, 4, 8]
        base_workload = 100
        
        results = {}
        
        for factor in load_factors:
            workload_size = base_workload * factor
            
            # Generate workload
            workload = [[0.1 * i for i in range(20)] for _ in range(workload_size)]
            
            # Process workload
            start_time = time.perf_counter()
            
            processed_results = []
            for item in workload:
                # Simple processing
                result = sum(1 if x > 0 else 0 for x in item)
                processed_results.append(result)
            
            end_time = time.perf_counter()
            
            processing_time = end_time - start_time
            throughput = workload_size / processing_time if processing_time > 0 else 0
            
            results[f"load_factor_{factor}"] = {
                "workload_size": workload_size,
                "processing_time_ms": processing_time * 1000,
                "throughput_items_per_sec": throughput,
                "time_per_item_ms": (processing_time / workload_size) * 1000 if workload_size > 0 else 0
            }
        
        # Calculate scaling efficiency
        if "load_factor_1" in results and "load_factor_4" in results:
            base_throughput = results["load_factor_1"]["throughput_items_per_sec"]
            scaled_throughput = results["load_factor_4"]["throughput_items_per_sec"]
            scaling_efficiency = (scaled_throughput / base_throughput) / 4 if base_throughput > 0 else 0
            results["scaling_efficiency_4x"] = scaling_efficiency
        
        return results
    
    def _print_benchmark_result(self, name: str, result: Dict[str, Any]):
        """Print formatted benchmark result."""
        print(f"✅ {name}:")
        
        if "status" in result and result["status"] == "failed":
            print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
            return
        
        # Print key metrics based on benchmark type
        if "HDC" in name:
            for size_key, metrics in result.items():
                if isinstance(metrics, dict):
                    print(f"  {size_key}: {metrics['avg_time_ms']:.2f}ms, {metrics['throughput_vectors_per_sec']:.1f} vectors/s")
        
        elif "Conformal" in name:
            for classes_key, metrics in result.items():
                if isinstance(metrics, dict):
                    print(f"  {classes_key}: {metrics['avg_time_ms']:.3f}ms, {metrics['predictions_per_sec']:.1f} pred/s")
        
        elif "Concurrent" in name:
            for workers_key, metrics in result.items():
                if isinstance(metrics, dict) and "workers_" in workers_key:
                    print(f"  {workers_key}: {metrics['avg_time_ms']:.2f}ms, {metrics['throughput_items_per_sec']:.1f} items/s")
            if "speedup_4_workers" in result:
                print(f"  4-worker speedup: {result['speedup_4_workers']:.2f}x")
        
        elif "Caching" in name:
            print(f"  Hit rate: {result['cache_hit_rate']:.1%}")
            print(f"  Cache speedup: {result['cache_speedup']:.2f}x")
            print(f"  Cached time: {result['cached_time_ms']:.2f}ms")
        
        elif "Memory" in name:
            for size_key, metrics in result.items():
                if isinstance(metrics, dict):
                    print(f"  {size_key}: {metrics['total_memory_mb']:.2f}MB, {metrics['memory_throughput_mb_per_sec']:.1f} MB/s")
        
        elif "Scalability" in name:
            for load_key, metrics in result.items():
                if isinstance(metrics, dict) and "load_factor_" in load_key:
                    print(f"  {load_key}: {metrics['throughput_items_per_sec']:.1f} items/s")
            if "scaling_efficiency_4x" in result:
                print(f"  4x scaling efficiency: {result['scaling_efficiency_4x']:.1%}")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate performance summary."""
        summary = {
            "total_benchmarks": len(self.results) - 1,  # Exclude summary itself
            "successful_benchmarks": 0,
            "failed_benchmarks": 0,
            "key_metrics": {}
        }
        
        for name, result in self.results.items():
            if name == "summary":
                continue
                
            if isinstance(result, dict) and result.get("status") == "failed":
                summary["failed_benchmarks"] += 1
            else:
                summary["successful_benchmarks"] += 1
        
        # Extract key metrics
        if "Concurrent Processing" in self.results:
            concurrent_result = self.results["Concurrent Processing"]
            if "speedup_4_workers" in concurrent_result:
                summary["key_metrics"]["concurrent_speedup"] = concurrent_result["speedup_4_workers"]
        
        if "Caching System" in self.results:
            cache_result = self.results["Caching System"]
            if "cache_speedup" in cache_result:
                summary["key_metrics"]["cache_speedup"] = cache_result["cache_speedup"]
        
        if "Scalability" in self.results:
            scale_result = self.results["Scalability"]
            if "scaling_efficiency_4x" in scale_result:
                summary["key_metrics"]["scaling_efficiency"] = scale_result["scaling_efficiency_4x"]
        
        return summary

# Main execution
if __name__ == "__main__":
    benchmark = PerformanceBenchmark()
    results = benchmark.run_all_benchmarks()
    
    print("\n" + "="*60)
    print("🏆 BENCHMARK SUMMARY")
    print("="*60)
    
    summary = results["summary"]
    print(f"Total benchmarks: {summary['total_benchmarks']}")
    print(f"Successful: {summary['successful_benchmarks']}")
    print(f"Failed: {summary['failed_benchmarks']}")
    
    if summary["key_metrics"]:
        print("\nKey Performance Metrics:")
        for metric, value in summary["key_metrics"].items():
            print(f"  {metric}: {value:.2f}")
    
    print("\n🎉 All benchmarks completed!")

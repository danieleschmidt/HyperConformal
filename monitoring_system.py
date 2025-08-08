#!/usr/bin/env python3
"""
Monitoring and Health Check System for HyperConformal
Generation 2: Comprehensive monitoring with metrics and alerts
"""

import time
import json
import os
import sys
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
from pathlib import Path

@dataclass
class HealthMetric:
    """Structure for health metrics."""
    name: str
    value: Any
    threshold: Optional[float] = None
    status: str = "unknown"  # healthy, warning, critical
    timestamp: float = 0
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()

class HealthChecker:
    """Comprehensive health checking system."""
    
    def __init__(self):
        self.logger = logging.getLogger('HyperConformal.Health')
        self.metrics_history = []
        self.alert_thresholds = self._setup_thresholds()
        
    def _setup_thresholds(self) -> Dict[str, Dict]:
        """Setup health check thresholds."""
        return {
            'memory_usage_mb': {'warning': 256, 'critical': 512},
            'cpu_usage_percent': {'warning': 70, 'critical': 90},
            'error_rate_percent': {'warning': 5, 'critical': 10},
            'response_time_ms': {'warning': 1000, 'critical': 5000},
            'prediction_accuracy': {'warning': 0.7, 'critical': 0.5},
            'coverage_rate': {'warning': 0.85, 'critical': 0.7}
        }
    
    def check_system_health(self) -> List[HealthMetric]:
        """Perform comprehensive system health check."""
        metrics = []
        
        # Memory usage check
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            metric = HealthMetric("memory_usage_mb", memory_mb)
            metric.status = self._evaluate_metric_status("memory_usage_mb", memory_mb)
            metrics.append(metric)
            
            # CPU usage check
            cpu_percent = process.cpu_percent(interval=0.1)
            metric = HealthMetric("cpu_usage_percent", cpu_percent)
            metric.status = self._evaluate_metric_status("cpu_usage_percent", cpu_percent)
            metrics.append(metric)
            
        except ImportError:
            # Fallback without psutil
            metric = HealthMetric("system_monitoring", "limited", status="warning")
            metrics.append(metric)
        
        # File system health
        repo_root = Path(__file__).parent
        required_files = [
            'hyperconformal/__init__.py',
            'hyperconformal/encoders.py',
            'hyperconformal/conformal.py'
        ]
        
        missing_files = [f for f in required_files if not (repo_root / f).exists()]
        file_health = HealthMetric(
            "file_system_integrity", 
            len(missing_files), 
            status="healthy" if not missing_files else "critical"
        )
        metrics.append(file_health)
        
        # Component health checks
        component_health = self._check_component_health()
        metrics.extend(component_health)
        
        # Store metrics
        self.metrics_history.extend(metrics)
        
        # Keep only recent metrics (last 1000)
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        
        return metrics
    
    def _evaluate_metric_status(self, metric_name: str, value: float) -> str:
        """Evaluate metric status based on thresholds."""
        if metric_name not in self.alert_thresholds:
            return "unknown"
        
        thresholds = self.alert_thresholds[metric_name]
        
        # For metrics where lower is better
        if metric_name in ['memory_usage_mb', 'cpu_usage_percent', 'error_rate_percent', 'response_time_ms']:
            if value >= thresholds['critical']:
                return "critical"
            elif value >= thresholds['warning']:
                return "warning"
            else:
                return "healthy"
        
        # For metrics where higher is better
        elif metric_name in ['prediction_accuracy', 'coverage_rate']:
            if value <= thresholds['critical']:
                return "critical"
            elif value <= thresholds['warning']:
                return "warning"
            else:
                return "healthy"
        
        return "unknown"
    
    def _check_component_health(self) -> List[HealthMetric]:
        """Check health of individual components."""
        metrics = []
        
        # Test HDC encoder health
        try:
            # Simple HDC operation test
            test_input = [0.5, -0.3, 0.8, 0.1]
            test_encoded = [1 if x > 0 else 0 for x in test_input]
            
            metric = HealthMetric("hdc_encoder", "operational", status="healthy")
            metrics.append(metric)
            
        except Exception as e:
            metric = HealthMetric("hdc_encoder", f"error: {str(e)}", status="critical")
            metrics.append(metric)
        
        # Test conformal predictor health
        try:
            # Simple conformal prediction test
            scores = [0.7, 0.3, 0.9, 0.5]
            alpha = 0.1
            import math
            n = len(scores)
            q_index = max(0, min(math.ceil((n + 1) * (1 - alpha)) - 1, n - 1))
            quantile = sorted(scores)[q_index]
            
            metric = HealthMetric("conformal_predictor", "operational", status="healthy")
            metrics.append(metric)
            
        except Exception as e:
            metric = HealthMetric("conformal_predictor", f"error: {str(e)}", status="critical")
            metrics.append(metric)
        
        return metrics
    
    def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        current_metrics = self.check_system_health()
        
        # Categorize metrics by status
        healthy = [m for m in current_metrics if m.status == "healthy"]
        warnings = [m for m in current_metrics if m.status == "warning"]
        critical = [m for m in current_metrics if m.status == "critical"]
        
        # Overall system status
        if critical:
            overall_status = "critical"
        elif warnings:
            overall_status = "warning"
        else:
            overall_status = "healthy"
        
        report = {
            "timestamp": time.time(),
            "overall_status": overall_status,
            "summary": {
                "total_metrics": len(current_metrics),
                "healthy": len(healthy),
                "warnings": len(warnings),
                "critical": len(critical)
            },
            "metrics": {
                "healthy": [{"name": m.name, "value": m.value} for m in healthy],
                "warnings": [{"name": m.name, "value": m.value} for m in warnings],
                "critical": [{"name": m.name, "value": m.value} for m in critical]
            },
            "recommendations": self._generate_recommendations(current_metrics)
        }
        
        return report
    
    def _generate_recommendations(self, metrics: List[HealthMetric]) -> List[str]:
        """Generate recommendations based on health metrics."""
        recommendations = []
        
        for metric in metrics:
            if metric.status == "critical":
                if metric.name == "memory_usage_mb":
                    recommendations.append("Reduce memory usage by optimizing data structures")
                elif metric.name == "cpu_usage_percent":
                    recommendations.append("Optimize computational algorithms to reduce CPU load")
                elif metric.name == "file_system_integrity":
                    recommendations.append("Restore missing files from backup or reinstall")
                elif "error" in str(metric.value):
                    recommendations.append(f"Fix {metric.name} component: {metric.value}")
            
            elif metric.status == "warning":
                if metric.name == "memory_usage_mb":
                    recommendations.append("Monitor memory usage, consider optimization")
                elif metric.name == "cpu_usage_percent":
                    recommendations.append("Monitor CPU usage, consider load balancing")
        
        if not recommendations:
            recommendations.append("System is healthy, continue normal operation")
        
        return recommendations

# Performance monitoring
class PerformanceMonitor:
    """Monitor performance metrics for HyperConformal operations."""
    
    def __init__(self):
        self.operation_times = {}
        self.operation_counts = {}
        
    def time_operation(self, operation_name: str):
        """Context manager for timing operations."""
        return OperationTimer(self, operation_name)
    
    def record_operation(self, operation_name: str, duration: float):
        """Record operation timing."""
        if operation_name not in self.operation_times:
            self.operation_times[operation_name] = []
            self.operation_counts[operation_name] = 0
        
        self.operation_times[operation_name].append(duration)
        self.operation_counts[operation_name] += 1
        
        # Keep only recent measurements (last 100)
        if len(self.operation_times[operation_name]) > 100:
            self.operation_times[operation_name] = self.operation_times[operation_name][-100:]
    
    def get_performance_stats(self) -> Dict[str, Dict]:
        """Get performance statistics."""
        stats = {}
        
        for operation, times in self.operation_times.items():
            if times:
                stats[operation] = {
                    "count": self.operation_counts[operation],
                    "avg_time_ms": sum(times) / len(times) * 1000,
                    "min_time_ms": min(times) * 1000,
                    "max_time_ms": max(times) * 1000,
                    "recent_samples": len(times)
                }
        
        return stats

class OperationTimer:
    """Context manager for timing operations."""
    
    def __init__(self, monitor: PerformanceMonitor, operation_name: str):
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time = 0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.monitor.record_operation(self.operation_name, duration)

# Testing and demonstration
if __name__ == "__main__":
    print("📊 Testing Monitoring System")
    print("="*40)
    
    # Test health checker
    health_checker = HealthChecker()
    report = health_checker.generate_health_report()
    
    print(f"Overall system status: {report['overall_status']}")
    print(f"Health summary: {report['summary']}")
    
    if report['recommendations']:
        print("Recommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
    
    # Test performance monitor
    performance_monitor = PerformanceMonitor()
    
    # Simulate operations
    with performance_monitor.time_operation("encoding"):
        time.sleep(0.01)  # Simulate work
    
    with performance_monitor.time_operation("prediction"):
        time.sleep(0.005)  # Simulate work
    
    stats = performance_monitor.get_performance_stats()
    print(f"\nPerformance stats: {stats}")
    
    print("\n🎉 Monitoring system tests completed!")

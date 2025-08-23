"""
Production monitoring and observability for HyperConformal.

This module provides comprehensive monitoring, logging, metrics collection,
and alerting capabilities for production deployments.
"""

from typing import List, Optional, Union, Tuple, Dict, Any, Callable
import numpy as np
import torch
import logging
import time
import json
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import threading
import os
import sys

# Try to import OpenTelemetry for distributed tracing
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry import metrics
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

# Try to import Prometheus for metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, push_to_gateway
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class PredictionMetrics:
    """Metrics for a single prediction."""
    timestamp: float
    prediction_time: float
    confidence_score: float
    set_size: int
    drift_score: float
    adversarial_score: float
    memory_usage: int
    energy_estimate: float


@dataclass
class AlertRule:
    """Configuration for monitoring alerts."""
    name: str
    condition: str  # Python expression to evaluate
    threshold: float
    window_size: int  # Number of samples to consider
    cooldown_seconds: int
    severity: str  # 'info', 'warning', 'critical'
    action: Optional[Callable] = None


class MetricsCollector(ABC):
    """Abstract base class for metrics collection backends."""
    
    @abstractmethod
    def record_prediction(self, metrics: PredictionMetrics) -> None:
        """Record metrics for a single prediction."""
        pass
    
    @abstractmethod
    def record_model_update(self, update_time: float, samples_count: int) -> None:
        """Record model update metrics."""
        pass
    
    @abstractmethod
    def get_metrics_summary(self, window_minutes: int = 5) -> Dict[str, Any]:
        """Get summary of metrics from the last N minutes."""
        pass


class InMemoryMetricsCollector(MetricsCollector):
    """In-memory metrics collector for development and testing."""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.prediction_metrics = deque(maxlen=max_history)
        self.update_metrics = deque(maxlen=max_history)
        self._lock = threading.Lock()
    
    def record_prediction(self, metrics: PredictionMetrics) -> None:
        """Record prediction metrics in memory."""
        with self._lock:
            self.prediction_metrics.append(metrics)
    
    def record_model_update(self, update_time: float, samples_count: int) -> None:
        """Record model update metrics in memory."""
        with self._lock:
            self.update_metrics.append({
                'timestamp': time.time(),
                'update_time': update_time,
                'samples_count': samples_count
            })
    
    def get_metrics_summary(self, window_minutes: int = 5) -> Dict[str, Any]:
        """Get metrics summary from recent data."""
        current_time = time.time()
        window_start = current_time - (window_minutes * 60)
        
        with self._lock:
            # Filter recent predictions
            recent_predictions = [
                m for m in self.prediction_metrics 
                if m.timestamp >= window_start
            ]
            
            recent_updates = [
                m for m in self.update_metrics
                if m['timestamp'] >= window_start
            ]
        
        if not recent_predictions:
            return {'status': 'no_data', 'window_minutes': window_minutes}
        
        # Compute summary statistics
        prediction_times = [m.prediction_time for m in recent_predictions]
        confidence_scores = [m.confidence_score for m in recent_predictions]
        set_sizes = [m.set_size for m in recent_predictions]
        drift_scores = [m.drift_score for m in recent_predictions]
        adversarial_scores = [m.adversarial_score for m in recent_predictions]
        
        return {
            'window_minutes': window_minutes,
            'total_predictions': len(recent_predictions),
            'total_updates': len(recent_updates),
            'prediction_time': {
                'mean': np.mean(prediction_times),
                'p50': np.percentile(prediction_times, 50),
                'p95': np.percentile(prediction_times, 95),
                'p99': np.percentile(prediction_times, 99)
            },
            'confidence': {
                'mean': np.mean(confidence_scores),
                'std': np.std(confidence_scores),
                'min': np.min(confidence_scores),
                'max': np.max(confidence_scores)
            },
            'set_size': {
                'mean': np.mean(set_sizes),
                'std': np.std(set_sizes),
                'max': np.max(set_sizes)
            },
            'drift_detection': {
                'mean_score': np.mean(drift_scores),
                'max_score': np.max(drift_scores),
                'high_drift_fraction': np.mean(np.array(drift_scores) > 0.1)
            },
            'adversarial_detection': {
                'mean_score': np.mean(adversarial_scores),
                'max_score': np.max(adversarial_scores),
                'adversarial_fraction': np.mean(np.array(adversarial_scores) > 2.0)
            },
            'throughput_per_second': len(recent_predictions) / (window_minutes * 60)
        }


class PrometheusMetricsCollector(MetricsCollector):
    """Prometheus metrics collector for production monitoring."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None, job_name: str = 'hyperconformal'):
        if not PROMETHEUS_AVAILABLE:
            raise ImportError("Prometheus client not available. Install with: pip install prometheus_client")
        
        self.registry = registry or CollectorRegistry()
        self.job_name = job_name
        
        # Define Prometheus metrics
        self.prediction_counter = Counter(
            'hyperconformal_predictions_total',
            'Total number of predictions made',
            registry=self.registry
        )
        
        self.prediction_time_histogram = Histogram(
            'hyperconformal_prediction_duration_seconds',
            'Time spent making predictions',
            registry=self.registry
        )
        
        self.confidence_histogram = Histogram(
            'hyperconformal_confidence_score',
            'Distribution of confidence scores',
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
            registry=self.registry
        )
        
        self.set_size_histogram = Histogram(
            'hyperconformal_set_size',
            'Distribution of prediction set sizes',
            buckets=[1, 2, 3, 4, 5, 10, 20, 50],
            registry=self.registry
        )
        
        self.drift_gauge = Gauge(
            'hyperconformal_drift_score',
            'Current drift detection score',
            registry=self.registry
        )
        
        self.adversarial_gauge = Gauge(
            'hyperconformal_adversarial_score',
            'Current adversarial detection score',
            registry=self.registry
        )
        
        self.memory_gauge = Gauge(
            'hyperconformal_memory_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.model_updates_counter = Counter(
            'hyperconformal_model_updates_total',
            'Total number of model updates',
            registry=self.registry
        )
        
        self.update_time_histogram = Histogram(
            'hyperconformal_update_duration_seconds',
            'Time spent updating model',
            registry=self.registry
        )
    
    def record_prediction(self, metrics: PredictionMetrics) -> None:
        """Record prediction metrics to Prometheus."""
        self.prediction_counter.inc()
        self.prediction_time_histogram.observe(metrics.prediction_time)
        self.confidence_histogram.observe(metrics.confidence_score)
        self.set_size_histogram.observe(metrics.set_size)
        self.drift_gauge.set(metrics.drift_score)
        self.adversarial_gauge.set(metrics.adversarial_score)
        self.memory_gauge.set(metrics.memory_usage)
    
    def record_model_update(self, update_time: float, samples_count: int) -> None:
        """Record model update metrics to Prometheus."""
        self.model_updates_counter.inc()
        self.update_time_histogram.observe(update_time)
    
    def get_metrics_summary(self, window_minutes: int = 5) -> Dict[str, Any]:
        """Get metrics summary (placeholder - use Prometheus queries in practice)."""
        # In practice, this would query Prometheus for aggregated metrics
        return {
            'status': 'prometheus_backend',
            'note': 'Use Prometheus queries for detailed metrics'
        }
    
    def push_to_gateway(self, gateway_url: str) -> None:
        """Push metrics to Prometheus pushgateway."""
        try:
            push_to_gateway(gateway_url, job=self.job_name, registry=self.registry)
        except Exception as e:
            logger.error(f"Failed to push metrics to gateway: {e}")


class OpenTelemetryTracer:
    """OpenTelemetry tracing integration."""
    
    def __init__(self, service_name: str = 'hyperconformal', jaeger_endpoint: Optional[str] = None):
        if not OTEL_AVAILABLE:
            raise ImportError("OpenTelemetry not available. Install with: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-jaeger")
        
        # Configure tracer
        trace.set_tracer_provider(TracerProvider())
        self.tracer = trace.get_tracer(service_name)
        
        # Configure Jaeger exporter if endpoint provided
        if jaeger_endpoint:
            jaeger_exporter = JaegerExporter(
                agent_host_name=jaeger_endpoint.split(':')[0],
                agent_port=int(jaeger_endpoint.split(':')[1]) if ':' in jaeger_endpoint else 14268,
            )
            
            span_processor = BatchSpanProcessor(jaeger_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
    
    def trace_prediction(self, model_name: str, input_shape: Tuple[int, ...]):
        """Create span for prediction tracing."""
        return self.tracer.start_as_current_span(
            "hyperconformal.predict",
            attributes={
                "model.name": model_name,
                "input.shape": str(input_shape)
            }
        )
    
    def trace_model_update(self, samples_count: int):
        """Create span for model update tracing."""
        return self.tracer.start_as_current_span(
            "hyperconformal.update",
            attributes={
                "update.samples": samples_count
            }
        )


class AlertManager:
    """Alert management system for monitoring."""
    
    def __init__(self):
        self.rules = []
        self.alert_history = deque(maxlen=1000)
        self.cooldown_timers = {}
        self._lock = threading.Lock()
    
    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        with self._lock:
            self.rules.append(rule)
    
    def evaluate_alerts(self, metrics_data: List[PredictionMetrics]) -> List[Dict[str, Any]]:
        """Evaluate all alert rules against recent metrics."""
        if not metrics_data:
            return []
        
        current_time = time.time()
        triggered_alerts = []
        
        with self._lock:
            for rule in self.rules:
                # Check cooldown
                last_triggered = self.cooldown_timers.get(rule.name, 0)
                if current_time - last_triggered < rule.cooldown_seconds:
                    continue
                
                # Evaluate rule condition
                try:
                    # Use recent metrics within window
                    window_data = metrics_data[-rule.window_size:] if len(metrics_data) >= rule.window_size else metrics_data
                    
                    # Create evaluation context
                    context = self._create_evaluation_context(window_data)
                    
                    # Evaluate condition safely (replace eval with safe expression evaluation)
                    try:
                        # For now, use a simple condition parser instead of eval
                        condition_met = self._safe_evaluate_condition(rule.condition, context)
                    except Exception as e:
                        logger.warning(f"Failed to evaluate rule condition '{rule.condition}': {e}")
                        condition_met = False
                    
                    if condition_met:
                        alert = {
                            'rule_name': rule.name,
                            'severity': rule.severity,
                            'timestamp': current_time,
                            'message': f"Alert triggered: {rule.name}",
                            'context': context
                        }
                        
                        triggered_alerts.append(alert)
                        self.alert_history.append(alert)
                        self.cooldown_timers[rule.name] = current_time
                        
                        # Execute action if provided
                        if rule.action:
                            try:
                                rule.action(alert)
                            except Exception as e:
                                logger.error(f"Alert action failed for {rule.name}: {e}")
                
                except Exception as e:
                    logger.error(f"Failed to evaluate alert rule {rule.name}: {e}")
        
        return triggered_alerts
    
    def _create_evaluation_context(self, metrics_data: List[PredictionMetrics]) -> Dict[str, Any]:
        """Create context for alert rule evaluation."""
        if not metrics_data:
            return {}
        
        # Extract metrics arrays
        prediction_times = [m.prediction_time for m in metrics_data]
        confidence_scores = [m.confidence_score for m in metrics_data]
        set_sizes = [m.set_size for m in metrics_data]
        drift_scores = [m.drift_score for m in metrics_data]
        adversarial_scores = [m.adversarial_score for m in metrics_data]
        
        return {
            # Basic statistics
            'count': len(metrics_data),
            'avg_prediction_time': np.mean(prediction_times),
            'max_prediction_time': np.max(prediction_times),
            'avg_confidence': np.mean(confidence_scores),
            'min_confidence': np.min(confidence_scores),
            'avg_set_size': np.mean(set_sizes),
            'max_set_size': np.max(set_sizes),
            'max_drift_score': np.max(drift_scores),
            'avg_drift_score': np.mean(drift_scores),
            'max_adversarial_score': np.max(adversarial_scores),
            
            # Derived metrics
            'high_drift_fraction': np.mean(np.array(drift_scores) > 0.1),
            'low_confidence_fraction': np.mean(np.array(confidence_scores) < 0.5),
            'large_set_fraction': np.mean(np.array(set_sizes) > 3),
            'adversarial_fraction': np.mean(np.array(adversarial_scores) > 2.0),
            
            # Numpy functions for complex conditions
            'np': np
        }
    
    def get_recent_alerts(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get recent alerts within the specified time window."""
        current_time = time.time()
        window_start = current_time - (minutes * 60)
        
        with self._lock:
            return [
                alert for alert in self.alert_history 
                if alert['timestamp'] >= window_start
            ]


class HyperConformalMonitor:
    """Main monitoring coordinator for HyperConformal systems."""
    
    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        tracer: Optional[OpenTelemetryTracer] = None,
        alert_manager: Optional[AlertManager] = None,
        enable_logging: bool = True
    ):
        """
        Initialize monitoring system.
        
        Args:
            metrics_collector: Backend for metrics collection
            tracer: OpenTelemetry tracer for distributed tracing
            alert_manager: Alert management system
            enable_logging: Whether to enable structured logging
        """
        self.metrics_collector = metrics_collector or InMemoryMetricsCollector()
        self.tracer = tracer
        self.alert_manager = alert_manager or AlertManager()
        
        # Metrics buffer for alert evaluation
        self.metrics_buffer = deque(maxlen=10000)
        self._lock = threading.Lock()
        
        # Setup logging
        if enable_logging:
            self._setup_structured_logging()
        
        # Background monitoring thread
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        # Default alert rules
        self._setup_default_alerts()
    
    def _setup_structured_logging(self):
        """Setup structured logging for monitoring."""
        # Create a custom logger for monitoring events
        self.monitor_logger = logging.getLogger('hyperconformal.monitoring')
        
        # JSON formatter for structured logs
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    'timestamp': time.time(),
                    'level': record.levelname,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno
                }
                
                # Add extra fields if present
                if hasattr(record, 'extra_fields'):
                    log_entry.update(record.extra_fields)
                
                return json.dumps(log_entry)
        
        # Add JSON handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        self.monitor_logger.addHandler(handler)
        self.monitor_logger.setLevel(logging.INFO)
    
    def _setup_default_alerts(self):
        """Setup default alert rules."""
        # High prediction time alert
        self.alert_manager.add_rule(AlertRule(
            name="high_prediction_time",
            condition="avg_prediction_time > 1.0",  # > 1 second
            threshold=1.0,
            window_size=100,
            cooldown_seconds=300,  # 5 minutes
            severity="warning"
        ))
        
        # Low confidence alert
        self.alert_manager.add_rule(AlertRule(
            name="low_confidence_predictions",
            condition="low_confidence_fraction > 0.5",  # >50% low confidence
            threshold=0.5,
            window_size=200,
            cooldown_seconds=600,  # 10 minutes
            severity="warning"
        ))
        
        # Drift detection alert
        self.alert_manager.add_rule(AlertRule(
            name="concept_drift_detected",
            condition="high_drift_fraction > 0.3",  # >30% high drift
            threshold=0.3,
            window_size=100,
            cooldown_seconds=1800,  # 30 minutes
            severity="critical"
        ))
        
        # Adversarial attack alert
        self.alert_manager.add_rule(AlertRule(
            name="adversarial_attack",
            condition="adversarial_fraction > 0.1",  # >10% adversarial
            threshold=0.1,
            window_size=50,
            cooldown_seconds=60,  # 1 minute
            severity="critical"
        ))
    
    def record_prediction(
        self,
        prediction_time: float,
        confidence_score: float,
        set_size: int,
        drift_score: float = 0.0,
        adversarial_score: float = 0.0,
        memory_usage: int = 0,
        energy_estimate: float = 0.0
    ) -> None:
        """Record metrics for a prediction."""
        metrics = PredictionMetrics(
            timestamp=time.time(),
            prediction_time=prediction_time,
            confidence_score=confidence_score,
            set_size=set_size,
            drift_score=drift_score,
            adversarial_score=adversarial_score,
            memory_usage=memory_usage,
            energy_estimate=energy_estimate
        )
        
        # Record to collector
        self.metrics_collector.record_prediction(metrics)
        
        # Add to buffer for alerting
        with self._lock:
            self.metrics_buffer.append(metrics)
        
        # Structured logging
        if hasattr(self, 'monitor_logger'):
            self.monitor_logger.info("Prediction recorded", extra={
                'extra_fields': asdict(metrics)
            })
    
    def record_model_update(self, update_time: float, samples_count: int) -> None:
        """Record model update metrics."""
        self.metrics_collector.record_model_update(update_time, samples_count)
        
        # Structured logging
        if hasattr(self, 'monitor_logger'):
            self.monitor_logger.info("Model updated", extra={
                'extra_fields': {
                    'update_time': update_time,
                    'samples_count': samples_count,
                    'timestamp': time.time()
                }
            })
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self._monitoring_active:
            try:
                time.sleep(30)  # Check every 30 seconds
                
                # Evaluate alerts
                with self._lock:
                    metrics_data = list(self.metrics_buffer)
                
                if metrics_data:
                    alerts = self.alert_manager.evaluate_alerts(metrics_data)
                    
                    # Log alerts
                    for alert in alerts:
                        if hasattr(self, 'monitor_logger'):
                            self.monitor_logger.warning("Alert triggered", extra={
                                'extra_fields': alert
                            })
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        # Get recent metrics summary
        metrics_summary = self.metrics_collector.get_metrics_summary(window_minutes=5)
        
        # Get recent alerts
        recent_alerts = self.alert_manager.get_recent_alerts(minutes=60)
        
        # Get system health info
        health_info = {
            'monitoring_active': self._monitoring_active,
            'metrics_buffer_size': len(self.metrics_buffer),
            'alert_rules_count': len(self.alert_manager.rules),
            'total_alerts_today': len([
                a for a in self.alert_manager.alert_history
                if time.time() - a['timestamp'] < 24 * 3600
            ])
        }
        
        return {
            'timestamp': time.time(),
            'metrics_summary': metrics_summary,
            'recent_alerts': recent_alerts,
            'health_info': health_info
        }
    
    def _safe_evaluate_condition(self, condition: str, context: Dict) -> bool:
        """Safely evaluate rule condition without using eval()."""
        # Simple condition parser for common patterns
        # This replaces the dangerous eval() with safe condition matching
        
        # Replace context variables in condition
        for var, value in context.items():
            condition = condition.replace(var, str(value))
        
        # Handle simple comparison operators
        if '<' in condition:
            parts = condition.split('<')
            if len(parts) == 2:
                try:
                    left = float(parts[0].strip())
                    right = float(parts[1].strip()) 
                    return left < right
                except ValueError:
                    return False
        elif '>' in condition:
            parts = condition.split('>')
            if len(parts) == 2:
                try:
                    left = float(parts[0].strip())
                    right = float(parts[1].strip())
                    return left > right
                except ValueError:
                    return False
        elif '==' in condition:
            parts = condition.split('==')
            if len(parts) == 2:
                return parts[0].strip() == parts[1].strip()
        
        # Default to False for unknown conditions
        logger.warning(f"Unknown condition format: {condition}")
        return False
    
    def export_metrics(self, format: str = 'json') -> str:
        """Export metrics in specified format."""
        dashboard_data = self.get_dashboard_data()
        
        if format == 'json':
            return json.dumps(dashboard_data, indent=2)
        elif format == 'prometheus':
            # Convert to Prometheus format (simplified)
            lines = []
            
            summary = dashboard_data['metrics_summary']
            if isinstance(summary, dict) and 'total_predictions' in summary:
                lines.append(f"hyperconformal_predictions_total {summary['total_predictions']}")
                
                if 'prediction_time' in summary:
                    lines.append(f"hyperconformal_avg_prediction_time {summary['prediction_time']['mean']}")
                
                if 'throughput_per_second' in summary:
                    lines.append(f"hyperconformal_throughput_per_second {summary['throughput_per_second']}")
            
            return '\n'.join(lines)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def shutdown(self):
        """Shutdown monitoring system."""
        self._monitoring_active = False
        if self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)


# Convenience functions for easy setup
def setup_basic_monitoring() -> HyperConformalMonitor:
    """Setup basic monitoring with in-memory collector."""
    return HyperConformalMonitor()


def setup_production_monitoring(
    prometheus_registry: Optional[CollectorRegistry] = None,
    jaeger_endpoint: Optional[str] = None
) -> HyperConformalMonitor:
    """Setup production monitoring with Prometheus and Jaeger."""
    # Setup metrics collector
    metrics_collector = None
    if PROMETHEUS_AVAILABLE:
        metrics_collector = PrometheusMetricsCollector(registry=prometheus_registry)
    else:
        logger.warning("Prometheus not available, using in-memory collector")
        metrics_collector = InMemoryMetricsCollector()
    
    # Setup tracer
    tracer = None
    if OTEL_AVAILABLE and jaeger_endpoint:
        try:
            tracer = OpenTelemetryTracer(jaeger_endpoint=jaeger_endpoint)
        except Exception as e:
            logger.warning(f"Failed to setup Jaeger tracing: {e}")
    
    return HyperConformalMonitor(
        metrics_collector=metrics_collector,
        tracer=tracer,
        enable_logging=True
    )
#!/usr/bin/env python3
"""
Auto-Scaling System for HyperConformal
Generation 3: Dynamic resource management and load balancing
"""

import time
import threading
import multiprocessing
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import queue
import statistics

@dataclass
class ResourceMetrics:
    """Metrics for resource utilization."""
    cpu_usage: float
    memory_usage: float
    throughput: float
    response_time: float
    queue_length: int
    error_rate: float
    timestamp: float = 0
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()

class AutoScaler:
    """Automatic scaling system for HyperConformal resources."""
    
    def __init__(self, min_workers: int = 1, max_workers: int = None, 
                 scale_up_threshold: float = 0.7, scale_down_threshold: float = 0.3):
        self.min_workers = min_workers
        self.max_workers = max_workers or multiprocessing.cpu_count() * 2
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        
        self.current_workers = min_workers
        self.metrics_history = []
        self.scaling_decisions = []
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self, check_interval: float = 5.0):
        """Start monitoring and auto-scaling."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(check_interval,)
        )
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _monitoring_loop(self, check_interval: float):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only recent metrics (last 20 measurements)
                if len(self.metrics_history) > 20:
                    self.metrics_history = self.metrics_history[-20:]
                
                # Make scaling decision
                decision = self._make_scaling_decision(metrics)
                if decision != 'no_action':
                    self.scaling_decisions.append({
                        'decision': decision,
                        'timestamp': time.time(),
                        'metrics': metrics,
                        'workers_before': self.current_workers
                    })
                
                time.sleep(check_interval)
                
            except Exception as e:
                print(f"Auto-scaling monitoring error: {e}")
                time.sleep(check_interval)
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics."""
        # Simulate metric collection
        import random
        
        # Simulate varying load
        base_cpu = 0.3 + 0.4 * random.random()
        base_memory = 0.2 + 0.3 * random.random()
        base_throughput = 100 + 50 * random.random()
        base_response_time = 0.1 + 0.2 * random.random()
        
        # Adjust based on current worker count
        worker_factor = self.min_workers / max(self.current_workers, 1)
        
        metrics = ResourceMetrics(
            cpu_usage=min(1.0, base_cpu * worker_factor),
            memory_usage=min(1.0, base_memory * worker_factor),
            throughput=base_throughput * self.current_workers,
            response_time=base_response_time * worker_factor,
            queue_length=max(0, int(20 * worker_factor - 10)),
            error_rate=max(0, (worker_factor - 1) * 0.1)  # Errors increase with overload
        )
        
        return metrics
    
    def _make_scaling_decision(self, current_metrics: ResourceMetrics) -> str:
        """Make scaling decision based on metrics."""
        if len(self.metrics_history) < 3:
            return 'no_action'  # Need sufficient data
        
        # Calculate trend over recent metrics
        recent_metrics = self.metrics_history[-3:]
        
        avg_cpu = statistics.mean([m.cpu_usage for m in recent_metrics])
        avg_response_time = statistics.mean([m.response_time for m in recent_metrics])
        avg_queue_length = statistics.mean([m.queue_length for m in recent_metrics])
        
        # Scale up conditions
        should_scale_up = (
            (avg_cpu > self.scale_up_threshold or 
             avg_response_time > 0.5 or 
             avg_queue_length > 10) and
            self.current_workers < self.max_workers
        )
        
        # Scale down conditions
        should_scale_down = (
            (avg_cpu < self.scale_down_threshold and 
             avg_response_time < 0.2 and 
             avg_queue_length < 2) and
            self.current_workers > self.min_workers
        )
        
        if should_scale_up:
            new_workers = min(self.current_workers + 1, self.max_workers)
            self.current_workers = new_workers
            return 'scale_up'
        elif should_scale_down:
            new_workers = max(self.current_workers - 1, self.min_workers)
            self.current_workers = new_workers
            return 'scale_down'
        else:
            return 'no_action'
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        latest_metrics = self.metrics_history[-1] if self.metrics_history else None
        recent_decisions = self.scaling_decisions[-5:] if self.scaling_decisions else []
        
        return {
            'current_workers': self.current_workers,
            'min_workers': self.min_workers,
            'max_workers': self.max_workers,
            'latest_metrics': latest_metrics.__dict__ if latest_metrics else None,
            'recent_decisions': recent_decisions,
            'total_scaling_events': len(self.scaling_decisions)
        }

class LoadBalancer:
    """Load balancing system for distributed processing."""
    
    def __init__(self, initial_workers: int = 2):
        self.workers = {}
        self.worker_stats = {}
        self.request_queue = queue.Queue()
        self.balancing_strategy = 'round_robin'
        self.current_worker_index = 0
        
        # Initialize workers
        for i in range(initial_workers):
            self._add_worker(f"worker_{i}")
    
    def _add_worker(self, worker_id: str):
        """Add new worker to the pool."""
        self.workers[worker_id] = {
            'id': worker_id,
            'active': True,
            'created_at': time.time(),
            'task_queue': queue.Queue(maxsize=50)
        }
        
        self.worker_stats[worker_id] = {
            'tasks_completed': 0,
            'total_processing_time': 0,
            'current_load': 0,
            'error_count': 0
        }
    
    def _remove_worker(self, worker_id: str):
        """Remove worker from the pool."""
        if worker_id in self.workers:
            self.workers[worker_id]['active'] = False
            # Don't delete immediately to allow graceful shutdown
    
    def add_workers(self, count: int):
        """Add multiple workers."""
        current_count = len([w for w in self.workers.values() if w['active']])
        
        for i in range(count):
            worker_id = f"worker_{current_count + i}"
            self._add_worker(worker_id)
    
    def remove_workers(self, count: int):
        """Remove multiple workers."""
        active_workers = [w_id for w_id, w in self.workers.items() if w['active']]
        
        for i in range(min(count, len(active_workers) - 1)):  # Keep at least one
            self._remove_worker(active_workers[i])
    
    def select_worker(self) -> Optional[str]:
        """Select worker based on load balancing strategy."""
        active_workers = [w_id for w_id, w in self.workers.items() if w['active']]
        
        if not active_workers:
            return None
        
        if self.balancing_strategy == 'round_robin':
            worker_id = active_workers[self.current_worker_index % len(active_workers)]
            self.current_worker_index += 1
            return worker_id
        
        elif self.balancing_strategy == 'least_loaded':
            # Select worker with lowest current load
            min_load = float('inf')
            selected_worker = None
            
            for worker_id in active_workers:
                load = self.worker_stats[worker_id]['current_load']
                if load < min_load:
                    min_load = load
                    selected_worker = worker_id
            
            return selected_worker
        
        else:
            # Default to round robin
            return active_workers[0]
    
    def submit_task(self, task_data: Any, task_type: str = 'encoding') -> Optional[str]:
        """Submit task to be load balanced."""
        worker_id = self.select_worker()
        if not worker_id:
            return None
        
        task = {
            'id': f"task_{time.time()}",
            'data': task_data,
            'type': task_type,
            'submitted_at': time.time(),
            'worker_id': worker_id
        }
        
        try:
            self.workers[worker_id]['task_queue'].put_nowait(task)
            self.worker_stats[worker_id]['current_load'] += 1
            return task['id']
        except queue.Full:
            # Worker queue is full, try another worker
            return None
    
    def get_load_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        active_workers = [w_id for w_id, w in self.workers.items() if w['active']]
        
        total_tasks = sum(stats['tasks_completed'] for stats in self.worker_stats.values())
        total_load = sum(stats['current_load'] for stats in self.worker_stats.values())
        
        worker_loads = {
            w_id: self.worker_stats[w_id]['current_load'] 
            for w_id in active_workers
        }
        
        return {
            'active_workers': len(active_workers),
            'total_workers': len(self.workers),
            'total_tasks_completed': total_tasks,
            'current_total_load': total_load,
            'worker_loads': worker_loads,
            'balancing_strategy': self.balancing_strategy
        }

class AdaptiveResourceManager:
    """Comprehensive adaptive resource management system."""
    
    def __init__(self):
        self.auto_scaler = AutoScaler(min_workers=2, max_workers=8)
        self.load_balancer = LoadBalancer(initial_workers=2)
        self.running = False
        
    def start(self):
        """Start adaptive resource management."""
        self.running = True
        self.auto_scaler.start_monitoring(check_interval=3.0)
        
        # Start resource adjustment thread
        self.adjustment_thread = threading.Thread(target=self._resource_adjustment_loop)
        self.adjustment_thread.daemon = True
        self.adjustment_thread.start()
    
    def stop(self):
        """Stop adaptive resource management."""
        self.running = False
        self.auto_scaler.stop_monitoring()
        
        if hasattr(self, 'adjustment_thread'):
            self.adjustment_thread.join(timeout=1.0)
    
    def _resource_adjustment_loop(self):
        """Adjust load balancer resources based on auto-scaler decisions."""
        while self.running:
            try:
                # Get scaling status
                status = self.auto_scaler.get_scaling_status()
                current_workers = status['current_workers']
                
                # Get load balancer status
                load_stats = self.load_balancer.get_load_stats()
                active_workers = load_stats['active_workers']
                
                # Adjust load balancer workers to match auto-scaler
                if current_workers > active_workers:
                    self.load_balancer.add_workers(current_workers - active_workers)
                elif current_workers < active_workers:
                    self.load_balancer.remove_workers(active_workers - current_workers)
                
                time.sleep(2.0)
                
            except Exception as e:
                print(f"Resource adjustment error: {e}")
                time.sleep(2.0)
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all systems."""
        return {
            'auto_scaler': self.auto_scaler.get_scaling_status(),
            'load_balancer': self.load_balancer.get_load_stats(),
            'running': self.running
        }

# Testing and demonstration
if __name__ == "__main__":
    print("📈 Testing Auto-Scaling System")
    print("="*50)
    
    # Test auto-scaler
    auto_scaler = AutoScaler(min_workers=1, max_workers=6)
    auto_scaler.start_monitoring(check_interval=1.0)
    
    print("Auto-scaler started, monitoring for 10 seconds...")
    time.sleep(10)
    
    status = auto_scaler.get_scaling_status()
    print(f"Auto-scaler status: {status['current_workers']} workers")
    print(f"Scaling events: {status['total_scaling_events']}")
    
    auto_scaler.stop_monitoring()
    
    # Test load balancer
    print("\nTesting load balancer...")
    load_balancer = LoadBalancer(initial_workers=3)
    
    # Submit some tasks
    task_ids = []
    for i in range(10):
        task_id = load_balancer.submit_task(f"test_data_{i}", "encoding")
        if task_id:
            task_ids.append(task_id)
    
    load_stats = load_balancer.get_load_stats()
    print(f"Load balancer stats: {load_stats}")
    
    # Test adaptive resource manager
    print("\nTesting adaptive resource manager...")
    manager = AdaptiveResourceManager()
    manager.start()
    
    print("Adaptive manager running for 8 seconds...")
    time.sleep(8)
    
    comprehensive_status = manager.get_comprehensive_status()
    print(f"Comprehensive status: {comprehensive_status}")
    
    manager.stop()
    
    print("\n🎉 Auto-scaling system tests completed!")

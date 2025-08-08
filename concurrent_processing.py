#!/usr/bin/env python3
"""
Concurrent Processing System for HyperConformal
Generation 3: High-throughput processing with parallel execution
"""

import time
import threading
import multiprocessing
import asyncio
from typing import Dict, List, Any, Callable, Optional, Coroutine
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
from dataclasses import dataclass

@dataclass
class ProcessingTask:
    """Structure for processing tasks."""
    id: str
    data: Any
    task_type: str
    priority: int = 0
    created_at: float = 0
    
    def __post_init__(self):
        if self.created_at == 0:
            self.created_at = time.time()

class ConcurrentHDCProcessor:
    """High-performance concurrent HDC processing system."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=self.max_workers)
        self.task_queue = queue.PriorityQueue()
        self.results = {}
        self.processing_stats = {
            'tasks_completed': 0,
            'total_processing_time': 0,
            'average_throughput': 0
        }
        
    def submit_encoding_task(self, task_id: str, input_data: List[float], 
                           priority: int = 0, use_process: bool = False) -> str:
        """Submit encoding task for concurrent processing."""
        task = ProcessingTask(
            id=task_id,
            data=input_data,
            task_type='encoding',
            priority=priority
        )
        
        if use_process:
            future = self.process_executor.submit(self._encode_worker, task.data)
        else:
            future = self.thread_executor.submit(self._encode_worker, task.data)
        
        # Store future for result retrieval
        self.results[task_id] = {
            'future': future,
            'task': task,
            'submitted_at': time.time()
        }
        
        return task_id
    
    def submit_prediction_task(self, task_id: str, scores: List[float], 
                             alpha: float = 0.1, priority: int = 0) -> str:
        """Submit conformal prediction task for concurrent processing."""
        task = ProcessingTask(
            id=task_id,
            data={'scores': scores, 'alpha': alpha},
            task_type='prediction',
            priority=priority
        )
        
        future = self.thread_executor.submit(self._predict_worker, scores, alpha)
        
        self.results[task_id] = {
            'future': future,
            'task': task,
            'submitted_at': time.time()
        }
        
        return task_id
    
    def _encode_worker(self, input_data: List[float]) -> List[int]:
        """Worker function for HDC encoding."""
        # Simulate optimized HDC encoding
        threshold = 0.0
        binary_input = [1 if x > threshold else 0 for x in input_data]
        
        # Simulate projection to hypervector
        hv_dim = len(input_data) * 10  # 10x expansion
        hypervector = []
        
        for i in range(hv_dim):
            # Simple hash-based projection
            bit_sum = sum(binary_input[j % len(binary_input)] 
                         for j in range(i, i + 3))
            hypervector.append(bit_sum % 2)
        
        return hypervector
    
    def _predict_worker(self, scores: List[float], alpha: float) -> List[int]:
        """Worker function for conformal prediction."""
        # Simulate calibration scores
        calibration_scores = [0.1 * i for i in range(50)]
        
        # Compute quantile
        import math
        n = len(calibration_scores)
        q_index = max(0, min(math.ceil((n + 1) * (1 - alpha)) - 1, n - 1))
        quantile = sorted(calibration_scores)[q_index]
        
        # Generate prediction set
        prediction_set = [i for i, score in enumerate(scores) if score >= quantile]
        
        return prediction_set
    
    def get_result(self, task_id: str, timeout: float = None) -> Optional[Any]:
        """Get result of completed task."""
        if task_id not in self.results:
            return None
        
        result_info = self.results[task_id]
        future = result_info['future']
        
        try:
            result = future.result(timeout=timeout)
            
            # Update statistics
            processing_time = time.time() - result_info['submitted_at']
            self.processing_stats['tasks_completed'] += 1
            self.processing_stats['total_processing_time'] += processing_time
            
            # Calculate average throughput
            if self.processing_stats['tasks_completed'] > 0:
                avg_time = (self.processing_stats['total_processing_time'] / 
                          self.processing_stats['tasks_completed'])
                self.processing_stats['average_throughput'] = 1.0 / avg_time if avg_time > 0 else 0
            
            # Clean up
            del self.results[task_id]
            
            return result
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_all_results(self, timeout: float = None) -> Dict[str, Any]:
        """Get all completed results."""
        results = {}
        
        for task_id in list(self.results.keys()):
            result = self.get_result(task_id, timeout=timeout)
            if result is not None:
                results[task_id] = result
        
        return results
    
    def process_batch_concurrent(self, input_batch: List[List[float]], 
                               task_prefix: str = "batch") -> List[List[int]]:
        """Process batch with concurrent execution."""
        start_time = time.time()
        
        # Submit all tasks
        task_ids = []
        for i, input_data in enumerate(input_batch):
            task_id = f"{task_prefix}_{i}"
            self.submit_encoding_task(task_id, input_data)
            task_ids.append(task_id)
        
        # Collect results in order
        results = []
        for task_id in task_ids:
            result = self.get_result(task_id, timeout=30.0)
            if result and 'error' not in result:
                results.append(result)
            else:
                # Fallback for failed tasks
                results.append([0] * (len(input_batch[0]) * 10))
        
        processing_time = time.time() - start_time
        throughput = len(input_batch) / processing_time if processing_time > 0 else 0
        
        return results
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        active_tasks = len(self.results)
        
        return {
            **self.processing_stats,
            'active_tasks': active_tasks,
            'max_workers': self.max_workers
        }
    
    def shutdown(self):
        """Shutdown executors."""
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)

class StreamingProcessor:
    """Streaming data processor for real-time applications."""
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.input_buffer = queue.Queue(maxsize=buffer_size)
        self.output_buffer = queue.Queue(maxsize=buffer_size)
        self.processing = False
        self.worker_threads = []
        self.stats = {
            'items_processed': 0,
            'processing_rate': 0,
            'buffer_utilization': 0
        }
        
    def start_processing(self, num_workers: int = 2):
        """Start streaming processing with worker threads."""
        self.processing = True
        
        # Start worker threads
        for i in range(num_workers):
            worker = threading.Thread(target=self._processing_worker, 
                                    args=(f"worker_{i}",))
            worker.daemon = True
            worker.start()
            self.worker_threads.append(worker)
    
    def _processing_worker(self, worker_id: str):
        """Worker thread for streaming processing."""
        processor = ConcurrentHDCProcessor(max_workers=1)
        
        while self.processing:
            try:
                # Get item from input buffer
                item = self.input_buffer.get(timeout=0.1)
                
                # Process item
                if item['type'] == 'encoding':
                    result = processor._encode_worker(item['data'])
                elif item['type'] == 'prediction':
                    result = processor._predict_worker(
                        item['data']['scores'], 
                        item['data']['alpha']
                    )
                else:
                    result = None
                
                # Put result in output buffer
                if result is not None:
                    output_item = {
                        'id': item.get('id', 'unknown'),
                        'result': result,
                        'processed_by': worker_id,
                        'processed_at': time.time()
                    }
                    
                    try:
                        self.output_buffer.put_nowait(output_item)
                        self.stats['items_processed'] += 1
                    except queue.Full:
                        # Output buffer full, drop item
                        pass
                
                self.input_buffer.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                # Log error and continue
                print(f"Processing error in {worker_id}: {e}")
    
    def add_item(self, item: Dict[str, Any]) -> bool:
        """Add item to processing queue."""
        try:
            self.input_buffer.put_nowait(item)
            return True
        except queue.Full:
            return False
    
    def get_result(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """Get processed result."""
        try:
            return self.output_buffer.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get streaming processing statistics."""
        input_utilization = (self.input_buffer.qsize() / self.buffer_size) * 100
        output_utilization = (self.output_buffer.qsize() / self.buffer_size) * 100
        
        return {
            **self.stats,
            'input_buffer_utilization': input_utilization,
            'output_buffer_utilization': output_utilization,
            'input_queue_size': self.input_buffer.qsize(),
            'output_queue_size': self.output_buffer.qsize(),
            'active_workers': len(self.worker_threads)
        }
    
    def stop_processing(self):
        """Stop streaming processing."""
        self.processing = False
        
        # Wait for workers to finish
        for worker in self.worker_threads:
            worker.join(timeout=1.0)

# Testing and demonstration
if __name__ == "__main__":
    print("🔄 Testing Concurrent Processing System")
    print("="*50)
    
    # Test concurrent HDC processor
    processor = ConcurrentHDCProcessor(max_workers=4)
    
    # Submit multiple encoding tasks
    test_vectors = [
        [0.1 * i for i in range(50)],
        [0.2 * i for i in range(50)],
        [0.3 * i for i in range(50)],
        [0.4 * i for i in range(50)]
    ]
    
    print("Submitting concurrent encoding tasks...")
    task_ids = []
    start_time = time.time()
    
    for i, vector in enumerate(test_vectors):
        task_id = processor.submit_encoding_task(f"encode_{i}", vector)
        task_ids.append(task_id)
    
    # Get results
    results = []
    for task_id in task_ids:
        result = processor.get_result(task_id, timeout=10.0)
        if result:
            results.append(len(result))
    
    concurrent_time = time.time() - start_time
    print(f"Concurrent processing: {len(results)} vectors in {concurrent_time:.3f}s")
    print(f"Result dimensions: {results}")
    print(f"Processing stats: {processor.get_processing_stats()}")
    
    # Test batch processing
    batch_size = 8
    test_batch = [[0.1 * i * j for i in range(20)] for j in range(batch_size)]
    
    start_time = time.time()
    batch_results = processor.process_batch_concurrent(test_batch, "batch_test")
    batch_time = time.time() - start_time
    
    print(f"\nBatch processing: {len(batch_results)} vectors in {batch_time:.3f}s")
    print(f"Throughput: {len(batch_results) / batch_time:.1f} vectors/s")
    
    # Test streaming processor
    print("\nTesting streaming processor...")
    stream_processor = StreamingProcessor(buffer_size=100)
    stream_processor.start_processing(num_workers=2)
    
    # Add streaming data
    for i in range(10):
        item = {
            'id': f"stream_{i}",
            'type': 'encoding',
            'data': [0.1 * i * j for j in range(30)]
        }
        stream_processor.add_item(item)
    
    # Collect results
    time.sleep(0.5)  # Allow processing
    
    stream_results = []
    while True:
        result = stream_processor.get_result(timeout=0.1)
        if result is None:
            break
        stream_results.append(result['id'])
    
    print(f"Streaming results: {len(stream_results)} items processed")
    print(f"Streaming stats: {stream_processor.get_stats()}")
    
    # Cleanup
    stream_processor.stop_processing()
    processor.shutdown()
    
    print("\n🎉 Concurrent processing tests completed!")

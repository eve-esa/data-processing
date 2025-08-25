"""
Standardized parallel processing framework for consistent performance across pipelines.
"""

import asyncio
import gc
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

import psutil


class ProcessingMode(Enum):
    """Processing mode options."""
    SEQUENTIAL = "sequential"
    THREADED = "threaded"
    MULTIPROCESS = "multiprocess"
    ASYNC = "async"
    ADAPTIVE = "adaptive"


@dataclass
class ProcessingConfig:
    """Configuration for parallel processing."""
    mode: ProcessingMode = ProcessingMode.ADAPTIVE
    max_workers: Optional[int] = None
    chunk_size: int = 10
    memory_limit_gb: float = 8.0
    cpu_threshold: float = 80.0
    enable_monitoring: bool = True
    cleanup_interval: int = 100


class ParallelProcessor:
    """
    Unified parallel processing framework that automatically selects the best
    processing strategy based on task characteristics and system resources.
    """

    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize parallel processor.

        Args:
            config: Processing configuration
        """
        self.config = config or ProcessingConfig()
        self.logger = logging.getLogger(__name__)

        # Auto-detect optimal worker count if not specified
        if self.config.max_workers is None:
            self.config.max_workers = self._calculate_optimal_workers()

        # Performance tracking
        self.stats = {
            'total_processed': 0,
            'total_time': 0,
            'memory_peak': 0,
            'mode_usage': {},
        }

    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal number of workers based on system resources."""
        cpu_count = multiprocessing.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)

        # Conservative approach: use 75% of CPUs, limit based on memory
        cpu_workers = max(1, int(cpu_count * 0.75))
        memory_workers = max(1, int(memory_gb / 2))  # 2GB per worker

        optimal = min(cpu_workers, memory_workers, 16)  # Cap at 16
        self.logger.info(f"Calculated optimal workers: {optimal} (CPU: {cpu_workers}, Memory: {memory_workers})")
        return optimal

    def process_items(
        self,
        items: list[Any],
        processor_func: Callable,
        mode: Optional[ProcessingMode] = None,
        **kwargs,
    ) -> list[Any]:
        """
        Process a list of items using the optimal parallel strategy.

        Args:
            items: List of items to process
            processor_func: Function to process each item
            mode: Optional override for processing mode
            **kwargs: Additional arguments for processor function

        Returns:
            List of processed results
        """
        if not items:
            return []

        start_time = time.time()
        processing_mode = mode or self._select_optimal_mode(items, processor_func)

        self.logger.info(f"Processing {len(items)} items using {processing_mode.value} mode")

        try:
            if processing_mode == ProcessingMode.SEQUENTIAL:
                results = self._process_sequential(items, processor_func, **kwargs)
            elif processing_mode == ProcessingMode.THREADED:
                results = self._process_threaded(items, processor_func, **kwargs)
            elif processing_mode == ProcessingMode.MULTIPROCESS:
                results = self._process_multiprocess(items, processor_func, **kwargs)
            elif processing_mode == ProcessingMode.ASYNC:
                results = self._process_async(items, processor_func, **kwargs)
            else:  # ADAPTIVE
                results = self._process_adaptive(items, processor_func, **kwargs)

            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(processing_mode, len(items), processing_time)

            return results

        except Exception as e:
            self.logger.error(f"Processing failed with {processing_mode.value}: {e}")
            # Fallback to sequential processing
            if processing_mode != ProcessingMode.SEQUENTIAL:
                self.logger.info("Falling back to sequential processing")
                return self._process_sequential(items, processor_func, **kwargs)
            raise

    def _select_optimal_mode(self, items: list[Any], processor_func: Callable) -> ProcessingMode:
        """Select optimal processing mode based on task characteristics."""
        num_items = len(items)

        # Check system resources
        memory_available = psutil.virtual_memory().available / (1024**3)
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Simple heuristics for mode selection
        if num_items < 5 or memory_available < 2.0 or cpu_percent > 90:
            return ProcessingMode.SEQUENTIAL
        elif num_items < 20:
            return ProcessingMode.THREADED
        elif self._is_io_bound(processor_func) or self._is_cpu_bound(processor_func):
            return ProcessingMode.MULTIPROCESS if self._is_cpu_bound(processor_func) else ProcessingMode.THREADED
        else:
            return ProcessingMode.THREADED

    def _is_io_bound(self, func: Callable) -> bool:
        """Heuristic to determine if function is I/O bound."""
        func_name = getattr(func, '__name__', str(func))
        io_indicators = ['download', 'upload', 'read', 'write', 'fetch', 'request', 's3']
        return any(indicator in func_name.lower() for indicator in io_indicators)

    def _is_cpu_bound(self, func: Callable) -> bool:
        """Heuristic to determine if function is CPU bound."""
        func_name = getattr(func, '__name__', str(func))
        cpu_indicators = ['compute', 'calculate', 'process', 'transform', 'parse', 'analyze']
        return any(indicator in func_name.lower() for indicator in cpu_indicators)

    def _process_sequential(self, items: list[Any], processor_func: Callable, **kwargs) -> list[Any]:
        """Process items sequentially."""
        results = []
        for i, item in enumerate(items):
            try:
                result = processor_func(item, **kwargs)
                results.append(result)

                # Periodic cleanup
                if self.config.enable_monitoring and i % self.config.cleanup_interval == 0:
                    self._cleanup_memory()

            except Exception as e:
                self.logger.error(f"Sequential processing failed for item {i}: {e}")
                results.append(None)

        return results

    def _process_threaded(self, items: list[Any], processor_func: Callable, **kwargs) -> list[Any]:
        """Process items using thread pool."""
        results = [None] * len(items)

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(processor_func, item, **kwargs): i
                for i, item in enumerate(items)
            }

            # Collect results
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    self.logger.error(f"Threaded processing failed for item {index}: {e}")
                    results[index] = None

        return results

    def _process_multiprocess(self, items: list[Any], processor_func: Callable, **kwargs) -> list[Any]:
        """Process items using process pool."""
        try:
            # Create a wrapper function that includes kwargs
            def wrapper_func(item):
                return processor_func(item, **kwargs)

            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                results = list(executor.map(wrapper_func, items))

            return results

        except Exception as e:
            self.logger.warning(f"Multiprocess execution failed: {e}, falling back to threaded")
            return self._process_threaded(items, processor_func, **kwargs)

    def _process_async(self, items: list[Any], processor_func: Callable, **kwargs) -> list[Any]:
        """Process items using async/await."""
        try:
            # Check if function is async
            if not asyncio.iscoroutinefunction(processor_func):
                # Wrap sync function for async execution
                async def async_wrapper(item):
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(None, processor_func, item)

                return asyncio.run(self._async_process_all(items, async_wrapper, **kwargs))
            else:
                return asyncio.run(self._async_process_all(items, processor_func, **kwargs))

        except Exception as e:
            self.logger.warning(f"Async processing failed: {e}, falling back to threaded")
            return self._process_threaded(items, processor_func, **kwargs)

    async def _async_process_all(self, items: list[Any], async_func: Callable, **kwargs) -> list[Any]:
        """Process all items asynchronously."""
        semaphore = asyncio.Semaphore(self.config.max_workers)

        async def process_with_semaphore(item):
            async with semaphore:
                try:
                    return await async_func(item, **kwargs)
                except Exception as e:
                    self.logger.error(f"Async processing failed for item: {e}")
                    return None

        tasks = [process_with_semaphore(item) for item in items]
        return await asyncio.gather(*tasks, return_exceptions=False)

    def _process_adaptive(self, items: list[Any], processor_func: Callable, **kwargs) -> list[Any]:
        """Adaptive processing that switches modes based on performance."""
        chunk_size = min(self.config.chunk_size, len(items))

        # Process first chunk to determine best mode
        first_chunk = items[:chunk_size]

        # Try different modes on first chunk
        modes_to_try = [ProcessingMode.THREADED, ProcessingMode.SEQUENTIAL]
        if len(items) > 50:
            modes_to_try.insert(0, ProcessingMode.MULTIPROCESS)

        best_mode = ProcessingMode.SEQUENTIAL
        best_time = float('inf')

        for mode in modes_to_try:
            try:
                start_time = time.time()
                self._process_chunk(first_chunk, processor_func, mode, **kwargs)
                chunk_time = time.time() - start_time

                if chunk_time < best_time:
                    best_time = chunk_time
                    best_mode = mode

            except Exception as e:
                self.logger.warning(f"Mode {mode.value} failed during adaptation: {e}")
                continue

        self.logger.info(f"Adaptive mode selected: {best_mode.value}")

        # Process all items with best mode
        return self._process_chunk(items, processor_func, best_mode, **kwargs)

    def _process_chunk(self, items: list[Any], processor_func: Callable, mode: ProcessingMode, **kwargs) -> list[Any]:
        """Process a chunk of items with specified mode."""
        if mode == ProcessingMode.SEQUENTIAL:
            return self._process_sequential(items, processor_func, **kwargs)
        elif mode == ProcessingMode.THREADED:
            return self._process_threaded(items, processor_func, **kwargs)
        elif mode == ProcessingMode.MULTIPROCESS:
            return self._process_multiprocess(items, processor_func, **kwargs)
        else:
            return self._process_sequential(items, processor_func, **kwargs)

    def _cleanup_memory(self):
        """Clean up memory and log resource usage."""
        gc.collect()

        if self.config.enable_monitoring:
            memory_usage = psutil.virtual_memory().percent

            if memory_usage > self.stats['memory_peak']:
                self.stats['memory_peak'] = memory_usage

            if memory_usage > 85:
                self.logger.warning(f"High memory usage: {memory_usage:.1f}%")
                # Force more aggressive cleanup
                gc.collect()
                gc.collect()  # Call twice for better cleanup

    def _update_stats(self, mode: ProcessingMode, num_items: int, processing_time: float):
        """Update processing statistics."""
        self.stats['total_processed'] += num_items
        self.stats['total_time'] += processing_time

        mode_name = mode.value
        if mode_name not in self.stats['mode_usage']:
            self.stats['mode_usage'][mode_name] = {'count': 0, 'total_time': 0}

        self.stats['mode_usage'][mode_name]['count'] += 1
        self.stats['mode_usage'][mode_name]['total_time'] += processing_time

    def get_performance_report(self) -> dict[str, Any]:
        """Get performance statistics report."""
        total_time = self.stats['total_time']
        if total_time == 0:
            return {"error": "No processing completed yet"}

        report = {
            'total_items_processed': self.stats['total_processed'],
            'total_processing_time': total_time,
            'average_time_per_item': total_time / max(1, self.stats['total_processed']),
            'peak_memory_usage_percent': self.stats['memory_peak'],
            'mode_performance': {},
        }

        for mode, stats in self.stats['mode_usage'].items():
            avg_time = stats['total_time'] / max(1, stats['count'])
            report['mode_performance'][mode] = {
                'usage_count': stats['count'],
                'total_time': stats['total_time'],
                'average_time': avg_time,
            }

        return report


# Convenience functions for common use cases
def process_files_parallel(
    file_paths: list[str],
    processor_func: Callable,
    max_workers: Optional[int] = None,
    mode: Optional[ProcessingMode] = None,
) -> list[Any]:
    """Convenience function for parallel file processing."""
    config = ProcessingConfig(
        max_workers=max_workers,
        mode=mode or ProcessingMode.ADAPTIVE,
    )
    processor = ParallelProcessor(config)
    return processor.process_items(file_paths, processor_func)


def process_data_parallel(
    data_items: list[Any],
    processor_func: Callable,
    chunk_size: int = 10,
    max_workers: Optional[int] = None,
) -> list[Any]:
    """Convenience function for parallel data processing."""
    config = ProcessingConfig(
        max_workers=max_workers,
        chunk_size=chunk_size,
        mode=ProcessingMode.ADAPTIVE,
    )
    processor = ParallelProcessor(config)
    return processor.process_items(data_items, processor_func)

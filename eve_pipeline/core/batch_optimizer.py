"""
Dynamic batch sizing optimizer for improved throughput and memory management.
"""

import logging
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Callable, Optional

import psutil


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    initial_batch_size: int = 10
    min_batch_size: int = 1
    max_batch_size: int = 100
    memory_limit_gb: float = 8.0
    target_processing_time: float = 2.0  # Target time per batch in seconds
    adaptation_factor: float = 0.2  # How much to adjust batch size
    memory_threshold: float = 0.85  # Memory usage threshold (85%)


class BatchOptimizer:
    """
    Dynamic batch size optimizer that adapts batch sizes based on:
    - Available memory
    - Processing time per item
    - System resource usage
    """

    def __init__(self, config: Optional[BatchConfig] = None):
        """
        Initialize batch optimizer.

        Args:
            config: Batch configuration
        """
        self.config = config or BatchConfig()
        self.logger = logging.getLogger(__name__)

        # Performance tracking
        self.current_batch_size = self.config.initial_batch_size
        self.processing_history = []
        self.memory_history = []
        self.lock = threading.Lock()

        # Statistics
        self.stats = {
            'total_batches': 0,
            'total_items': 0,
            'total_time': 0,
            'avg_batch_size': 0,
            'adaptations': 0,
        }

    def create_batches(self, items: list[Any]) -> Iterator[list[Any]]:
        """
        Create optimally-sized batches from a list of items.

        Args:
            items: List of items to batch

        Yields:
            Batches of items with optimal size
        """
        if not items:
            return

        total_items = len(items)
        processed = 0

        while processed < total_items:
            # Calculate current batch size
            current_batch_size = self._calculate_batch_size()

            # Extract batch
            end_idx = min(processed + current_batch_size, total_items)
            batch = items[processed:end_idx]

            self.logger.debug(f"Created batch of size {len(batch)} (total processed: {processed}/{total_items})")

            yield batch
            processed = end_idx

    def process_with_adaptive_batching(
        self,
        items: list[Any],
        processor_func: Callable,
        **kwargs,
    ) -> list[Any]:
        """
        Process items with adaptive batching for optimal performance.

        Args:
            items: Items to process
            processor_func: Function to process each batch
            **kwargs: Additional arguments for processor function

        Returns:
            List of processed results
        """
        if not items:
            return []

        all_results = []

        for batch in self.create_batches(items):
            batch_start_time = time.time()
            memory_before = self._get_memory_usage()

            try:
                # Process batch
                batch_results = processor_func(batch, **kwargs)

                # Track performance
                batch_time = time.time() - batch_start_time
                memory_after = self._get_memory_usage()

                self._record_batch_performance(
                    batch_size=len(batch),
                    processing_time=batch_time,
                    memory_before=memory_before,
                    memory_after=memory_after,
                )

                all_results.extend(batch_results if batch_results else [])

                # Adapt batch size based on performance
                self._adapt_batch_size(batch_time, memory_after)

            except Exception as e:
                self.logger.error(f"Batch processing failed: {e}")
                # Reduce batch size on failure
                self._reduce_batch_size()
                continue

        return all_results

    def _calculate_batch_size(self) -> int:
        """Calculate optimal batch size based on current conditions."""
        with self.lock:
            # Start with current batch size
            batch_size = self.current_batch_size

            # Check memory constraints
            memory_usage = self._get_memory_usage()
            if memory_usage > self.config.memory_threshold:
                # Reduce batch size if memory is high
                batch_size = max(
                    self.config.min_batch_size,
                    int(batch_size * 0.7),
                )
                self.logger.info(f"Reduced batch size to {batch_size} due to high memory usage ({memory_usage:.1%})")

            # Ensure batch size is within bounds
            batch_size = max(self.config.min_batch_size, min(self.config.max_batch_size, batch_size))

            return batch_size

    def _adapt_batch_size(self, processing_time: float, memory_usage: float):
        """Adapt batch size based on processing performance."""
        with self.lock:
            current_size = self.current_batch_size

            # Calculate target batch size based on processing time
            if processing_time > 0:
                time_ratio = self.config.target_processing_time / processing_time

                if time_ratio > 1.2:  # Processing is too fast, can increase batch size
                    new_size = int(current_size * (1 + self.config.adaptation_factor))
                elif time_ratio < 0.8:  # Processing is too slow, reduce batch size
                    new_size = int(current_size * (1 - self.config.adaptation_factor))
                else:
                    new_size = current_size  # No change needed
            else:
                new_size = current_size

            # Memory-based adjustment
            if memory_usage > self.config.memory_threshold:
                new_size = int(new_size * 0.8)  # Reduce more aggressively for memory
            elif memory_usage < 0.5:  # Low memory usage, can increase
                new_size = int(new_size * 1.1)

            # Apply bounds
            new_size = max(self.config.min_batch_size, min(self.config.max_batch_size, new_size))

            if new_size != current_size:
                self.logger.debug(f"Adapted batch size: {current_size} -> {new_size} (time: {processing_time:.2f}s, memory: {memory_usage:.1%})")
                self.current_batch_size = new_size
                self.stats['adaptations'] += 1

    def _reduce_batch_size(self):
        """Reduce batch size due to failure."""
        with self.lock:
            old_size = self.current_batch_size
            self.current_batch_size = max(
                self.config.min_batch_size,
                int(self.current_batch_size * 0.5),
            )
            self.logger.warning(f"Reduced batch size due to failure: {old_size} -> {self.current_batch_size}")
            self.stats['adaptations'] += 1

    def _record_batch_performance(
        self,
        batch_size: int,
        processing_time: float,
        memory_before: float,
        memory_after: float,
    ):
        """Record batch performance metrics."""
        with self.lock:
            # Update statistics
            self.stats['total_batches'] += 1
            self.stats['total_items'] += batch_size
            self.stats['total_time'] += processing_time

            if self.stats['total_batches'] > 0:
                self.stats['avg_batch_size'] = self.stats['total_items'] / self.stats['total_batches']

            # Keep rolling history (last 10 batches)
            self.processing_history.append({
                'batch_size': batch_size,
                'processing_time': processing_time,
                'time_per_item': processing_time / max(1, batch_size),
                'memory_delta': memory_after - memory_before,
            })

            if len(self.processing_history) > 10:
                self.processing_history.pop(0)

            self.memory_history.append(memory_after)
            if len(self.memory_history) > 20:
                self.memory_history.pop(0)

    def _get_memory_usage(self) -> float:
        """Get current memory usage as a percentage."""
        try:
            return psutil.virtual_memory().percent / 100.0
        except Exception:
            return 0.5  # Default fallback

    def get_optimal_batch_size_estimate(self, _item_count: int) -> int:
        """
        Estimate optimal batch size for a given number of items.

        Args:
            item_count: Total number of items to process

        Returns:
            Estimated optimal batch size
        """
        if not self.processing_history:
            return self.config.initial_batch_size

        # Calculate average time per item from history
        avg_time_per_item = sum(h['time_per_item'] for h in self.processing_history) / len(self.processing_history)

        if avg_time_per_item > 0:
            # Estimate batch size for target processing time
            estimated_size = int(self.config.target_processing_time / avg_time_per_item)
        else:
            estimated_size = self.current_batch_size

        # Apply memory constraints
        memory_usage = self._get_memory_usage()
        if memory_usage > 0.7:
            estimated_size = int(estimated_size * (1.0 - memory_usage))

        # Apply bounds
        return max(self.config.min_batch_size, min(self.config.max_batch_size, estimated_size))

    def get_performance_report(self) -> dict:
        """Get performance statistics and recommendations."""
        with self.lock:
            if self.stats['total_batches'] == 0:
                return {"error": "No batches processed yet"}

            avg_processing_time = self.stats['total_time'] / self.stats['total_batches']
            avg_memory = sum(self.memory_history) / len(self.memory_history) if self.memory_history else 0

            report = {
                'total_batches_processed': self.stats['total_batches'],
                'total_items_processed': self.stats['total_items'],
                'average_batch_size': self.stats['avg_batch_size'],
                'current_batch_size': self.current_batch_size,
                'total_processing_time': self.stats['total_time'],
                'average_time_per_batch': avg_processing_time,
                'average_memory_usage': avg_memory,
                'total_adaptations': self.stats['adaptations'],
                'efficiency_score': self._calculate_efficiency_score(),
            }

            # Add recommendations
            report['recommendations'] = self._generate_recommendations()

            return report

    def _calculate_efficiency_score(self) -> float:
        """Calculate efficiency score (0-1) based on performance metrics."""
        if not self.processing_history:
            return 0.5

        # Factors for efficiency:
        # 1. How close processing time is to target
        # 2. Memory usage efficiency
        # 3. Batch size stability (fewer adaptations = better)

        time_efficiency = 0.5
        if self.processing_history:
            avg_time = sum(h['processing_time'] for h in self.processing_history) / len(self.processing_history)
            time_diff = abs(avg_time - self.config.target_processing_time)
            time_efficiency = max(0, 1.0 - (time_diff / self.config.target_processing_time))

        memory_efficiency = 0.5
        if self.memory_history:
            avg_memory = sum(self.memory_history) / len(self.memory_history)
            # Optimal memory usage is around 60-70%
            if 0.6 <= avg_memory <= 0.7:
                memory_efficiency = 1.0
            else:
                memory_efficiency = max(0, 1.0 - abs(avg_memory - 0.65) / 0.35)

        stability_efficiency = 1.0
        if self.stats['total_batches'] > 0:
            adaptation_rate = self.stats['adaptations'] / self.stats['total_batches']
            stability_efficiency = max(0, 1.0 - adaptation_rate)

        return (time_efficiency + memory_efficiency + stability_efficiency) / 3.0

    def _generate_recommendations(self) -> list[str]:
        """Generate performance recommendations."""
        recommendations = []

        if not self.processing_history:
            return ["Process more batches to generate recommendations"]

        avg_time = sum(h['processing_time'] for h in self.processing_history) / len(self.processing_history)
        avg_memory = sum(self.memory_history) / len(self.memory_history) if self.memory_history else 0

        if avg_time > self.config.target_processing_time * 1.5:
            recommendations.append("Consider reducing batch size - processing time is above target")
        elif avg_time < self.config.target_processing_time * 0.5:
            recommendations.append("Consider increasing batch size - processing time is well below target")

        if avg_memory > 0.8:
            recommendations.append("High memory usage detected - consider reducing batch size or adding memory cleanup")
        elif avg_memory < 0.3:
            recommendations.append("Low memory usage - you can likely increase batch size for better throughput")

        if self.stats['adaptations'] > self.stats['total_batches'] * 0.5:
            recommendations.append("High adaptation rate - consider adjusting initial batch size or target processing time")

        return recommendations if recommendations else ["Performance is optimal"]

    def reset_stats(self):
        """Reset all statistics and history."""
        with self.lock:
            self.current_batch_size = self.config.initial_batch_size
            self.processing_history.clear()
            self.memory_history.clear()
            self.stats = {
                'total_batches': 0,
                'total_items': 0,
                'total_time': 0,
                'avg_batch_size': 0,
                'adaptations': 0,
            }


# Convenience functions
def create_adaptive_batches(
    items: list[Any],
    initial_batch_size: int = 10,
    max_batch_size: int = 100,
    memory_limit_gb: float = 8.0,
) -> Iterator[list[Any]]:
    """Create adaptive batches with simple configuration."""
    config = BatchConfig(
        initial_batch_size=initial_batch_size,
        max_batch_size=max_batch_size,
        memory_limit_gb=memory_limit_gb,
    )
    optimizer = BatchOptimizer(config)
    return optimizer.create_batches(items)


def process_with_optimal_batching(
    items: list[Any],
    processor_func: Callable,
    target_time: float = 2.0,
    max_batch_size: int = 50,
) -> list[Any]:
    """Process items with automatic batch optimization."""
    config = BatchConfig(
        target_processing_time=target_time,
        max_batch_size=max_batch_size,
    )
    optimizer = BatchOptimizer(config)
    return optimizer.process_with_adaptive_batching(items, processor_func)

"""
Comprehensive resource monitoring and memory management for pipeline operations.
"""

import gc
import logging
import threading
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import psutil

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False


@dataclass
class ResourceMetrics:
    """Resource usage metrics."""
    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_available_gb: float = 0.0
    memory_used_gb: float = 0.0
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    gpu_utilization: float = 0.0
    process_memory_mb: float = 0.0
    open_file_descriptors: int = 0


@dataclass
class MonitoringConfig:
    """Configuration for resource monitoring."""
    enable_monitoring: bool = True
    monitoring_interval: float = 5.0  # seconds
    memory_cleanup_threshold: float = 85.0  # percentage
    aggressive_cleanup_threshold: float = 90.0  # percentage
    log_interval: float = 30.0  # seconds
    enable_gpu_monitoring: bool = True
    enable_process_monitoring: bool = True
    history_size: int = 100


class ResourceMonitor:
    """
    Comprehensive resource monitor with automatic memory cleanup and alerting.
    """

    def __init__(self, config: Optional[MonitoringConfig] = None):
        """
        Initialize resource monitor.

        Args:
            config: Monitoring configuration
        """
        self.config = config or MonitoringConfig()
        self.logger = logging.getLogger(__name__)

        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.metrics_history: list[ResourceMetrics] = []
        self.lock = threading.Lock()

        # Process tracking
        self.process = psutil.Process()
        self.initial_memory = self._get_current_metrics().memory_used_gb

        # Cleanup callbacks
        self.cleanup_callbacks: list[Callable] = []

        # Statistics
        self.stats = {
            'cleanup_count': 0,
            'peak_memory_gb': 0.0,
            'peak_gpu_memory_gb': 0.0,
            'avg_cpu_percent': 0.0,
            'monitoring_duration': 0.0,
        }

        # Initialize monitoring if enabled
        if self.config.enable_monitoring:
            self.start_monitoring()

    def start_monitoring(self):
        """Start background resource monitoring."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Resource monitoring started")

    def stop_monitoring(self):
        """Stop background resource monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        self.logger.info("Resource monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        last_log_time = 0
        start_time = time.time()

        while self.is_monitoring:
            try:
                current_time = time.time()
                metrics = self._get_current_metrics()

                with self.lock:
                    self.metrics_history.append(metrics)
                    if len(self.metrics_history) > self.config.history_size:
                        self.metrics_history.pop(0)

                    # Update stats
                    self.stats['peak_memory_gb'] = max(self.stats['peak_memory_gb'], metrics.memory_used_gb)
                    self.stats['peak_gpu_memory_gb'] = max(self.stats['peak_gpu_memory_gb'], metrics.gpu_memory_used_gb)
                    self.stats['monitoring_duration'] = current_time - start_time

                # Check for cleanup triggers
                self._check_cleanup_triggers(metrics)

                # Periodic logging
                if current_time - last_log_time >= self.config.log_interval:
                    self._log_resource_status(metrics)
                    last_log_time = current_time

                time.sleep(self.config.monitoring_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1.0)

    def _get_current_metrics(self) -> ResourceMetrics:
        """Get current system resource metrics."""
        metrics = ResourceMetrics()

        try:
            # System metrics
            metrics.cpu_percent = psutil.cpu_percent(interval=0.1)

            memory = psutil.virtual_memory()
            metrics.memory_percent = memory.percent
            metrics.memory_available_gb = memory.available / (1024**3)
            metrics.memory_used_gb = memory.used / (1024**3)

            # Process metrics
            if self.config.enable_process_monitoring:
                try:
                    process_memory = self.process.memory_info()
                    metrics.process_memory_mb = process_memory.rss / (1024**2)
                    metrics.open_file_descriptors = self.process.num_fds() if hasattr(self.process, 'num_fds') else 0
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            # GPU metrics
            if self.config.enable_gpu_monitoring and TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    metrics.gpu_memory_used_gb = torch.cuda.memory_allocated() / (1024**3)
                    metrics.gpu_memory_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

                    if GPUTIL_AVAILABLE:
                        gpus = GPUtil.getGPUs()
                        if gpus:
                            metrics.gpu_utilization = gpus[0].load * 100
                except Exception:
                    pass  # GPU monitoring is optional

        except Exception as e:
            self.logger.warning(f"Failed to collect metrics: {e}")

        return metrics

    def _check_cleanup_triggers(self, metrics: ResourceMetrics):
        """Check if cleanup should be triggered based on resource usage."""
        memory_percent = metrics.memory_percent

        if memory_percent >= self.config.aggressive_cleanup_threshold:
            self.logger.warning(f"Critical memory usage: {memory_percent:.1f}%. Triggering aggressive cleanup.")
            self.cleanup_memory(aggressive=True)
        elif memory_percent >= self.config.memory_cleanup_threshold:
            self.logger.info(f"High memory usage: {memory_percent:.1f}%. Triggering cleanup.")
            self.cleanup_memory(aggressive=False)

    def cleanup_memory(self, aggressive: bool = False):
        """
        Perform memory cleanup operations.

        Args:
            aggressive: Whether to perform aggressive cleanup
        """
        start_time = time.time()
        initial_memory = psutil.virtual_memory().percent

        try:
            # Standard cleanup
            gc.collect()

            # GPU cleanup
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                if aggressive:
                    torch.cuda.synchronize()

            # Call registered cleanup callbacks
            for callback in self.cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    self.logger.warning(f"Cleanup callback failed: {e}")

            # Aggressive cleanup
            if aggressive:
                # Multiple garbage collection passes
                for _ in range(3):
                    gc.collect()

                # Clear warnings cache
                warnings.resetwarnings()

                # Try to reduce memory fragmentation
                if hasattr(gc, 'set_threshold'):
                    gc.set_threshold(100, 10, 10)  # More aggressive GC

            # Update statistics
            final_memory = psutil.virtual_memory().percent
            cleanup_time = time.time() - start_time
            memory_freed = initial_memory - final_memory

            with self.lock:
                self.stats['cleanup_count'] += 1

            self.logger.info(
                f"Memory cleanup completed in {cleanup_time:.2f}s. "
                f"Memory usage: {initial_memory:.1f}% -> {final_memory:.1f}% "
                f"(freed {memory_freed:.1f}%)",
            )

        except Exception as e:
            self.logger.error(f"Memory cleanup failed: {e}")

    def register_cleanup_callback(self, callback: Callable):
        """Register a cleanup callback function."""
        self.cleanup_callbacks.append(callback)

    def _log_resource_status(self, metrics: ResourceMetrics):
        """Log current resource status."""
        log_msg = (
            f"Resources: CPU {metrics.cpu_percent:.1f}%, "
            f"RAM {metrics.memory_percent:.1f}% ({metrics.memory_used_gb:.1f}GB), "
            f"Process {metrics.process_memory_mb:.0f}MB"
        )

        if metrics.gpu_memory_used_gb > 0:
            log_msg += f", GPU {metrics.gpu_memory_used_gb:.1f}GB/{metrics.gpu_memory_total_gb:.1f}GB"
            if metrics.gpu_utilization > 0:
                log_msg += f" ({metrics.gpu_utilization:.1f}%)"

        if metrics.open_file_descriptors > 0:
            log_msg += f", FDs {metrics.open_file_descriptors}"

        self.logger.info(log_msg)

    def get_current_status(self) -> dict[str, Any]:
        """Get current resource status."""
        metrics = self._get_current_metrics()

        with self.lock:
            avg_cpu = sum(m.cpu_percent for m in self.metrics_history[-10:]) / min(10, len(self.metrics_history)) if self.metrics_history else 0

        return {
            'timestamp': metrics.timestamp,
            'cpu_percent': metrics.cpu_percent,
            'memory_percent': metrics.memory_percent,
            'memory_used_gb': metrics.memory_used_gb,
            'memory_available_gb': metrics.memory_available_gb,
            'gpu_memory_used_gb': metrics.gpu_memory_used_gb,
            'gpu_utilization': metrics.gpu_utilization,
            'process_memory_mb': metrics.process_memory_mb,
            'average_cpu_10_samples': avg_cpu,
            'cleanup_count': self.stats['cleanup_count'],
            'peak_memory_gb': self.stats['peak_memory_gb'],
            'monitoring_duration': self.stats['monitoring_duration'],
        }

    def get_memory_report(self) -> dict[str, Any]:
        """Get detailed memory usage report."""
        current = self._get_current_metrics()

        with self.lock:
            history_count = len(self.metrics_history)
            if history_count > 0:
                avg_memory = sum(m.memory_percent for m in self.metrics_history) / history_count
                max_memory = max(m.memory_percent for m in self.metrics_history)
                min_memory = min(m.memory_percent for m in self.metrics_history)
            else:
                avg_memory = max_memory = min_memory = current.memory_percent

        return {
            'current_memory_percent': current.memory_percent,
            'current_memory_gb': current.memory_used_gb,
            'available_memory_gb': current.memory_available_gb,
            'average_memory_percent': avg_memory,
            'peak_memory_percent': max_memory,
            'minimum_memory_percent': min_memory,
            'memory_growth_gb': current.memory_used_gb - self.initial_memory,
            'cleanup_triggered_count': self.stats['cleanup_count'],
            'process_memory_mb': current.process_memory_mb,
            'recommendations': self._generate_memory_recommendations(current),
        }

    def _generate_memory_recommendations(self, metrics: ResourceMetrics) -> list[str]:
        """Generate memory optimization recommendations."""
        recommendations = []

        if metrics.memory_percent > 85:
            recommendations.append("Critical: Memory usage is very high. Consider reducing batch sizes or adding more cleanup.")
        elif metrics.memory_percent > 70:
            recommendations.append("Warning: Memory usage is high. Monitor for potential memory leaks.")

        if metrics.memory_used_gb - self.initial_memory > 5:
            recommendations.append("Significant memory growth detected. Check for memory leaks in processing logic.")

        if self.stats['cleanup_count'] > 10:
            recommendations.append("Frequent cleanup triggers suggest memory management issues. Consider optimizing data structures.")

        if metrics.gpu_memory_used_gb > metrics.gpu_memory_total_gb * 0.9:
            recommendations.append("GPU memory is nearly full. Consider reducing model batch sizes or enabling gradient checkpointing.")

        if not recommendations:
            recommendations.append("Memory usage appears optimal.")

        return recommendations

    @contextmanager
    def monitor_operation(self, operation_name: str):
        """Context manager to monitor a specific operation."""
        start_time = time.time()
        start_metrics = self._get_current_metrics()

        self.logger.info(f"Starting monitored operation: {operation_name}")

        try:
            yield
        finally:
            end_time = time.time()
            end_metrics = self._get_current_metrics()
            duration = end_time - start_time

            memory_delta = end_metrics.memory_used_gb - start_metrics.memory_used_gb
            gpu_memory_delta = end_metrics.gpu_memory_used_gb - start_metrics.gpu_memory_used_gb

            self.logger.info(
                f"Operation '{operation_name}' completed in {duration:.2f}s. "
                f"Memory delta: {memory_delta:+.2f}GB"
                + (f", GPU delta: {gpu_memory_delta:+.2f}GB" if gpu_memory_delta != 0 else ""),
            )

            # Trigger cleanup if significant memory increase
            if memory_delta > 1.0 or end_metrics.memory_percent > 80:
                self.cleanup_memory()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()

        # Final cleanup
        if self.config.enable_monitoring:
            self.cleanup_memory(aggressive=True)


# Global monitor instance
_global_monitor: Optional[ResourceMonitor] = None


def get_global_monitor() -> ResourceMonitor:
    """Get or create global resource monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = ResourceMonitor()
    return _global_monitor


def cleanup_memory_global(aggressive: bool = False):
    """Convenience function for global memory cleanup."""
    monitor = get_global_monitor()
    monitor.cleanup_memory(aggressive=aggressive)


def monitor_resources_context(operation_name: str):
    """Convenience context manager for monitoring operations."""
    monitor = get_global_monitor()
    return monitor.monitor_operation(operation_name)

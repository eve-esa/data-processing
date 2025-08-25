"""
Model lifecycle management for efficient memory usage and model reuse.
"""

import gc
import logging
import threading
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelManager:
    """
    Centralized model management with proper lifecycle and memory management.
    Implements singleton pattern to ensure single instance across application.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, '_initialized'):
            return

        self._models: dict[str, Any] = {}
        self._tokenizers: dict[str, Any] = {}
        self._model_lock = threading.RLock()
        self._last_access: dict[str, float] = {}
        self._cleanup_interval = 300  # 5 minutes
        self._max_idle_time = 600  # 10 minutes
        self.logger = logging.getLogger(__name__)

        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._periodic_cleanup, daemon=True)
        self._cleanup_thread.start()

        self._initialized = True

    def get_model(self, model_id: str, model_type: str = "causal_lm", **kwargs) -> tuple:
        """
        Get model and tokenizer, loading if necessary.

        Args:
            model_id: HuggingFace model identifier
            model_type: Type of model (causal_lm, sequence_classification, etc.)
            **kwargs: Additional arguments for model loading

        Returns:
            Tuple of (model, tokenizer)
        """
        with self._model_lock:
            cache_key = f"{model_id}_{model_type}"

            # Update access time
            self._last_access[cache_key] = time.time()

            # Return cached model if available
            if cache_key in self._models:
                self.logger.debug(f"Using cached model: {model_id}")
                return self._models[cache_key], self._tokenizers[cache_key]

            # Load new model
            self.logger.info(f"Loading new model: {model_id}")
            model, tokenizer = self._load_model(model_id, model_type, **kwargs)

            # Cache the model
            self._models[cache_key] = model
            self._tokenizers[cache_key] = tokenizer

            return model, tokenizer

    def _load_model(self, model_id: str, model_type: str, **kwargs):
        """Load model and tokenizer with optimized settings."""
        try:
            # Default loading arguments for memory efficiency
            default_args = {
                'torch_dtype': torch.float16,
                'device_map': 'auto',
                'trust_remote_code': True,
                'low_cpu_mem_usage': True,
            }
            default_args.update(kwargs)

            # Load model based on type
            if model_type == "causal_lm":
                model = AutoModelForCausalLM.from_pretrained(model_id, **default_args)
            else:
                # Add other model types as needed
                from transformers import AutoModel
                model = AutoModel.from_pretrained(model_id, **default_args)

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
            )

            # Set pad token if not exists
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Move to eval mode for inference
            model.eval()

            # Warmup with small computation
            self._warmup_model(model, tokenizer)

            return model, tokenizer

        except Exception as e:
            self.logger.error(f"Failed to load model {model_id}: {e}")
            raise

    def _warmup_model(self, model, tokenizer):
        """Warmup model with a small computation to initialize CUDA kernels."""
        try:
            dummy_text = "This is a warmup sentence."
            inputs = tokenizer(dummy_text, return_tensors="pt")

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                _ = model(**inputs)

            # Clear warmup from memory
            del inputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            self.logger.warning(f"Model warmup failed: {e}")

    def _periodic_cleanup(self):
        """Periodically clean up unused models."""
        while True:
            try:
                time.sleep(self._cleanup_interval)
                self._cleanup_idle_models()
            except Exception as e:
                self.logger.error(f"Error in periodic cleanup: {e}")

    def _cleanup_idle_models(self):
        """Remove models that haven't been accessed recently."""
        current_time = time.time()
        models_to_remove = []

        with self._model_lock:
            for cache_key, last_access in self._last_access.items():
                if current_time - last_access > self._max_idle_time:
                    models_to_remove.append(cache_key)

            for cache_key in models_to_remove:
                self.logger.info(f"Cleaning up idle model: {cache_key}")
                self._remove_model(cache_key)

    def _remove_model(self, cache_key: str):
        """Remove a model from cache and free memory."""
        if cache_key in self._models:
            del self._models[cache_key]
        if cache_key in self._tokenizers:
            del self._tokenizers[cache_key]
        if cache_key in self._last_access:
            del self._last_access[cache_key]

        # Force garbage collection and clear CUDA cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def cleanup_all(self):
        """Clean up all cached models."""
        with self._model_lock:
            cache_keys = list(self._models.keys())
            for cache_key in cache_keys:
                self._remove_model(cache_key)

            self.logger.info("All models cleaned up")

    def get_memory_usage(self) -> dict[str, float]:
        """Get current memory usage statistics."""
        stats = {
            'cached_models': len(self._models),
            'ram_usage_gb': 0,
            'gpu_usage_gb': 0,
        }

        try:
            import psutil
            stats['ram_usage_gb'] = psutil.virtual_memory().used / (1024**3)
        except ImportError:
            pass

        if torch.cuda.is_available():
            stats['gpu_usage_gb'] = torch.cuda.memory_allocated() / (1024**3)

        return stats

    def __del__(self):
        """Cleanup on destruction."""
        self.cleanup_all()


# Global instance
model_manager = ModelManager()

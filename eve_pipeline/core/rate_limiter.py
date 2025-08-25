"""Rate limiting utilities for API calls, especially OpenAI."""

import asyncio
import random
import time
from typing import Any, Callable

from eve_pipeline.core.logging import LoggerManager
from eve_pipeline.core.prompts import OpenAIConfig
from eve_pipeline.core.utils import RetryUtils


class RateLimiter:
    """Rate limiter for API calls with exponential backoff."""

    def __init__(
        self,
        requests_per_minute: int = OpenAIConfig.DEFAULT_REQUESTS_PER_MINUTE,
        tokens_per_minute: int = OpenAIConfig.DEFAULT_TOKENS_PER_MINUTE,
        max_retries: int = OpenAIConfig.DEFAULT_MAX_RETRIES,
        min_wait: float = OpenAIConfig.DEFAULT_MIN_WAIT,
        max_wait: float = OpenAIConfig.DEFAULT_MAX_WAIT,
    ) -> None:
        """Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute.
            tokens_per_minute: Maximum tokens per minute.
            max_retries: Maximum number of retry attempts.
            min_wait: Minimum wait time in seconds.
            max_wait: Maximum wait time in seconds.
        """
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.max_retries = max_retries
        self.min_wait = min_wait
        self.max_wait = max_wait

        self.request_times: list[float] = []
        self.token_usage: list[tuple[float, int]] = []

        self.logger = LoggerManager.get_logger("RateLimiter")

    def _clean_old_records(self) -> None:
        """Remove records older than 1 minute."""
        current_time = time.time()
        cutoff_time = current_time - 60

        self.request_times = [t for t in self.request_times if t > cutoff_time]

        self.token_usage = [(t, tokens) for t, tokens in self.token_usage if t > cutoff_time]

    def _should_wait_for_requests(self) -> bool:
        """Check if we should wait due to request rate limit."""
        self._clean_old_records()
        return len(self.request_times) >= self.requests_per_minute

    def _should_wait_for_tokens(self, estimated_tokens: int = 0) -> bool:
        """Check if we should wait due to token rate limit."""
        self._clean_old_records()
        current_tokens = sum(tokens for _, tokens in self.token_usage)
        return (current_tokens + estimated_tokens) >= self.tokens_per_minute

    def _calculate_wait_time(self) -> float:
        """Calculate how long to wait before next request."""
        if not self.request_times and not self.token_usage:
            return 0.0

        current_time = time.time()

        oldest_request = min(self.request_times) if self.request_times else current_time
        oldest_token = min(t for t, _ in self.token_usage) if self.token_usage else current_time
        oldest_time = min(oldest_request, oldest_token)

        wait_time = 60 - (current_time - oldest_time)
        return max(0, wait_time)

    async def wait_if_needed(self, estimated_tokens: int = 0) -> None:
        """Wait if rate limits would be exceeded."""
        if self._should_wait_for_requests() or self._should_wait_for_tokens(estimated_tokens):
            wait_time = self._calculate_wait_time()
            if wait_time > 0:
                self.logger.info(f"Rate limit approached, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)

    def record_request(self, tokens_used: int = 0) -> None:
        """Record a completed request."""
        current_time = time.time()
        self.request_times.append(current_time)
        if tokens_used > 0:
            self.token_usage.append((current_time, tokens_used))

    async def execute_with_backoff(
        self,
        func: Callable,
        *args,
        estimated_tokens: int = 0,
        **kwargs,
    ) -> Any:
        """Execute function with rate limiting and exponential backoff.

        Args:
            func: Function to execute.
            *args: Function arguments.
            estimated_tokens: Estimated token usage for the request.
            **kwargs: Function keyword arguments.

        Returns:
            Function result.

        Raises:
            Exception: If all retry attempts fail.
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                await self.wait_if_needed(estimated_tokens)

                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                self.record_request(estimated_tokens)
                return result

            except Exception as e:
                last_exception = e

                if self._is_rate_limit_error(e):
                    wait_time = RetryUtils.exponential_backoff(
                        attempt, self.min_wait, self.max_wait,
                    )
                    jitter = random.uniform(0, 0.1) * wait_time
                    total_wait = wait_time + jitter

                    self.logger.warning(
                        f"Rate limit hit (attempt {attempt + 1}/{self.max_retries}), "
                        f"waiting {total_wait:.2f} seconds",
                    )
                    await asyncio.sleep(total_wait)
                    continue

                elif self._should_retry_error(e):
                    if RetryUtils.should_retry(attempt, self.max_retries):
                        wait_time = RetryUtils.exponential_backoff(attempt)
                        self.logger.warning(
                            f"Retryable error (attempt {attempt + 1}/{self.max_retries}): {e}, "
                            f"waiting {wait_time:.2f} seconds",
                        )
                        await asyncio.sleep(wait_time)
                        continue

                raise e

        self.logger.error(f"All {self.max_retries} retry attempts failed")
        raise last_exception or Exception("Unknown error occurred")

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if error is a rate limit error."""
        error_str = str(error).lower()
        error_indicators = [
            "rate limit",
            "too many requests",
            "429",
            "quota exceeded",
            "rate_limit_exceeded",
        ]
        return any(indicator in error_str for indicator in error_indicators)

    def _should_retry_error(self, error: Exception) -> bool:
        """Check if error should be retried."""
        error_str = str(error).lower()
        retryable_indicators = [
            "timeout",
            "connection",
            "network",
            "502",
            "503",
            "504",
            "internal server error",
            "bad gateway",
            "service unavailable",
            "gateway timeout",
        ]
        return any(indicator in error_str for indicator in retryable_indicators)

    def get_stats(self) -> dict[str, Any]:
        """Get rate limiter statistics."""
        self._clean_old_records()
        current_tokens = sum(tokens for _, tokens in self.token_usage)

        return {
            "requests_in_last_minute": len(self.request_times),
            "tokens_in_last_minute": current_tokens,
            "requests_limit": self.requests_per_minute,
            "tokens_limit": self.tokens_per_minute,
            "requests_remaining": max(0, self.requests_per_minute - len(self.request_times)),
            "tokens_remaining": max(0, self.tokens_per_minute - current_tokens),
        }


class OpenAIRateLimiter(RateLimiter):
    """Specialized rate limiter for OpenAI-compatible API calls (OpenRouter, OpenAI, etc.)."""

    def __init__(self, **kwargs) -> None:
        """Initialize OpenAI-compatible rate limiter with appropriate defaults."""
        super().__init__(
            requests_per_minute=kwargs.get('requests_per_minute', 500),
            tokens_per_minute=kwargs.get('tokens_per_minute', 200000),
            **kwargs,
        )

    def estimate_tokens(self, text: str, _model: str = "gpt-4o-mini") -> int:
        """Estimate token count for text.

        Args:
            text: Input text.
            model: Model name for token estimation.

        Returns:
            Estimated token count.
        """
        return len(text) // 3

    async def make_openai_request(
        self,
        client_method: Callable,
        messages: list,
        model: str = OpenAIConfig.DEFAULT_MODEL,
        **api_kwargs,
    ) -> Any:
        """Make an OpenAI-compatible API request with rate limiting.

        Args:
            client_method: OpenAI-compatible client method to call.
            messages: Chat messages.
            model: Model to use.
            **api_kwargs: Additional API arguments.

        Returns:
            API response.
        """
        total_text = " ".join(
            msg.get("content", "") for msg in messages if isinstance(msg.get("content"), str)
        )
        estimated_tokens = self.estimate_tokens(total_text, model)

        estimated_tokens = int(estimated_tokens * 1.5)

        return await self.execute_with_backoff(
            client_method,
            messages=messages,
            model=model,
            estimated_tokens=estimated_tokens,
            **api_kwargs,
        )

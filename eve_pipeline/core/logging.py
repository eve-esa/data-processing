"""Centralized logging configuration for eve-pipeline."""

import logging
import sys
from pathlib import Path
from typing import ClassVar, Optional, Union

from eve_pipeline.core.enums import LogLevel


class LoggerManager:
    """Centralized logger manager for consistent logging across the pipeline."""

    _loggers: ClassVar = {}
    _configured: ClassVar = False

    @classmethod
    def setup_logging(
        cls,
        level: Union[str, LogLevel] = LogLevel.INFO,
        log_file: Optional[Union[str, Path]] = None,
        format_string: Optional[str] = None,
        force: bool = False,
    ) -> None:
        """Set up logging configuration for the entire pipeline.

        Args:
            level: Logging level.
            log_file: Optional log file path.
            format_string: Custom format string.
            force: Force reconfiguration even if already configured.
        """
        if cls._configured and not force:
            return

        if isinstance(level, LogLevel):
            level = level.value

        if format_string is None:
            format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format=format_string,
            force=force,
            handlers=[],
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_formatter = logging.Formatter(format_string)
        console_handler.setFormatter(console_formatter)

        root_logger = logging.getLogger()
        root_logger.addHandler(console_handler)

        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, level.upper()))
            file_formatter = logging.Formatter(format_string)
            file_handler.setFormatter(file_formatter)

            root_logger.addHandler(file_handler)

        cls._configured = True

    @classmethod
    def get_logger(
        cls,
        name: str,
        level: Optional[Union[str, LogLevel]] = None,
    ) -> logging.Logger:
        """Get a logger instance with consistent configuration.

        Args:
            name: Logger name (typically module name).
            level: Optional specific level for this logger.

        Returns:
            Configured logger instance.
        """
        if name in cls._loggers:
            return cls._loggers[name]

        if not cls._configured:
            cls.setup_logging()

        logger = logging.getLogger(f"eve_pipeline.{name}")

        if level:
            if isinstance(level, LogLevel):
                level = level.value
            logger.setLevel(getattr(logging, level.upper()))

        cls._loggers[name] = logger
        return logger

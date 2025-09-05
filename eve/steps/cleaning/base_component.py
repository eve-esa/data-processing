"""Base class for data processing components."""

from abc import ABC, abstractmethod
from typing import Optional

from eve.logging import get_logger


class DataProcessingComponent(ABC):
    """Abstract base class for all data processing components."""

    def __init__(self, debug: bool = False):
        """Initialize the data processing component.
        
        Args:
            debug: Enable debug output.
        """
        self.debug = debug
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    async def process(self, content: str, filename: str) -> Optional[str]:
        """Process the content and return the cleaned result.
        
        Args:
            content: The text content to process.
            filename: Name of the file being processed.
            
        Returns:
            Processed content or None if processing fails.
        """
        pass

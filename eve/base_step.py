from abc import ABC, abstractmethod
from typing import Any, Optional

from eve.logging import get_logger


class PipelineStep(ABC):
    """abstract base class for all pipeline steps."""

    def __init__(self, config: Any, name: Optional[str] = None):
        """initialize the pipeline step.

        Args:
            config: Configuration specific to the step.
            output_dir: Directory where output files will be saved.
            name: Optional name for the step (used for logging).
        """
        self.config = config
        self.logger = get_logger(name or self.__class__.__name__)

    @abstractmethod
    async def execute(self, input_data: Any) -> Any: # TBD
        """Execute the pipeline step.

        Args:
            input_data: Input data to process.

        Returns:
            Processed data or result of the step.
        """
        pass

    async def __call__(self, input_data: Any) -> Any:
        """shortway of calling `execute` method.

        Args:
            input_data: Input data to process.

        Returns:
            Processed data or result of the step.
        """
        return await self.execute(input_data)

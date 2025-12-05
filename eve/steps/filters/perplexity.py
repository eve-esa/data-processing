"""Perplexity-based document filtering step."""

from typing import Any, List

from eve.base_step import PipelineStep
from transformers import AutoTokenizer, AutoModelForCausalLM

from eve.model.document import Document
from eve.steps.filters.compute_ppl import perplexity


class PerplexityFilterStep(PipelineStep):
    """Filter documents based on perplexity scores calculated by a language model.

    This step computes perplexity scores for documents using a causal language model
    and optionally filters them based on a threshold. Lower perplexity indicates more
    natural, coherent text.

    Config parameters:
        model_name (str): Hugging Face model name (default: "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        stride (int): Stride for sliding window perplexity calculation (default: 128)
        batch_size (int): Batch size for model inference (default: 128)
        max_length (int): Maximum sequence length for model (default: 1024)
        threshold (float): Perplexity threshold for filtering (default: 0.0)
        enable_threshold (bool): Whether to apply threshold-based filtering (default: False)

    Examples:
        # Calculate perplexity without filtering
        config: {model_name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0", enable_threshold: false}

        # Keep only documents with perplexity below 50
        config: {threshold: 50.0, enable_threshold: true}
    """

    def __init__(self, config: dict):
        """Initialize the perplexity filter step.

        Args:
            config: Configuration dictionary containing model and filtering parameters
        """
        super().__init__(config, name="PerplexityFilterStep")

        self.tokenizer = None
        self.model = None
        self.model_name = config.get("model_name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.stride = config.get("stride", 128)
        self.batch_size = config.get("batch_size", 128)
        self.max_length = config.get("max_length", 1024)
        self.threshold = config.get("threshold", 0.0)

        # Enable filtering on threshold
        self.enable_filter = config.get("enable_threshold", False)

        self.init_model_tokenizer()

    def init_model_tokenizer(self):
        """Initialize the language model and tokenizer.

        Loads the model from Hugging Face and sets up the tokenizer with
        appropriate padding configuration.
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", torch_dtype="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    async def execute(self, documents: List[Document]) -> List[Document]:
        """Execute perplexity calculation and optional filtering on documents.

        Computes perplexity scores for all documents and adds them to metadata.
        If enable_threshold is True, filters documents based on the perplexity threshold.

        Args:
            documents: List of documents to process

        Returns:
            List of documents (filtered if enable_threshold is True, otherwise all documents
            with perplexity scores in metadata)
        """
        try:
            for doc in documents:
                ppl = perplexity(
                    [doc.content],
                    self.model,
                    self.tokenizer,
                    self.stride,
                    self.batch_size,
                    self.max_length,
                )
                doc.metadata["perplexity"] = ppl
        except Exception as e:
            self.logger.warning(
                f"Failed computing ppl for {doc.filename}, exception {e}"
            )

        if self.enable_filter:
            documents = [
                doc for doc in documents if doc.metadata["perplexity"] <= self.threshold
            ]

        return documents

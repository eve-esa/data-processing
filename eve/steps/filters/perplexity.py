from typing import Any, List

from eve.base_step import PipelineStep
from transformers import AutoTokenizer, AutoModelForCausalLM

from eve.model.document import Document
from eve.steps.filters.compute_ppl import perplexity


class PerplexityFilterStep(PipelineStep):

    def __init__(self, config: dict):
        super().__init__(config, name="ChunkerStep")

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
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="auto", torch_dtype="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    async def execute(self, documents: List[Document]) -> Any:
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

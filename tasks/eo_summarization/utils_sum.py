# Import centralized LLM judge utilities
from metrics.judge_utils import process_summary_results, aggregate_llm_judge


def process_results(doc: dict, results: list[str]) -> dict:
    """Process summarization results with multi-dimensional LLM judge evaluation."""
    return process_summary_results(doc=doc, results=results, input_key="input", output_key="output")


def aggregate_mean_score(items) -> float:
    """Aggregate mean scores for summarization metrics."""
    return aggregate_llm_judge(items)

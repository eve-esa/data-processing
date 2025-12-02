# Import centralized LLM judge utilities
from metrics.judge_utils import LoggableFuture, process_qa_results, aggregate_llm_judge


def process_results(doc: dict, results: list[str]) -> dict:
    """
    Process results for refusal questions.
    Auto-detects field names (question/answer vs Question/Answer).
    """
    # Auto-detect field names
    if "Question" in doc:
        question_key = "Question"
        answer_key = "Answer"
    elif "question" in doc:
        question_key = "question"
        answer_key = "answer"
    else:
        # Fallback: try to find any key containing 'question'
        question_keys = [k for k in doc.keys() if "question" in k.lower()]
        answer_keys = [k for k in doc.keys() if "answer" in k.lower()]
        question_key = question_keys[0] if question_keys else "question"
        answer_key = answer_keys[0] if answer_keys else "answer"

    return process_qa_results(
        doc=doc,
        results=results,
        question_key=question_key,
        answer_key=answer_key,
        sleep_time=2.0,  # Stagger requests to avoid rate limits
    )


def doc_to_text(doc: dict) -> str:
    """Converts a document dictionary to a text representation."""
    context = doc["Context"] + "\n"

    return f"Context: {context}\n\nQuestion: {doc['Question']}\n"

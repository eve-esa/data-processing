import re
import sys
from pathlib import Path

import datasets

# Add parent directory to path to import common MCQA utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from mcqa_utils import extract_labels


def doc_to_text(doc):
    """
    Format the document into a hallucination detection prompt.

    Expected doc format:
    - 'question': The original question
    - 'answer': The answer to evaluate for hallucination
    """
    question = doc.get("Question", "")
    answer = doc.get("Answer", "")

    string = f"Question: {question}\n"
    string += f"Answer: {answer}\n\n"
    string += "Is this answer hallucinated (contains false or unsupported information)?\n"
    string += "A. Yes\n"
    string += "B. No\n"

    return string


def process_answer(answer):
    """
    Process model answer to extract the choice (A or B).

    Returns a list containing the extracted label.
    """
    labels = extract_labels(answer)

    # If no valid label found, default to empty list
    if not labels:
        return []

    # Only take the first label if multiple are returned
    return [labels[0]] if labels else []


def map_label_to_letter(is_hallucinated):
    """
    Map boolean or string label to letter choice.

    Args:
        is_hallucinated: Can be bool, int, or string indicating if answer is hallucinated

    Returns:
        str: 'A' if hallucinated, 'B' if not
    """
    # Handle various formats
    if isinstance(is_hallucinated, bool):
        return "A" if is_hallucinated else "B"
    elif isinstance(is_hallucinated, int):
        return "A" if is_hallucinated == 1 else "B"
    elif isinstance(is_hallucinated, str):
        # Handle string representations
        lower = is_hallucinated.lower().strip()
        if lower in ["true", "yes", "1", "a", "hallucinated"]:
            return "A"
        else:
            return "B"
    else:
        return "B"  # Default to not hallucinated


def process_results(doc: datasets.Dataset, results):
    """
    Process results for hallucination detection task.

    Args:
        doc: Document containing ground truth label
        results: Model predictions

    Returns:
        dict: Per-example metrics (accuracy, TP, FP, FN, TN, and prediction tuple)
    """
    print(f"Results: {results}")
    # Get model prediction
    pred_text = results[0]
    pred_labels = process_answer(pred_text)
    print(f"Predicted labels: {pred_labels}")

    # Extract the predicted label (first one if multiple, or default to "B")
    pred_label = pred_labels[0] if pred_labels else "B"

    # Get ground truth label
    # The label in doc should indicate if the answer is hallucinated
    # Map it to letter format (A = hallucinated, B = not hallucinated)
    ground_truth_label = doc.get("label", doc.get("is_hallucinated", True))
    reference = map_label_to_letter(ground_truth_label)

    # Convert to binary for sklearn metrics (A=1, B=0)
    reference_binary = 1 if reference == "A" else 0
    pred_binary = 1 if pred_label == "A" else 0

    # Calculate per-example accuracy (1.0 if correct, 0.0 if incorrect)
    acc = 1.0 if pred_label == reference else 0.0

    # Track TP, FP, FN, TN for precision, recall, and F1 calculation
    # A = hallucinated (positive class), B = not hallucinated (negative class)
    tp = 1.0 if reference == "A" and pred_label == "A" else 0.0  # True Positive
    fp = 1.0 if reference == "B" and pred_label == "A" else 0.0  # False Positive
    fn = 1.0 if reference == "A" and pred_label == "B" else 0.0  # False Negative
    tn = 1.0 if reference == "B" and pred_label == "B" else 0.0  # True Negative

    return {
        "acc": acc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": (reference_binary, pred_binary),
        "recall": (reference_binary, pred_binary),
        "f1": (reference_binary, pred_binary),
    }


def aggregate_predictions_precision(items):
    """
    Aggregate precision from prediction tuples.

    Args:
        items: List of (reference, prediction) tuples

    Returns:
        float: Precision score
    """
    if not items:
        return 0.0

    references, predictions = zip(*items)

    tp = sum(1 for ref, pred in zip(references, predictions) if ref == 1 and pred == 1)
    fp = sum(1 for ref, pred in zip(references, predictions) if ref == 0 and pred == 1)

    if tp + fp == 0:
        return 0.0

    return tp / (tp + fp)


def aggregate_predictions_recall(items):
    """
    Aggregate recall from prediction tuples.

    Args:
        items: List of (reference, prediction) tuples

    Returns:
        float: Recall score
    """
    if not items:
        return 0.0

    references, predictions = zip(*items)

    tp = sum(1 for ref, pred in zip(references, predictions) if ref == 1 and pred == 1)
    fn = sum(1 for ref, pred in zip(references, predictions) if ref == 1 and pred == 0)

    if tp + fn == 0:
        return 0.0

    return tp / (tp + fn)


def aggregate_predictions_f1(items):
    """
    Aggregate F1 score from prediction tuples.

    Args:
        items: List of (reference, prediction) tuples

    Returns:
        float: F1 score
    """
    if not items:
        return 0.0

    references, predictions = zip(*items)

    tp = sum(1 for ref, pred in zip(references, predictions) if ref == 1 and pred == 1)
    fp = sum(1 for ref, pred in zip(references, predictions) if ref == 0 and pred == 1)
    fn = sum(1 for ref, pred in zip(references, predictions) if ref == 1 and pred == 0)

    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)

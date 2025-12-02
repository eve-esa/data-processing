import re
import sys
from pathlib import Path

import datasets

# Add parent directory to path to import common MCQA utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from mcqa_utils import extract_labels


def accuracy(references, predictions):
    """
    Calculate accuracy for single-answer MCQA.

    Args:
        references: List of single correct answer letters (e.g., ['A', 'B', 'C'])
        predictions: List of predicted answer letters (e.g., ['A', 'B', 'D'])

    Returns:
        float: Accuracy score
    """
    correct_count = 0
    for correct, pred in zip(references, predictions):
        # For single answer, take first prediction if multiple provided
        pred_first = pred[0] if isinstance(pred, list) and pred else pred
        if pred_first == correct:
            correct_count += 1
    return correct_count / len(references) if references else 0


def doc_to_text(doc):
    """
    Format document into MCQA prompt.

    Expected doc format:
    - 'question': The question text
    - 'choices': List of answer options (A, B, C, D)
    """
    question = doc.get("Question", "")
    choices = doc.get("Choices", [])

    # Build the question with choices
    string = f"{question}\n"

    labels = ["A", "B", "C", "D", "E"]
    for i, choice in enumerate(choices[:5]):  # Limit to 4 choices
        string += f"{labels[i]}. {choice}\n"

    return string


def process_answer(answer):
    """
    Process model answer to extract letter choice.

    This is a wrapper around the common extract_labels function from mcqa_utils.

    Handles multiple formats:
    - Simple letter: "A", "B"
    - Letter with period: "D.", "A."
    - Letter with full text: "D. Sea ice forms on ocean water..."
    - With prefix: "Answer: C. SIRAL"

    For single answer MCQA, only the first letter is used if multiple are extracted.
    """
    labels = extract_labels(answer)
    # Return first label only for single-answer MCQA
    return labels[0] if labels else ""


def process_results(doc: datasets.Dataset, results):
    """
    Process results for single-answer MCQA task.

    Args:
        doc: Document containing ground truth answer
        results: Model predictions

    Returns:
        dict: Metrics (accuracy)
    """
    # Get model prediction
    pred_text = results[0]
    pred_label = process_answer(pred_text)

    # Get ground truth answer
    reference = doc.get("Answer", "")

    # Calculate accuracy
    acc = 1.0 if pred_label == reference else 0.0

    return {"acc": acc}

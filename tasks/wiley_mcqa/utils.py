"""
Wiley MCQA utilities for formatting multiple choice questions.

Note: Label extraction for this task is handled by the regex pattern in the YAML
configuration file (_wiley_mcqa_template). The regex pattern is defined in the
common mcqa_utils module (tasks/mcqa_utils.py) as LABEL_EXTRACTION_REGEX.

If you need to modify label extraction logic, update the mcqa_utils module
and the regex pattern in the YAML file accordingly.
"""

import sys
from pathlib import Path

# Add parent directory to path to import common MCQA utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from mcqa_utils import extract_label, LABEL_EXTRACTION_REGEX  # noqa: F401


def format_multiple_choice(question, choices):
    """
    Format a multiple choice question with dynamic letter labels.

    Args:
        question (str): The question text
        choices (list): List of answer choices

    Returns:
        str: Formatted multiple choice question
    """
    # Strip whitespace from question
    question = question.strip()

    # Generate letter labels (A, B, C, D, E, F, etc.)
    letters = [chr(65 + i) for i in range(len(choices))]

    # Build the formatted string
    formatted_choices = "\n".join(f"{letter}. {choice}" for letter, choice in zip(letters, choices))

    return f"{question}\n{formatted_choices}\n:"


def doc_to_multiple_choice(doc):
    """
    Convert a document containing questions and choices to formatted multiple choice questions.

    Args:
        doc (dict or list): Document with question(s) and choices.
                           Expected format: {"question": str, "choices": list}
                           Or a list of such dictionaries

    Returns:
        str or list: Formatted multiple choice question(s)
    """
    # Handle single question
    if isinstance(doc, dict):
        return format_multiple_choice(doc["Question"], doc["choices"])

    # Handle multiple questions
    if isinstance(doc, list):
        return [format_multiple_choice(item["Question"], item["choices"]) for item in doc]

    raise ValueError("Document must be a dict or list of dicts")

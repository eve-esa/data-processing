"""
Common utilities for Multiple Choice Question Answering (MCQA) tasks.

This module provides shared functionality across all MCQA tasks including
label extraction, answer processing, and formatting.
"""

import re


def extract_label(answer: str) -> str:
    """
    Extract a single letter label from a model's answer.

    This function is designed to work with the lm-evaluation-harness regex filter.
    It extracts the first capital letter (A, B, C, D, etc.) from the answer,
    handling various formats including long reasoning followed by an answer.

    Handles formats like:
    - Simple letter: "A", "B"
    - Letter with period: "A.", "D."
    - Letter with text: "A. Some text here"
    - With prefix: "Answer: C"
    - Bold markdown: "**B**", "**A**\n\nExplanation...", "**A. 4.65**", "**(B)**"
    - Italic markdown: "*B*", "*(A)*"
    - Parentheses: "(B)", "(A)"
    - With reasoning: "...long reasoning... the answer is **(B)**."

    Args:
        answer: The model's answer string

    Returns:
        The extracted letter label (e.g., "A", "B", "C", "D")
        Returns empty string if no valid label is found.

    Examples:
        >>> extract_label("A")
        'A'
        >>> extract_label("Answer: B. Some text")
        'B'
        >>> extract_label("D.")
        'D'
        >>> extract_label("**B**")
        'B'
        >>> extract_label("**C**\\n\\nHere's why...")
        'C'
        >>> extract_label("...reasoning... the answer is **(B)**.")
        'B'
    """
    # First, try to find "answer is" pattern anywhere in the text
    # This handles: "the answer is (B)", "answer is **B**", etc.
    answer_is_match = re.search(
        r"(?:the\s+)?answer\s+is\s+[*_()]*([A-Z])[*_().\s]*", answer, flags=re.IGNORECASE
    )
    if answer_is_match:
        return answer_is_match.group(1)

    # Strip common prefixes at the beginning like "Answer: ", "answer: ", etc.
    answer = re.sub(
        r"^(?:Answer|answer|The answer is|the answer is):\s*", "", answer, flags=re.IGNORECASE
    )

    # Try to match markdown/formatted patterns: **(A)**, **B**, *(C)*, (D), etc.
    # This regex strips common markdown and finds the letter inside
    formatted_match = re.search(r"^\s*[*_]*\(?\s*([A-Z])\s*\)?[*_]*(?:\.|[^A-Z]|$)", answer)
    if formatted_match:
        return formatted_match.group(1)

    # Extract first capital letter followed by optional period, space, or end
    match = re.search(r"^\s*([A-Z])(?:\.|[\s,]|$)", answer)
    if match:
        return match.group(1)

    return ""


def extract_labels(answer: str) -> list[str]:
    """
    Extract multiple letter labels from a model's answer.

    This function handles cases where multiple correct answers are expected.
    It extracts all capital letters (A, B, C, D, etc.) found in the answer,
    including from answers with reasoning followed by "answer is" statements.

    Handles multiple formats:
    - Simple letters: "A", "B, C", "A,B,C"
    - Letters with periods: "A. B.", "D."
    - Letters with text: "A. text, B. more text"
    - With prefix: "Answer: A, C"
    - Bold markdown: "**B**", "**A**, **C**", "**A. 4.65**", "**(A), (C)**"
    - Italic markdown: "*A*, *B*", "*(D)*"
    - With reasoning: "...reasoning... the answers are **(A)** and **(C)**."

    Args:
        answer: The model's answer string

    Returns:
        List of extracted letter labels, preserving order and removing duplicates.
        Returns empty list if no valid labels are found.

    Examples:
        >>> extract_labels("A, B, C")
        ['A', 'B', 'C']
        >>> extract_labels("Answer: A. Text here, B. More text")
        ['A', 'B']
        >>> extract_labels("D")
        ['D']
        >>> extract_labels("**B**")
        ['B']
        >>> extract_labels("**A**, **C**")
        ['A', 'C']
        >>> extract_labels("...reasoning... the answers are **(A)** and **(C)**.")
        ['A', 'C']
    """
    # First, try to find "answer is/are" pattern and extract from that portion
    answer_is_match = re.search(
        r"(?:the\s+)?answers?\s+(?:is|are)\s+(.+?)(?:\.|$)", answer, flags=re.IGNORECASE | re.DOTALL
    )
    if answer_is_match:
        # Extract from the "answer is/are" portion only
        answer = answer_is_match.group(1)

    # Strip common prefixes at the beginning like "Answer: ", "answer: ", etc.
    answer = re.sub(
        r"^(?:Answer|answer|The answer is|the answer is):\s*", "", answer, flags=re.IGNORECASE
    )

    # First, try to find bold markdown labels: **A**, **B**, **(A)**, **A. 4.65**, etc.
    # Pattern handles optional parentheses and other characters between ** markers
    bold_matches = re.findall(r"\*\*[^A-Z]*([A-Z])[^*]*\*\*", answer)
    if bold_matches:
        # Remove duplicates while preserving order
        seen = set()
        labels_list = []
        for letter in bold_matches:
            if letter not in seen:
                seen.add(letter)
                labels_list.append(letter)
        return labels_list

    # Find all single capital letters followed by optional period
    # This pattern matches: "A", "A.", "D.", etc.
    letter_pattern = r"\b([A-Z])\.?(?:\s|,|and|&|$)"
    matches = re.findall(letter_pattern, answer)

    if matches:
        # Remove duplicates while preserving order
        seen = set()
        labels_list = []
        for letter in matches:
            if letter not in seen:
                seen.add(letter)
                labels_list.append(letter)
        return labels_list

    # Fallback: split by comma and strip
    labels_list = answer.split(",")
    labels_list = [label.strip() for label in labels_list]
    return labels_list


# Regex pattern for use in YAML configuration files
# This pattern extracts a single capital letter label
# Handles multiple formats including reasoning followed by "answer is" statements
# Formats: (A, A., Answer: A), bold markdown (**A**, **A. 4.65**, **(B)**),
# and "answer is" patterns (the answer is **(B)**.)
LABEL_EXTRACTION_REGEX = r"(?:(?:the\s+)?answer\s+is\s+[*_()]*([A-Z])[*_().\s]*|^\s*(?:Answer:\s*)?[*_]*\(?\s*([A-Z])\s*\)?[*_]*(?:\.|[^A-Z]|$))"

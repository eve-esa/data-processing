"""Common regex patterns used across the pipeline."""

import re
from typing import Pattern


# LaTeX formula patterns
INLINE_MATH_PATTERN: Pattern[str] = re.compile(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)')
DISPLAY_MATH_PATTERN: Pattern[str] = re.compile(r'\$\$(.*?)\$\$', re.DOTALL)
BRACKET_MATH_PATTERN: Pattern[str] = re.compile(r'\\[(](.*?)\\[)]', re.DOTALL)
SQUARE_BRACKET_MATH_PATTERN: Pattern[str] = re.compile(r'\\\[(.*?)\\\]', re.DOTALL)
LATEX_ENV_PATTERN: Pattern[str] = re.compile(r'\\begin\{([^}]+)\}(.*?)\\end\{\1\}', re.DOTALL)

# OCR correction patterns
DIGIT_LETTER_PATTERN: Pattern[str] = re.compile(r'(\d+)([A-Za-z]{2,})')

# Nougat artifact patterns
WARNING_PATTERN: Pattern[str] = re.compile(
    r'\+\+\+\s*==WARNING: Truncated because of repetitions==.*?\+\+\+',
    re.DOTALL
)
ERROR_PATTERN: Pattern[str] = re.compile(
    r'\+\+\+\s*==ERROR: No output for this page==.*?\+\+\+',
    re.DOTALL
)

# Text cleaning patterns
EXCESSIVE_NEWLINES_PATTERN: Pattern[str] = re.compile(r'\n{3,}')
SINGLE_SYMBOL_LINE_PATTERN: Pattern[str] = re.compile(r'^\s*[^\w\s]\s*$', re.MULTILINE)
REFERENCE_PATTERN: Pattern[str] = re.compile(r"^\* \[\d+\]", flags=re.MULTILINE)

# PII detection patterns
EMAIL_PATTERN: Pattern[str] = re.compile(r'([-a-zA-Z0-9.`?{}]+@\w+(?:\.\w+)+)')

# LaTeX table cleaning patterns
DOUBLED_BACKSLASH_PATTERN: Pattern[str] = re.compile(r'\\{2,}')


def get_latex_formula_patterns() -> dict[str, Pattern[str]]:
    """
    Get all LaTeX formula patterns in a dictionary.
    
    Returns:
        Dictionary mapping pattern names to compiled regex patterns
    """
    return {
        'inline': INLINE_MATH_PATTERN,
        'display': DISPLAY_MATH_PATTERN,
        'bracket': BRACKET_MATH_PATTERN,
        'square_bracket': SQUARE_BRACKET_MATH_PATTERN,
        'environment': LATEX_ENV_PATTERN
    }


def clean_doubled_backslashes(text: str) -> str:
    """Clean up doubled backslashes in LaTeX content."""
    return DOUBLED_BACKSLASH_PATTERN.sub(lambda m: '\\' * (len(m.group()) // 2), text)


def normalize_excessive_newlines(text: str) -> str:
    """Replace 3+ consecutive newlines with exactly 2."""
    return EXCESSIVE_NEWLINES_PATTERN.sub('\n\n', text)


def remove_single_symbol_lines(text: str) -> str:
    """Remove lines that contain only a single symbol or punctuation."""
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        stripped = line.strip()
        if re.search(r'\w', stripped) or len(stripped) != 1:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def fix_ocr_digit_letter_spacing(text: str) -> str:
    """Fix OCR issues where digits are concatenated with letters."""
    return DIGIT_LETTER_PATTERN.sub(r'\1 \2', text)


def remove_nougat_artifacts(text: str) -> str:
    """Remove Nougat-specific warning and error artifacts."""
    text = WARNING_PATTERN.sub('', text)
    text = ERROR_PATTERN.sub('', text)
    text = text.replace('[MISSING_PAGE_POST]', '')
    return text

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

# LaTeX table cleaning patterns
DOUBLED_BACKSLASH_PATTERN: Pattern[str] = re.compile(r'\\{2,}')

# HTML metadata extraction patterns
HTML_TITLE_PATTERN: Pattern[str] = re.compile(r'<title[^>]*>(.*?)</title>', re.IGNORECASE | re.DOTALL)
HTML_TAG_PATTERN: Pattern[str] = re.compile(r'<[^>]+>')
HTML_ENTITY_PATTERN: Pattern[str] = re.compile(r'&[a-zA-Z]+;')
HTML_NUMERIC_ENTITY_PATTERN: Pattern[str] = re.compile(r'&#\d+;')
JSON_LD_SCRIPT_PATTERN: Pattern[str] = re.compile(r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', re.IGNORECASE | re.DOTALL)

doi_regexp = [
    r'doi[\s\.\:]{0,2}(10\.\d{4}[\d\:\.\-\/a-z]+)(?:[\s\n\"<]|$)',
    r'(10\.\d{4}[\d\:\.\-\/a-z]+)(?:[\s\n\"<]|$)',
    r'(10\.\d{4}[\:\.\-\/a-z]+[\:\.\-\d]+)(?:[\s\na-z\"<]|$)',
    r'https?://[ -~]*doi[ -~]*/(10\.\d{4,9}/[-._;()/:a-z0-9]+)(?:[\s\n\"<]|$)',
    r'^(10\.\d{4,9}/[-._;()/:a-z0-9]+)$'
]
arxiv_regexp = [
    r'arxiv[\s]*\:[\s]*(\d{4}\.\d+)(?:v\d+)?(?:[\s\n\"<]|$)',
    r'(\d{4}\.\d+)(?:v\d+)?(?:\.pdf)',
    r'^(\d{4}\.\d+)(?:v\d+)?$'
]
isbn_regexp = [
    r'(?:ISBN(?:-1[03])?:? )?(?=[0-9X]{10}$|(?=(?:[0-9]+[- ]){3})[- 0-9X]{13}$|97[89][0-9]{10}$)(?:97[89][- ]?)?[0-9]{1,5}[- ]?[0-9]+[- ]?[0-9]+[- ]?[0-9X]',
    r'\bISBN(?:-1[03])?:? (?=[0-9X]{10}$|(?=(?:[0-9]+[- ]){3})[- 0-9X]{13}$|97[89][0-9]{10}$)(?:97[89][- ]?)?[0-9]{1,5}[- ]?[0-9]+[- ]?[0-9]+[- ]?[0-9X]\b'
]


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


def extract_html_title(html_content: str) -> str:
    """
    Extract title from HTML content.
    
    Args:
        html_content: HTML content as string
        
    Returns:
        Extracted and cleaned title, or None if not found
    """
    if not html_content:
        return None
        
    title_match = HTML_TITLE_PATTERN.search(html_content)
    
    if title_match:
        title = title_match.group(1)
        
        title = HTML_TAG_PATTERN.sub('', title)
        title = HTML_ENTITY_PATTERN.sub(' ', title)
        title = HTML_NUMERIC_ENTITY_PATTERN.sub(' ', title)
        
        return title.strip()
    
    return None


def extract_html_meta_tags(html_content: str) -> dict[str, str]:
    """
    Extract metadata from HTML meta tags.
    
    Args:
        html_content: HTML content as string
        
    Returns:
        Dictionary containing extracted meta tag information
    """
    meta_data = {}
    
    if not html_content:
        return meta_data

    meta_patterns = {
        'description': r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']*)["\']',
        'keywords': r'<meta[^>]*name=["\']keywords["\'][^>]*content=["\']([^"\']*)["\']',
        'author': r'<meta[^>]*name=["\']author["\'][^>]*content=["\']([^"\']*)["\']',
        'og_title': r'<meta[^>]*property=["\']og:title["\'][^>]*content=["\']([^"\']*)["\']',
        'og_description': r'<meta[^>]*property=["\']og:description["\'][^>]*content=["\']([^"\']*)["\']',
        'twitter_title': r'<meta[^>]*name=["\']twitter:title["\'][^>]*content=["\']([^"\']*)["\']',
    }

    for key, pattern in meta_patterns.items():
        match = re.search(pattern, html_content, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if value:
                meta_data[key] = value

    return meta_data


def extract_json_ld_count(html_content: str) -> int:
    """
    Count JSON-LD structured data blocks in HTML.
    
    Args:
        html_content: HTML content as string
        
    Returns:
        Number of JSON-LD script blocks found
    """
    if not html_content:
        return 0
        
    json_ld_matches = JSON_LD_SCRIPT_PATTERN.findall(html_content)
    return len(json_ld_matches)

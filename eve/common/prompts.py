"""Common prompts used across the pipeline."""


LATEX_CORRECTION_PROMPT = """Please correct the following LaTeX formula that has a syntax error:

Formula type: {formula_type}
Error message: {error_message}
Formula: {formula}

Context (first 1000 chars): {context_snippet}

Please provide ONLY the corrected LaTeX formula without any explanations or surrounding text. Keep the mathematical meaning intact while fixing the syntax errors."""


def get_latex_correction_prompt(
    formula_type: str,
    error_message: str,
    formula: str,
    context: str
) -> str:
    """
    Generate a LaTeX correction prompt.
    
    Args:
        formula_type: Type of LaTeX formula (inline, display, etc.)
        error_message: The error message from LaTeX compilation
        formula: The problematic formula
        context: Surrounding context for better understanding
        
    Returns:
        Formatted prompt string
    """
    context_snippet = context[:1000] + "..." if len(context) > 1000 else context
    
    return LATEX_CORRECTION_PROMPT.format(
        formula_type=formula_type,
        error_message=error_message,
        formula=formula,
        context_snippet=context_snippet
    )

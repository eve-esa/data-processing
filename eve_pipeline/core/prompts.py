"""Prompts and templates for AI-powered processing."""

class LatexCorrectionPrompts:
    """Prompts for LaTeX correction using AI."""

    SYSTEM_PROMPT = (
        "You are an expert LaTeX mathematician specializing in fixing syntax errors in scientific formulas. "
        "Your corrections must preserve mathematical meaning while ensuring LaTeX compilation success. "
        "You have extensive knowledge of common LaTeX errors from OCR extraction, web scraping, "
        "encoding issues, and missing packages. "
        "Return ONLY the corrected formula - no explanations, no markdown formatting, no surrounding text."
    )

    USER_PROMPT_TEMPLATE = """You are correcting a LaTeX formula extracted from a scientific document.

SURROUNDING CONTEXT:
{context}

FORMULA DETAILS:
- Type: {formula_type}
- Error: {error_message}
- Original: {original_formula}

Please provide ONLY the corrected LaTeX formula without any explanation."""

    @classmethod
    def format_user_prompt(
        cls,
        formula: str,
        error_message: str,
        formula_type: str,
        context: str = "",
    ) -> str:
        """Format the user prompt with specific values."""
        return cls.USER_PROMPT_TEMPLATE.format(
            context=context,
            formula_type=formula_type,
            error_message=error_message,
            original_formula=formula,
        )


class OpenAIConfig:
    """Configuration for OpenAI-compatible API calls (OpenRouter, OpenAI, etc.)."""

    # OpenRouter configuration
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL = "openai/gpt-4o-mini"  # OpenRouter model format
    DEFAULT_TEMPERATURE = 0.1
    DEFAULT_MAX_TOKENS = 4000
    DEFAULT_TIMEOUT = 30

    # Rate limiting configuration (adjusted for OpenRouter)
    DEFAULT_REQUESTS_PER_MINUTE = 500
    DEFAULT_TOKENS_PER_MINUTE = 200000

    # Retry configuration
    DEFAULT_MAX_RETRIES = 6
    DEFAULT_MIN_WAIT = 1
    DEFAULT_MAX_WAIT = 60

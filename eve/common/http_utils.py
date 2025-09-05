"""Common HTTP utilities for making API calls across the pipeline."""

import aiohttp
from typing import Dict, Any, Optional
from eve.logging import get_logger

logger = get_logger(__name__)


async def post_request(
    url: str,
    headers: Dict[str, str],
    data: Dict[str, Any],
    timeout: int = 30
) -> Optional[Dict[str, Any]]:
    """
    Make an async POST request and return JSON response.
    
    Args:
        url: The URL to make the request to
        headers: Request headers
        data: Request data to send as JSON
        timeout: Request timeout in seconds
        
    Returns:
        Response JSON data or None if request failed
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, 
                headers=headers, 
                json=data,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"HTTP request failed with status {response.status}")
                    return None
    except Exception as e:
        logger.error(f"HTTP request failed: {str(e)}")
        return None


async def make_openrouter_request(
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int = 1000,
    temperature: float = 0.1
) -> Optional[str]:
    """
    Make a request to OpenRouter API for LLM completion.
    
    Args:
        api_key: OpenRouter API key
        model: Model name to use
        prompt: The prompt to send
        max_tokens: Maximum tokens in response
        temperature: Temperature for response generation
        
    Returns:
        Response content or None if request failed
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    response = await post_request(url, headers, data)
    if response and "choices" in response:
        content = response["choices"][0]["message"]["content"].strip()
        import re
        content = re.sub(r'^```latex\n?', '', content)
        content = re.sub(r'\n?```$', '', content)
        return content.strip()
    
    return None

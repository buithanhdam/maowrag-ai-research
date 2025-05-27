import logging
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_chain,
    wait_fixed,
    before_sleep_log,
)
from google.api_core.exceptions import GoogleAPICallError
from src.logger import get_formatted_logger

def retry_on_transient_error():
    def is_transient_error(exception: Exception) -> bool:
        if isinstance(exception, GoogleAPICallError):
            # Retry for 429, 500, 502, 503, 504
            return exception.code.value in {429, 500, 502, 503, 504}
        return any(code in str(exception) for code in ["429", "500", "502", "503", "504"])
    
    wait_strategy = wait_chain(wait_fixed(3), wait_fixed(5), wait_fixed(10))
    retry_condition = retry_if_exception(is_transient_error)

    return retry(
        retry=retry_condition,
        stop=stop_after_attempt(3),
        wait=wait_strategy,
        before_sleep=before_sleep_log(get_formatted_logger(__file__), logging.WARNING),
        reraise=True,
    )

class LLMResponse(BaseModel):
    """LLM response with token usage information."""
    content: str
    prompt_token_count: int = 0
    query_token_count: int = 0
    request_token_count: int = 0
    response_token_count: int = 0
    total_token_count: int = 0
    cached_content_token_count: int = 0
    
    def __str__(self) -> str:
        return self.content
    
    def __repr__(self) -> str:
        return f"LLMResponse(content='{self.content[:50]}...', total_tokens={self.total_token_count})"
    
    @property
    def usage_metadata(self) -> dict:
        return {
            "prompt_token_count": self.prompt_token_count,
            "query_token_count": self.query_token_count,
            "request_token_count": self.request_token_count,
            "response_token_count": self.response_token_count,
            "total_token_count": self.total_token_count,
            "cached_content_token_count": self.cached_content_token_count,
        }

class StreamChunk(BaseModel):
    """Streaming chunk with just content - no token calculation."""
    content: str
    is_final: bool = False
    
    def __str__(self) -> str:
        return self.content

class TokenUsage(BaseModel):
    """Token usage information extracted from API response."""
    prompt_tokens: int = 0
    response_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
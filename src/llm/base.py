from contextlib import asynccontextmanager
from llama_index.core.types import PydanticProgramMode
from typing import AsyncGenerator, Generator, List, Optional
from llama_index.core.llms import ChatMessage, ChatResponse
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
import asyncio
from llama_index.core.llms.function_calling import FunctionCallingLLM
import tiktoken
# from llama_index.llms.anthropic import Anthropic
from llama_index.llms.gemini import Gemini
from google.api_core.exceptions import GoogleAPICallError
from llama_index.llms.openai import OpenAI

from src.config import LLMProviderType, get_llm_config
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

class BaseLLM:
    def __init__(
        self,
        api_key: str = None,
        provider: LLMProviderType = LLMProviderType.GOOGLE,
        model_id: str = None,
        temperature: float = None,
        max_tokens: int = None,
        system_prompt: str = None,
    ):
        LLM_CONFIG = get_llm_config(provider)
        self.api_key = api_key or LLM_CONFIG.api_key
        self.provider = provider or LLM_CONFIG.provider
        self.model_id = model_id or LLM_CONFIG.model_id
        self.temperature = temperature or LLM_CONFIG.temperature
        self.max_tokens = max_tokens or LLM_CONFIG.max_tokens
        self.system_prompt = system_prompt or LLM_CONFIG.system_prompt
        self.token_encoding_model = tiktoken.get_encoding("cl100k_base")
        self.logger = get_formatted_logger(__file__)
        self._initialize_model()

    def _set_system_prompt(self, system_prompt: str) -> None:
        self.system_prompt = (
            f"Base prompt: {self.system_prompt}\nUser prompt: {system_prompt}"
        )

    def _get_system_prompt(self) -> str:
        return self.system_prompt

    def _initialize_model(self) -> None:
        try:
            if self.provider == LLMProviderType.GOOGLE:
                self.model = Gemini(
                    api_key=self.api_key,
                    model=self.model_id,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    pydantic_program_mode=PydanticProgramMode.OPENAI,
                )
            elif self.provider == LLMProviderType.OPENAI:
                self.model = OpenAI(
                    api_key=self.api_key,
                    model=self.model_id,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    pydantic_program_mode=PydanticProgramMode.OPENAI,
                )
            else:
                raise ValueError(f"Unsupported model type: {self.provider}")
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.provider} model: {str(e)}")
            raise

    def _get_model(self) -> FunctionCallingLLM:
        return self.model

    def _prepare_messages(
        self, query: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> List[ChatMessage]:
        messages = []
        if self.system_prompt:
            messages.append(ChatMessage(role="system", content=self.system_prompt))
            messages.append(
                ChatMessage(
                    role="assistant",
                    content="I understand and will follow these instructions.",
                )
            )

        if chat_history:
            messages.extend(chat_history)

        messages.append(ChatMessage(role="user", content=query))
        return messages

    def _count_tokens(self, text: str) -> int:
        """Count tokens in given text."""
        try:
            num_tokens = len(self.token_encoding_model.encode(text))
            return num_tokens
        except Exception as e:
            self.logger.warning(f"Error counting tokens for {self.provider}: {str(e)}")
            # Fallback estimation
            return len(text) // 4

    def _count_messages_tokens(self, messages: List[ChatMessage]) -> int:
        """Count tokens in list of messages."""
        total_tokens = 0
        for message in messages:
            total_tokens += self._count_tokens(message.content or "")
        return total_tokens

    def _extract_content(self, response: ChatResponse) -> str:
        """Extract only content from response - no token calculation."""
        try:
            if hasattr(response, 'message') and response.message:
                return response.message.content or ""
            elif hasattr(response, 'content'):
                return response.content or ""
            else:
                return str(response)
        except Exception as e:
            self.logger.warning(f"Error extracting content: {str(e)}")
            return str(response) if response else ""

    def _extract_token_usage(self, response: ChatResponse) -> TokenUsage:
        """Extract token usage information from API response."""
        try:
            prompt_tokens = 0
            response_tokens = 0
            total_tokens = 0
            cached_tokens = 0

            if hasattr(response, 'raw') and response.raw:
                # For Google Gemini
                if isinstance(response.raw, dict):
                    usage_metadata = response.raw.get("usage_metadata", {})
                    prompt_tokens = usage_metadata.get("prompt_token_count", 0)
                    response_tokens = usage_metadata.get("candidates_token_count", 0)
                    total_tokens = usage_metadata.get("total_token_count", 0)
                    cached_tokens = usage_metadata.get("cached_content_token_count", 0)
                
                # For OpenAI
                elif hasattr(response.raw, 'usage'):
                    usage = response.raw.usage
                    prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                    response_tokens = getattr(usage, 'completion_tokens', 0)
                    total_tokens = getattr(usage, 'total_tokens', 0)

            return TokenUsage(
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
                total_tokens=total_tokens,
                cached_tokens=cached_tokens
            )

        except Exception as e:
            self.logger.warning(f"Error extracting token usage: {str(e)}")
            return TokenUsage()

    def _calculate_fallback_tokens(self, content: str, query: str = "", chat_history: Optional[List[ChatMessage]] = None) -> TokenUsage:
        """Calculate tokens manually when API doesn't provide usage info."""
        query_tokens = self._count_tokens(query) if query else 0
        prompt_tokens = self._count_tokens(self.system_prompt) if self.system_prompt else 0
        chat_history_tokens = self._count_messages_tokens(chat_history) if chat_history else 0
        response_tokens = self._count_tokens(content)
        
        total_prompt_tokens = prompt_tokens + query_tokens + chat_history_tokens
        total_tokens = total_prompt_tokens + response_tokens
        
        return TokenUsage(
            prompt_tokens=total_prompt_tokens,
            response_tokens=response_tokens,
            total_tokens=total_tokens,
            cached_tokens=0
        )

    def _create_llm_response(self, content: str, token_usage: TokenUsage, query: str = "", chat_history: Optional[List[ChatMessage]] = None) -> LLMResponse:
        """Create final LLMResponse with content and token information."""
        # If API didn't provide token usage, calculate fallback
        if token_usage.total_tokens == 0:
            token_usage = self._calculate_fallback_tokens(content, query, chat_history)
        
        # Calculate query and request tokens for detailed breakdown
        query_tokens = self._count_tokens(query) if query else 0
        prompt_tokens = self._count_tokens(self.system_prompt) if self.system_prompt else 0

        return LLMResponse(
            content=content,
            prompt_token_count=prompt_tokens,
            query_token_count=query_tokens,
            request_token_count=token_usage.prompt_tokens,
            response_token_count=token_usage.response_tokens,
            total_token_count=token_usage.total_tokens,
            cached_content_token_count=token_usage.cached_tokens
        )

    @retry_on_transient_error()
    def chat(self, query: str, chat_history: Optional[List[ChatMessage]] = None) -> LLMResponse:
        try:
            messages = self._prepare_messages(query, chat_history)
            response = self.model.chat(messages)
            
            content = self._extract_content(response)
            token_usage = self._extract_token_usage(response)
            
            return self._create_llm_response(content, token_usage, query, chat_history)
        except Exception as e:
            self.logger.error(f"Error in {self.provider} chat: {str(e)}")
            raise e

    @retry_on_transient_error()
    async def achat(
        self, query: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> LLMResponse:
        try:
            messages = self._prepare_messages(query, chat_history)
            response = await self.model.achat(messages)
            
            content = self._extract_content(response)
            token_usage = self._extract_token_usage(response)
            
            return self._create_llm_response(content, token_usage, query, chat_history)
        except Exception as e:
            self.logger.error(f"Error in {self.provider} async chat: {str(e)}")
            raise e

    @retry_on_transient_error()
    def stream_chat(
        self, query: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> Generator[StreamChunk, None, None]:
        try:
            messages = self._prepare_messages(query, chat_history)
            response_stream = self.model.stream_chat(messages)
            
            for response in response_stream:
                content = self._extract_content(response)
                yield StreamChunk(content=content)
                
        except Exception as e:
            self.logger.error(f"Error in {self.provider} stream chat: {str(e)}")
            raise e

    @retry_on_transient_error()
    async def astream_chat(
        self, query: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AsyncGenerator[StreamChunk, None]:
        try:
            messages = self._prepare_messages(query, chat_history)
            response = await self.model.astream_chat(messages)

            if asyncio.iscoroutine(response):
                response = await response

            if hasattr(response, "__aiter__"):
                async for chunk in response:
                    content = self._extract_content(chunk)
                    yield StreamChunk(content=content)
            else:
                content = self._extract_content(response)
                yield StreamChunk(content=content, is_final=True)

        except Exception as e:
            self.logger.error(f"Error in {self.provider} async stream chat: {str(e)}")
            raise e

    def finalize_stream_response(self, accumulated_content: str, query: str = "", chat_history: Optional[List[ChatMessage]] = None) -> LLMResponse:
        """Convert accumulated streaming content to final LLMResponse with token calculation."""
        # For streaming, we don't have API token usage, so calculate manually
        token_usage = self._calculate_fallback_tokens(accumulated_content, query, chat_history)
        return self._create_llm_response(accumulated_content, token_usage, query, chat_history)
    def get_token_usage_summary(self, responses: List[LLMResponse]) -> dict:
        """Get summary of token usage across multiple responses."""
        total_prompt_tokens = sum(r.prompt_token_count for r in responses)
        total_response_tokens = sum(r.response_token_count for r in responses)
        total_tokens = sum(r.total_token_count for r in responses)
        total_cached_tokens = sum(r.cached_content_token_count for r in responses)
        
        return {
            "total_requests": len(responses),
            "total_prompt_tokens": total_prompt_tokens,
            "total_response_tokens": total_response_tokens,
            "total_tokens": total_tokens,
            "total_cached_tokens": total_cached_tokens,
            "average_tokens_per_request": total_tokens / len(responses) if responses else 0
        }
    def _get_provider(self) -> str:
        return self.provider

    def _get_model_config(self) -> dict:
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_prompt": self.system_prompt,
        }

    @asynccontextmanager
    async def session(self):
        """Context manager to manage model session"""
        try:
            yield self
        finally:
            # Cleanup code if needed
            pass
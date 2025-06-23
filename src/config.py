from dataclasses import dataclass
import enum
from typing import Any, Dict
from pydantic import BaseModel
import dotenv 
dotenv.load_dotenv()
import os
from src.prompt import (LLM_SYSTEM_PROMPT)
from src.constants import (
    SUPPORTED_MEDIA_FILE_EXTENSIONS,SUPPORTED_NORMAL_FILE_EXTENSIONS,
    LLMProviderType,LLMModelID, LLM_MODEL_MAX_OUTPUT_TOKEN, LLM_MODEL_MAX_CONTEXT_WINDOW
)

class LLMConfig(BaseModel):
    api_key: str
    provider: LLMProviderType
    model_id: str
    temperature: float = 0.7
    max_tokens: int = 2048
    context_window: int = 128_000
    timeout: float = 100
    system_prompt: str = "You are a helpful assistant."
   
class LLMConfig(BaseModel):
    api_key: str
    provider: LLMProviderType
    model_id: str
    temperature: float = 0.7
    max_tokens: int = 2048
    system_prompt: str = "You are a helpful assistant."
    
class RAGConfig(BaseModel):
    """Configuration for RAG Manager"""
    chunk_size: int = 512
    chunk_overlap: int = 64
    default_collection: str = "documents"
    max_results: int = 5
    similarity_threshold: float = 0.7
    
class ReaderConfig(BaseModel):
    num_threads: int = 3
    image_resolution_scale: float = 2.0
    enable_ocr: bool = True
    llm_provider: LLMProviderType = LLMProviderType.OPENAI
    enable_agentic: bool = True
    max_pages: int = 100
    max_file_size: int = 20971520  # 20MB
    chunk_size: int = 6000
    chunk_overlap: int = 64
    supported_formats: list[str] = (
        SUPPORTED_MEDIA_FILE_EXTENSIONS + SUPPORTED_NORMAL_FILE_EXTENSIONS
    )     
class QdrantPayload(BaseModel):
    """Payload for vectors in Qdrant"""
    document_id: str | int
    text: str
    vector_id: str
    metadata: Dict[str, Any]
    
class Config:
    # LLM Config
    OPENAI_CONFIG: LLMConfig = LLMConfig(
        api_key=os.environ.get("OPENAI_BASE_API_KEY", ""),
        provider=LLMProviderType.OPENAI,
        model_id=LLMModelID.GPT_4_1_MINI.value,
        temperature=0.7,
        max_tokens=LLM_MODEL_MAX_OUTPUT_TOKEN.get(LLMModelID.GPT_4_1_MINI),
        context_window=LLM_MODEL_MAX_CONTEXT_WINDOW.get(LLMModelID.GPT_4_1_MINI),
        system_prompt="You are a helpful assistant.",
        timeout=100,
    )
    GEMINI_CONFIG: LLMConfig = LLMConfig(
        api_key=os.environ.get("GOOGLE_BASE_API_KEY", ""),
        provider=LLMProviderType.GOOGLE,
        model_id=LLMModelID.GEMINI_2_0_FLASH.value,
        temperature=0.7,
        max_tokens=LLM_MODEL_MAX_OUTPUT_TOKEN.get(LLMModelID.GEMINI_2_0_FLASH),
        context_window=LLM_MODEL_MAX_CONTEXT_WINDOW.get(LLMModelID.GEMINI_2_0_FLASH),
        system_prompt="You are a helpful assistant.",
        timeout=100,
    )
    READER_CONFIG: ReaderConfig = ReaderConfig()
    QDRANT_URL = os.environ.get("QDRANT_URL")
    RAG_CONFIG: RAGConfig = RAGConfig()

global_config = Config()
def get_llm_config(llm_type: LLMProviderType) -> LLMConfig:
    if llm_type == LLMProviderType.OPENAI:
        return global_config.OPENAI_CONFIG
    elif llm_type == LLMProviderType.GOOGLE:
        return global_config.GEMINI_CONFIG
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}")


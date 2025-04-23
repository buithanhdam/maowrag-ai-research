from dataclasses import dataclass
import enum
from typing import Any, Dict
from pydantic import BaseModel
import dotenv 
dotenv.load_dotenv()
import os
from src.prompt import (LLM_SYSTEM_PROMPT)

SUPPORTED_FILE_EXTENSIONS = [
    ".pdf",
    ".docx",
    ".html",
    ".txt",
    ".csv",
    ".xlsx",
    # ".json",
    # ".pptx",
]

class RAGType(enum.Enum):
    NORMAL = "normal_rag"
    HYBRID = "hybrid_rag"
    CONTEXTUAL = "contextual_rag"
    FUSION = "fusion_rag"
    HYDE = "hyde_rag"
    NAIVE = "naive_rag"
    
class LLMType(enum.Enum):
    GEMINI = "gemini"
    OPENAI = "openai"
    CLAUDE = "claude" 
     
class LLMConfig(BaseModel):
    api_key: str
    model_name: str
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
       
class QdrantPayload(BaseModel):
    """Payload for vectors in Qdrant"""
    document_id: str | int
    text: str
    vector_id: str
    metadata: Dict[str, Any]
    
class Config:
    OPENAI_CONFIG = LLMConfig(
        api_key=os.environ.get('OPENAI_API_KEY'),
        model_name=LLMType.OPENAI,
        model_id="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens= 2048,
        system_prompt=LLM_SYSTEM_PROMPT
    )

    GEMINI_CONFIG = LLMConfig(
        api_key=os.environ.get('GOOGLE_API_KEY'),
        model_name=LLMType.GEMINI,
        model_id="models/gemini-2.0-flash",
        temperature=0.8,
        max_tokens = 2048,
        system_prompt=LLM_SYSTEM_PROMPT
    )
    CLAUDE_CONFIG = LLMConfig(
        api_key=os.environ.get('ANTHROPIC_API_KEY'),
        model_name=LLMType.CLAUDE,
        model_id="claude-3-haiku-20240307",
        temperature=0.7,
        max_tokens=4000,
        system_prompt=LLM_SYSTEM_PROMPT
    )
    QDRANT_URL = os.environ.get("QDRANT_URL")
    RAG_CONFIG: RAGConfig = RAGConfig()

global_config = Config()
def get_llm_config(llm_type: LLMType) -> LLMConfig:
    if llm_type == LLMType.OPENAI:
        return global_config.OPENAI_CONFIG
    elif llm_type == LLMType.GEMINI:
        return global_config.GEMINI_CONFIG
    elif llm_type == LLMType.CLAUDE:
        return global_config.CLAUDE_CONFIG
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}")


import re
from typing import Any, List
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.core import Document
from llama_index.core.constants import DEFAULT_CHUNK_SIZE
from llama_index.embeddings.openai import OpenAIEmbedding
from src.config import global_config
from src.llm import GeminiEmbedding
def sanitize_json_string(s: str) -> str:
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', s)
def clean_json_response(response: str) -> str:
    """Clean JSON response from LLM output to make it parseable"""
    # Extract JSON content if wrapped in markdown code blocks
    if "```json" in response:
        start = response.find("```json") + 7
        end = response.find("```", start)
        response = response[start:end].strip()
    elif "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        response = response[start:end].strip()
    
    # Remove any potential trailing commas before closing brackets/braces
    response = response.replace(",]", "]").replace(",}", "}")
    
    return response

def text_splitter(
    text: Any,
    chunk_size : int =DEFAULT_CHUNK_SIZE,
    active: str = "sentence",
    embed_model: str = "google"
) -> List[str]:
    """Splitter doc text to chunks

    Args:
        text (Any): document text
        chunk_size (int, optional): chunk_size to split. Defaults to DEFAULT_CHUNK_SIZE.
        active (str, optional): splitter type active. Defaults to "sentence" or "semantic".
        embed_model (str, optional): embed model for sematic splitter. Defaults to "gemini" or "openai".

    Returns:
        List[str]: list of doc chunk
    """
    text = str(text)
    doc = Document(text=text,doc_id ="1",extra_info={})
    if active == "semantic":
        if embed_model == "openai":
            model =OpenAIEmbedding(
                api_key=global_config.OPENAI_CONFIG.api_key,
            )
        else:
            model =GeminiEmbedding(
                api_key=global_config.GEMINI_CONFIG.api_key,
                model_name="models/text-embedding-004",
                output_dimensionality=768
            )
        splitter = SemanticSplitterNodeParser(
            buffer_size=1, breakpoint_percentile_threshold=95, embed_model=model
        )    
    else:
        splitter = SentenceSplitter(chunk_size)
    nodes =splitter.get_nodes_from_documents([doc])
    results = [node.get_content() for node in nodes]
    return results
    
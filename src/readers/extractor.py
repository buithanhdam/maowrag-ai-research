from pathlib import Path
from .markitdown import MarkItDown
from google import genai
from openai import OpenAI
from src.config import global_config, get_llm_config

def get_extractor():
    llm_config = get_llm_config(global_config.READER_CONFIG.llm_provider)
    if llm_config.provider.value == "openai":
        llm_client = OpenAI(api_key=llm_config.api_key,)
    elif llm_config.provider.value == "google":
        llm_client=genai.Client(api_key=llm_config.api_key)
    md = MarkItDown(
        llm_client=llm_client,
        llm_model=llm_config.model_id,
        enable_plugins=False
    )
    map_extractor = {}
    [
        map_extractor.update({ext: md})
        for ext in global_config.READER_CONFIG.supported_formats
    ]
    return map_extractor
    
class FileExtractor:
    def __init__(self) -> None:
        self.extractor = get_extractor()

    def get_extractor_for_file(self, file_path: str | Path) -> dict[str, str]:
        file_suffix = Path(file_path).suffix
        return {
            file_suffix: self.extractor[file_suffix],
        }
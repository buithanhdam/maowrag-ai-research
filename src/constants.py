# Các constants không đổi
import enum

SUPPORTED_NORMAL_FILE_EXTENSIONS = [
    ".pdf", ".docx", ".json", ".jsonl", ".txt", ".pptx", 
    ".md", ".text", ".markdown", ".csv", ".msg", ".html", ".xlsx", ".xls",
]

SUPPORTED_MEDIA_FILE_EXTENSIONS = [     
    ".wav", ".mp3", ".m4a", ".mp4", ".jpg", ".jpeg", ".png"
]

ACCEPTED_MIME_MEDIA_TYPE_PREFIXES = [
    "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp4",
    "video/mp4", "image/jpeg", "image/png",
]
class RAGType(enum.Enum):
    NORMAL = "normal_rag"
    HYBRID = "hybrid_rag"
    CONTEXTUAL = "contextual_rag"
    FUSION = "fusion_rag"
    HYDE = "hyde_rag"
    NAIVE = "naive_rag"
    
class LLMProviderType(enum.Enum):
    GOOGLE = "google"
    OPENAI = "openai"

class LLMModelID(enum.Enum):
    GPT_4_1_NANO = "gpt-4.1-nano-2025-04-14"
    GPT_4_1_MINI = "gpt-4.1-mini-2025-04-14"
    GPT_4O_MINI = "gpt-4o-mini-2024-07-18"
    GEMINI_2_5_FLASH = "models/gemini-2.5-flash"
    GEMINI_2_5_FLASH_LITE = "models/gemini-2.5-flash-lite-preview-06-17"
    GEMINI_2_0_FLASH = "models/gemini-2.0-flash"

LLM_MODEL_MAX_OUTPUT_TOKEN = {
    LLMModelID.GEMINI_2_0_FLASH: 8_192,
    LLMModelID.GEMINI_2_5_FLASH: 65_536,
    LLMModelID.GEMINI_2_5_FLASH_LITE: 64_000,
    LLMModelID.GPT_4_1_MINI: 32_768,
    LLMModelID.GPT_4_1_NANO: 32_768,
    LLMModelID.GPT_4O_MINI: 16_384,
}

LLM_MODEL_MAX_CONTEXT_WINDOW={
    LLMModelID.GEMINI_2_0_FLASH: 1_048_576,
    LLMModelID.GEMINI_2_5_FLASH: 1_048_576,
    LLMModelID.GEMINI_2_5_FLASH_LITE: 1_000_000,
    LLMModelID.GPT_4_1_MINI: 1_047_576,
    LLMModelID.GPT_4_1_NANO: 1_047_576,
    LLMModelID.GPT_4O_MINI: 128_000,
}
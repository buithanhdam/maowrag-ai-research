from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from datetime import datetime
from llama_index.core import Document
import ast
from tqdm import tqdm
from src.config import global_config
from .markitdown import DocumentConverterResult
from .chunker import Chunker

load_dotenv()


def check_valid_extenstion(file_path: str | Path) -> bool:
    """Check if the file extension is supported"""
    return Path(file_path).suffix in global_config.READER_CONFIG.supported_formats


def get_files_from_folder_or_file_paths(files_or_folders: list[str]) -> list[str]:
    """Get all files from the list of file paths or folders"""
    files = []

    for file_or_folder in files_or_folders:
        if Path(file_or_folder).is_dir():
            files.extend([
                str(file_path.resolve())
                for file_path in Path(file_or_folder).rglob("*")
                if check_valid_extenstion(file_path)
            ])
        else:
            if check_valid_extenstion(file_or_folder):
                files.append(str(Path(file_or_folder).resolve()))
            else:
                print(f"Invalid file: {file_or_folder}")

    return files


def create_base_metadata(result: DocumentConverterResult, file_path: Path) -> dict:
    """Create base metadata for documents"""
    return {
        "title": result.title,
        "created_at": datetime.now().isoformat(),
        "file_name": file_path.name,
        "file_path": str(file_path),
        "original_index": 0  # Will be updated for multi-part files
    }


def create_document_from_parts(text_parts: list, images: list, base_metadata: dict) -> list[Document]:
    """Create documents from text parts and images"""
    documents = []
    
    # Ensure images list matches text_parts length
    if not images:
        images = [None] * len(text_parts)
    elif len(images) < len(text_parts):
        images.extend([None] * (len(text_parts) - len(images)))
    
    for idx, (text, image) in enumerate(zip(text_parts, images)):
        metadata = base_metadata.copy()
        metadata["original_index"] = idx
        
        if image:
            metadata["images"] = [image]  # Consistent list format
        
        documents.append(Document(text=text, metadata=metadata))
    
    return documents


def parse_multiple_files(
    files_or_folder: list[str] | str, 
    extractor: dict[str, Any],
    show_progress: bool = True
) -> list[Document]:
    """Read the content of multiple files."""
    logger = global_config.GET_LOGGER(__name__)
    assert extractor, "Extractor is required."

    if isinstance(files_or_folder, str):
        files_or_folder = [files_or_folder]

    valid_files = get_files_from_folder_or_file_paths(files_or_folder)

    if len(valid_files) == 0:
        raise ValueError("No valid files found.")

    logger.info(f"Valid files: {valid_files}")

    documents: list[Document] = []
    worker = Chunker()
    
    files_to_process = tqdm(valid_files, desc="Starting parse files", unit="file") if show_progress else valid_files
    
    for file in files_to_process:
        file_path_obj = Path(file)
        file_suffix = file_path_obj.suffix.lower()
        file_extractor = extractor[file_suffix]
        result: DocumentConverterResult = file_extractor.convert(file)
        
        base_metadata = create_base_metadata(result, file_path_obj)
        
        # Handle structured files (Excel, PDF with multiple parts)
        if file_suffix in ['.xlsx', '.xls', '.pdf']:
            try:
                text_contents: list = ast.literal_eval(result.text_content)
                documents.extend(create_document_from_parts(
                    text_contents, 
                    result.base64_images or [], 
                    base_metadata
                ))
            except Exception as e:
                logger.warning(f"Failed to parse structured content in {file_path_obj.name}: {e}")
                # Fallback to single document
                if result.base64_images:
                    base_metadata["images"] = result.base64_images
                documents.append(Document(text=result.text_content, metadata=base_metadata))
        
        # Handle regular files
        else:
            if result.base64_images:
                base_metadata["images"] = result.base64_images
            documents.append(Document(text=result.text_content, metadata=base_metadata))

    # Apply chunking strategy
    if global_config.READER_CONFIG.enable_agentic:
        documents = worker.chunking_document_by_agentic(documents)
    else:
        documents = worker.chunking_document_by_chunk_size(documents)
    
    logger.info(f"Parsed {len(valid_files)} files into {len(documents)} chunks")
    return documents

# This file contains utility functions for the readers module.
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from datetime import datetime
from llama_index.core import Document
import ast
from tqdm import tqdm
from src.config import global_config
from .markitdown import DocumentConverterResult
from .tokenizer import ReaderWorker

load_dotenv()


def check_valid_extenstion(file_path: str | Path) -> bool:
    """
    Check if the file extension is supported

    Args:
        file_path (str | Path): File path to check

    Returns:
        bool: True if the file extension is supported, False otherwise.
    """
    return Path(file_path).suffix in global_config.READER_CONFIG.supported_formats


def get_files_from_folder_or_file_paths(files_or_folders: list[str]) -> list[str]:
    """
    Get all files from the list of file paths or folders

    Args:
        files_or_folders (list[str]): List of file paths or folders

    Returns:
        list[str]: List of valid file paths.
    """
    files = []

    for file_or_folder in files_or_folders:
        if Path(file_or_folder).is_dir():
            files.extend(
                [
                    str(file_path.resolve())
                    for file_path in Path(file_or_folder).rglob("*")
                    if check_valid_extenstion(file_path)
                ]
            )

        else:
            if check_valid_extenstion(file_or_folder):
                files.append(str(Path(file_or_folder).resolve()))
            else:
                print(f"Invalid file: {file_or_folder}")

    return files

def parse_multiple_files(
    files_or_folder: list[str] | str, extractor: dict[str, Any],
    show_progress: bool = True
) -> list[Document]:
    """
    Read the content of multiple files.

    Args:
        files_or_folder (list[str] | str): List of file paths or folder paths containing files.
        extractor (dict[str, Any]): Extractor to extract content from files.
    Returns:
        list[Document]: List of documents from all files.
    """
    logger = global_config.GET_LOGGER(__name__)
    assert extractor, "Extractor is required."

    if isinstance(files_or_folder, str):
        files_or_folder = [files_or_folder]

    valid_files = get_files_from_folder_or_file_paths(files_or_folder)

    if len(valid_files) == 0:
        raise ValueError("No valid files found.")

    logger.info(f"Valid files: {valid_files}")

    documents: list[Document] = []
    worker = ReaderWorker()
    files_to_process = tqdm(valid_files, desc="Starting parse files", unit="file") if show_progress else valid_files
    for file in files_to_process:
        file_path_obj = Path(file)
        file_suffix = file_path_obj.suffix.lower()
        file_extractor = extractor[file_suffix]
        result: DocumentConverterResult = file_extractor.convert(file)
        metadata={
            "title": result.title,
            "created_at": datetime.now().isoformat(),
            "file_name": file_path_obj.name,
        }
        if file_suffix in ['.xlsx', '.xls', '.pdf']:
            try:
                text_contents: list = ast.literal_eval(result.text_content)
                images = result.base64_images
                if not images:
                    images = [None] * len(text_contents)

                for idx, (text, image) in enumerate(zip(text_contents, images)):
                    text_metadata = metadata.copy()
                    text_metadata["index"] = idx
                    if image:
                        text_metadata["images"] = image
                    documents.append(
                        Document(
                            text=text,
                            metadata=text_metadata,
                        )
                    )
            except Exception as e:
                logger.warning(f"Failed to parse structured sheets in {file_path_obj.name}: {e}")
                documents.append(
                    Document(
                        text=result.text_content,
                        metadata=metadata,
                    )
                )
        elif file_suffix in global_config.READER_CONFIG.supported_formats:

            metadata["images"] = result.base64_images
                
            documents.append(
                Document(
                    text=result.text_content,
                    metadata=metadata,
                )
            )
    if global_config.READER_CONFIG.enable_agentic:
        documents = worker.clustering_document_by_agentic_chunker(documents)
    else:
        documents = worker.clustering_document_by_block_token(documents)
    logger.info(f"Parse files successfully with {files_or_folder} split to {len(documents)} documents")
    return documents
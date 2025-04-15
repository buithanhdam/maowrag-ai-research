# This file contains utility functions for the readers module.

import sys
from pathlib import Path
from typing import Any
from dotenv import load_dotenv
from llama_index.core import Document
from llama_index.core import SimpleDirectoryReader
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config import SUPPORTED_FILE_EXTENSIONS
from src.logger import get_formatted_logger


load_dotenv()
logger = get_formatted_logger(__file__)


def check_valid_extenstion(file_path: str | Path) -> bool:
    """
    Check if the file extension is supported

    Args:
        file_path (str | Path): File path to check

    Returns:
        bool: True if the file extension is supported, False otherwise.
    """
    return Path(file_path).suffix in SUPPORTED_FILE_EXTENSIONS


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
                logger.warning(f"Invalid file: {file_or_folder}")

    return files

def split_text(text, max_tokens) -> list[str]:
    words = text.split()  # Tokenize by spaces and newlines
    chunks = []
    current_chunk = []
    current_token_count = 0

    for word in words:
        token_count = len(word.split())  # Number of tokens in the word

        # If adding the current word exceeds max_tokens, finalize the current chunk
        if current_token_count + token_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]  # Start a new chunk with the current word
            current_token_count = token_count
        else:
            current_chunk.append(word)
            current_token_count += token_count

    # Add the final chunk if any tokens are left
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
def parse_multiple_files(
    files_or_folder: list[str] | str, extractor: dict[str, Any]
) -> list[Document]:
    """
    Read the content of multiple files.

    Args:
        files_or_folder (list[str] | str): List of file paths or folder paths containing files.
        extractor (dict[str, Any]): Extractor to extract content from files.
    Returns:
        list[Document]: List of documents from all files.
    """
    assert extractor, "Extractor is required."

    if isinstance(files_or_folder, str):
        files_or_folder = [files_or_folder]

    valid_files = get_files_from_folder_or_file_paths(files_or_folder)

    if len(valid_files) == 0:
        raise ValueError("No valid files found.")

    logger.info(f"Valid files: {valid_files}")

    documents = SimpleDirectoryReader(
        input_files=valid_files,
        file_extractor=extractor,
    ).load_data(show_progress=True)
    logger.info(len(documents))
    return documents
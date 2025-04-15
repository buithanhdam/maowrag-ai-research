# from llama_index.readers.json import JSONReader
from pathlib import Path
from llama_index.readers.file import (
    PandasCSVReader,
    PptxReader,  # noqa
    UnstructuredReader,
    MarkdownReader,
    IPYNBReader,
    MboxReader,
    XMLReader,
    RTFReader
)

from src.readers.loaders import (DocxReader,TxtReader,ExcelReader,HtmlReader,MhtmlReader,PDFReader,PDFThumbnailReader,PandasExcelReader)

def get_extractor():
    return {
        ".pdf": PDFReader(),
        ".docx": DocxReader(),
        ".html": HtmlReader(),
        ".csv": PandasCSVReader(pandas_config=dict(on_bad_lines="skip")),
        ".xlsx": ExcelReader(),
        # ".json": JSONReader(),
        ".txt": TxtReader(),
        # ".pptx": PptxReader(),
        ".md": MarkdownReader(),
        ".ipynb": IPYNBReader(),
        ".mbox": MboxReader(),
        ".xml": XMLReader(),
        ".rtf": RTFReader(),
    }
    
class FileExtractor:
    def __init__(self) -> None:
        self.extractor = get_extractor()

    def get_extractor_for_file(self, file_path: str | Path) -> dict[str, str]:
        file_suffix = Path(file_path).suffix
        return {
            file_suffix: self.extractor[file_suffix],
        }

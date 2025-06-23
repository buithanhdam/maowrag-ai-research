import sys
from typing import BinaryIO, Any
from .html_converter import HtmlConverter
from .._base_converter import DocumentConverter, DocumentConverterResult
from .._exceptions import MissingDependencyException, MISSING_DEPENDENCY_MESSAGE
from .._stream_info import StreamInfo
from io import BytesIO
from ._custom_llm_caption import llm_caption
# Try loading optional (but in this case, required) dependencies
# Save reporting of any exceptions for later
_xlsx_dependency_exc_info = None
try:
    import pandas as pd
    import openpyxl
    from openpyxl_image_loader import SheetImageLoader
except ImportError:
    _xlsx_dependency_exc_info = sys.exc_info()

_xls_dependency_exc_info = None
try:
    import pandas as pd
    import xlrd
except ImportError:
    _xls_dependency_exc_info = sys.exc_info()

ACCEPTED_XLSX_MIME_TYPE_PREFIXES = [
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
]
ACCEPTED_XLSX_FILE_EXTENSIONS = [".xlsx"]

ACCEPTED_XLS_MIME_TYPE_PREFIXES = [
    "application/vnd.ms-excel",
    "application/excel",
]
ACCEPTED_XLS_FILE_EXTENSIONS = [".xls"]


class XlsxConverter(DocumentConverter):
    """
    Converts XLSX files to Markdown, with each sheet presented as a separate Markdown table.
    """

    def __init__(self):
        super().__init__()
        self._html_converter = HtmlConverter()

    def accepts(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,  # Options to pass to the converter
    ) -> bool:
        mimetype = (stream_info.mimetype or "").lower()
        extension = (stream_info.extension or "").lower()

        if extension in ACCEPTED_XLSX_FILE_EXTENSIONS:
            return True

        for prefix in ACCEPTED_XLSX_MIME_TYPE_PREFIXES:
            if mimetype.startswith(prefix):
                return True

        return False

    def convert(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,  # Options to pass to the converter
    ) -> DocumentConverterResult:
        # Check the dependencies
        if _xlsx_dependency_exc_info is not None:
            raise MissingDependencyException(
                MISSING_DEPENDENCY_MESSAGE.format(
                    converter=type(self).__name__,
                    extension=".xlsx",
                    feature="xlsx",
                )
            ) from _xlsx_dependency_exc_info[
                1
            ].with_traceback(  # type: ignore[union-attr]
                _xlsx_dependency_exc_info[2]
            )
        llm_client = kwargs.get("llm_client")
        llm_model = kwargs.get("llm_model")
        llm_prompt=kwargs.get("llm_prompt")
        
        # Tạo BytesIO cho openpyxl
        file_bytes = file_stream.read()
        
        pxl_doc = openpyxl.load_workbook(BytesIO(file_bytes), data_only=True)
        sheets = pd.read_excel(BytesIO(file_bytes), sheet_name=None, engine="openpyxl")
        
        md_contents = []
        images = []
        for sheet_name in sheets:
            sheet_images=[]
            s_content = f"## {sheet_name}\n"
            html_content = sheets[sheet_name].to_html(index=False)
            
            s_content += (
                self._html_converter.convert_string(
                    html_content, **kwargs
                ).markdown.strip()
                + "\n\n"
            )
            try:
                sheet = pxl_doc[sheet_name]
                print(sheet)
                if hasattr(sheet, '_images') and sheet._images:
                    for idx,img in enumerate(sheet._images):
                        print(img)
                        if img:
                            cell = img.ref
                            img_data = img._data()
                            img_io = BytesIO(img_data)
                            img_stream_info = StreamInfo(
                                extension=".png",
                                mimetype="image/png",
                            )
                            img_description , base64_image = llm_caption (
                                file_stream=BytesIO(img_io.getvalue()),
                                stream_info=img_stream_info,
                                client=llm_client,
                                model=llm_model,
                                prompt=llm_prompt,
                            )
                            s_content += f"[Image at {idx}] with content: {img_description} \n\n"
                            sheet_images.append(base64_image)
            except Exception as e:
                print(f"Error in process XLSX Converter: {e}")
                pass
            md_contents.append(s_content)
            images.append(sheet_images if sheet_images else None)

        return DocumentConverterResult(markdown=f"{md_contents}",base64_images=images)


class XlsConverter(DocumentConverter):
    """
    Converts XLS files to Markdown, with each sheet presented as a separate Markdown table.
    """

    def __init__(self):
        super().__init__()
        self._html_converter = HtmlConverter()

    def accepts(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,  # Options to pass to the converter
    ) -> bool:
        mimetype = (stream_info.mimetype or "").lower()
        extension = (stream_info.extension or "").lower()

        if extension in ACCEPTED_XLS_FILE_EXTENSIONS:
            return True

        for prefix in ACCEPTED_XLS_MIME_TYPE_PREFIXES:
            if mimetype.startswith(prefix):
                return True

        return False

    def convert(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,  # Options to pass to the converter
    ) -> DocumentConverterResult:
        # Load the dependencies
        if _xls_dependency_exc_info is not None:
            raise MissingDependencyException(
                MISSING_DEPENDENCY_MESSAGE.format(
                    converter=type(self).__name__,
                    extension=".xls",
                    feature="xls",
                )
            ) from _xls_dependency_exc_info[
                1
            ].with_traceback(  # type: ignore[union-attr]
                _xls_dependency_exc_info[2]
            )

        sheets = pd.read_excel(file_stream, sheet_name=None, engine="xlrd")
        md_content = []
        for s in sheets:
            s_content = f"## {s}\n"
            html_content = sheets[s].to_html(index=False)
            
            s_content += (
                self._html_converter.convert_string(
                    html_content, **kwargs
                ).markdown.strip()
                + "\n\n"
            )
            
            md_content.append(s_content)

        return DocumentConverterResult(markdown=f"{md_content}")
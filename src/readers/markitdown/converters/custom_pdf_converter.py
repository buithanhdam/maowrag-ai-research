import sys
import io

from typing import BinaryIO, Any, Dict, List
from io import BytesIO
from .._base_converter import DocumentConverter, DocumentConverterResult
from .._stream_info import StreamInfo
from .._exceptions import MissingDependencyException, MISSING_DEPENDENCY_MESSAGE
from ._custom_llm_caption import llm_caption

# Try loading optional (but in this case, required) dependencies
# Save reporting of any exceptions for later
_dependency_exc_info = None
try:
    import pdfminer
    import pdfminer.high_level
    import fitz
except ImportError:
    # Preserve the error and stack trace for later
    _dependency_exc_info = sys.exc_info()


ACCEPTED_MIME_TYPE_PREFIXES = [
    "application/pdf",
    "application/x-pdf",
]

ACCEPTED_FILE_EXTENSIONS = [".pdf"]


def extract_images_from_page(pdf_document, page_num: int) -> List[bytes]:
    """Extract images from a specific page using PyMuPDF"""
    images = []
    try:
        page = pdf_document[page_num]
        image_list = page.get_images()

        for img_index, img in enumerate(image_list):
            xref = img[0]
            pix = fitz.Pixmap(pdf_document, xref)

            # Convert to PNG if not already
            if pix.n - pix.alpha < 4:  # GRAY or RGB
                img_data = pix.tobytes("png")
                images.append(img_data)
            else:  # CMYK: convert to RGB first
                pix1 = fitz.Pixmap(fitz.csRGB, pix)
                img_data = pix1.tobytes("png")
                images.append(img_data)
                pix1 = None
            pix = None

    except Exception as e:
        print(f"Error extracting images from page {page_num}: {e}")

    return images


def process_page_content(
    pdf_document,
    page_num: int,
    llm_client=None,
    llm_model=None,
    llm_prompt=None
) -> Dict[str, Any]:
    """Process a single page: extract text and images"""
    page_content = {
        "page_number": page_num + 1,
        "text": "",
        "images": [],
    }
    images = []
    image_descriptions = []

    try:
        # Extract text from page
        page = pdf_document[page_num]
        page_text = page.get_text()
        # pdfminer.high_level.extract_text()
        page_content["text"] = page_text

        # Extract images from page
        if llm_client:
            images_data = extract_images_from_page(pdf_document, page_num)

            for idx, img_data in enumerate(images_data):
                try:
                    img_io = BytesIO(img_data)
                    img_stream_info = StreamInfo(
                        extension=".png",
                        mimetype="image/png",
                    )

                    # Generate caption using LLM
                    img_description, imgbase64 = llm_caption(
                        file_stream=BytesIO(img_io.getvalue()),
                        stream_info=img_stream_info,
                        client=llm_client,
                        model=llm_model,
                        prompt=llm_prompt,
                    )
                    images.append(imgbase64)
                    image_descriptions.append(img_description)

                except Exception as e:
                    print(f"Error processing image {idx} on page {page_num + 1}: {e}")

        # Generate markdown for the page
        page_markdown = f"## Page {page_num + 1}\n\n"

        if page_text.strip():
            page_markdown += f"{page_text}\n\n"

        # Add images to markdown
        for idx,des in enumerate(image_descriptions):
            page_markdown += f"[Image at {idx}] with content: {des}\n\n"

        page_content["text"] = page_markdown

    except Exception as e:
        print(f"Error processing page {page_num + 1}: {e}")
        page_content["text"] = (
            f"## Page {page_num + 1}\n\nPage content notfound.\n\n"
        )
    page_content["images"] = images
    return page_content


class PdfConverter(DocumentConverter):
    """
    Converts PDFs to Markdown. Most style information is ignored, so the results are essentially plain-text.
    """

    def accepts(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,  # Options to pass to the converter
    ) -> bool:
        mimetype = (stream_info.mimetype or "").lower()
        extension = (stream_info.extension or "").lower()

        if extension in ACCEPTED_FILE_EXTENSIONS:
            return True

        for prefix in ACCEPTED_MIME_TYPE_PREFIXES:
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
        if _dependency_exc_info is not None:
            raise MissingDependencyException(
                MISSING_DEPENDENCY_MESSAGE.format(
                    converter=type(self).__name__,
                    extension=".pdf",
                    feature="pdf",
                )
            ) from _dependency_exc_info[
                1
            ].with_traceback(  # type: ignore[union-attr]
                _dependency_exc_info[2]
            )
        llm_client = kwargs.get("llm_client")
        llm_model = kwargs.get("llm_model")
        llm_prompt=kwargs.get("llm_prompt")
        assert isinstance(file_stream, io.IOBase)  # for mypy
        # Read the PDF file into memory
        file_stream.seek(0)
        pdf_bytes = file_stream.read()
        md_contents = []
        images = []
        
        try:
            # Open PDF with PyMuPDF for better image handling
            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            
            # Process each page
            for page_num in range(len(pdf_document)):
                page_content = process_page_content(
                    pdf_document=pdf_document,
                    page_num=page_num,
                    llm_client=llm_client,
                    llm_model=llm_model,
                    llm_prompt=llm_prompt,
                )
                
                md_contents.append(page_content['text'])
                images.append(page_content['images'])
            
            pdf_document.close()
            
        except Exception as e:
            print(f"Error processing PDF with PyMuPDF: {e}")
            # Fallback to basic text extraction
            file_stream.seek(0)
            basic_text = pdfminer.high_level.extract_text(file_stream)
            md_contents.append(basic_text)
            images.append(None)
        
        return DocumentConverterResult(
            markdown=f"{md_contents}",
            base64_images=images
            
        )

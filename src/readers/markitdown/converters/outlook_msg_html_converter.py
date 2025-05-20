import sys
import io
from typing import Any, BinaryIO
from .._base_converter import DocumentConverter, DocumentConverterResult
from .._stream_info import StreamInfo
from .._exceptions import MissingDependencyException, MISSING_DEPENDENCY_MESSAGE
from .html_converter import HtmlConverter

try:
    import extract_msg
except ImportError:
    raise MissingDependencyException(
        MISSING_DEPENDENCY_MESSAGE.format(
            converter="OutlookMsgConverter",
            extension=".msg",
            feature="extract-msg",
        )
    )

ACCEPTED_MIME_TYPE_PREFIXES = ["application/vnd.ms-outlook"]
ACCEPTED_FILE_EXTENSIONS = [".msg"]

class OutlookMsgHTMLConverter(DocumentConverter):
    def accepts(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,
    ) -> bool:
        mimetype = (stream_info.mimetype or "").lower()
        extension = (stream_info.extension or "").lower()
        return (
            extension in ACCEPTED_FILE_EXTENSIONS
            or any(mimetype.startswith(prefix) for prefix in ACCEPTED_MIME_TYPE_PREFIXES)
        )

    def convert(
        self,
        file_stream: BinaryIO,
        stream_info: StreamInfo,
        **kwargs: Any,
    ) -> DocumentConverterResult:
        # Save stream to temporary in-memory file because extract_msg requires a file path or file-like
        import tempfile
        with tempfile.NamedTemporaryFile(delete=True) as tmp_file:
            # Ensure we're working with bytes data
            content = file_stream.read()
            if not isinstance(content, bytes):
                content = content.encode('utf-8')
            
            tmp_file.write(content)
            tmp_file.flush()
            
            # Parse the MSG file
            msg = extract_msg.Message(tmp_file.name)
            
            headers = {
                "From": msg.sender or "",
                "To": msg.to or "",
                "Subject": msg.subject or "",
            }
            
            # Try to get HTML body first
            html_body = None
            plain_body = None
            
            try:
                html_body = msg.htmlBody
                if html_body and not isinstance(html_body, str):
                    html_body = html_body.decode('utf-8', errors='replace')
            except Exception:
                html_body = None
                
            try:
                plain_body = msg.body
                if plain_body and not isinstance(plain_body, str):
                    plain_body = plain_body.decode('utf-8', errors='replace')
            except Exception:
                plain_body = None
            
            # Prefer HTML -> convert via HtmlConverter
            if html_body and "<html" in html_body.lower():
                try:
                    html_converter = HtmlConverter()
                    # Use bytes input if the HtmlConverter expects bytes
                    if hasattr(html_converter, 'convert_bytes'):
                        html_result = html_converter.convert_bytes(html_body.encode('utf-8'), **kwargs)
                    else:
                        html_result = html_converter.convert_string(html_body, **kwargs)
                    body_md = html_result.markdown
                except Exception as e:
                    # Fallback to plaintext if HTML conversion fails
                    body_md = plain_body or f"(HTML conversion failed: {str(e)})"
            elif plain_body:
                body_md = plain_body
            else:
                body_md = "(No content)"
                
            # Compose final markdown
            md_content = "# Email Message\n\n"
            for key, value in headers.items():
                if value:
                    md_content += f"**{key}:** {value}\n"
            md_content += "\n## Content\n\n"
            md_content += body_md.strip()
            
            # Check for attachments
            try:
                attachments = msg.attachments
                if attachments:
                    md_content += "\n\n## Attachments\n\n"
                    for i, attachment in enumerate(attachments):
                        md_content += f"- Attachment {i+1}: {attachment.longFilename or 'Unnamed'}\n"
            except Exception as e:
                md_content += f"\n\n## Attachments\n\n(Error retrieving attachments: {str(e)})"
            
            return DocumentConverterResult(
                markdown=md_content.strip(),
                title=headers.get("Subject", "Email Message"),
            )
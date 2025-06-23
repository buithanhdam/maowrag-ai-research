import io
import re
import sys
from typing import BinaryIO, Any, Dict, List, Optional
from .._base_converter import DocumentConverter, DocumentConverterResult
from .._stream_info import StreamInfo
from .._exceptions import MissingDependencyException, MISSING_DEPENDENCY_MESSAGE
from .html_converter import HtmlConverter
from ._custom_llm_caption import llm_caption
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

def _extract_cid_from_markdown(markdown_content: str) -> Dict[str, str]:
    """
    Extract CID references from markdown content.
    Returns dict mapping CID to the full markdown image syntax.
    """
    # Pattern to match ![](cid:filename@contentid) or ![alt](cid:filename@contentid)
    cid_pattern = r'!\[([^\]]*)\]\(cid:([^@\)]+)@?[^\)]*\)'
    matches = re.findall(cid_pattern, markdown_content)
    
    cid_map = {}
    for alt_text, filename in matches:
        # Store the full markdown syntax for replacement
        full_match = re.search(rf'!\[[^\]]*\]\(cid:{re.escape(filename)}@?[^\)]*\)', markdown_content)
        if full_match:
            cid_map[filename] = full_match.group(0)
    
    return cid_map
def _find_matching_attachment(cid_filename: str, attachments) -> Optional[Any]:
    """
    Find attachment that matches the CID filename.
    """
    for attachment in attachments:
        # Try different filename attributes
        attachment_names = [
            getattr(attachment, 'longFilename', None),
            getattr(attachment, 'shortFilename', None),
            getattr(attachment, 'displayName', None),
        ]
        
        for name in attachment_names:
            if name and name.lower() == cid_filename.lower():
                return attachment
                
    # Fallback: partial match
    for attachment in attachments:
        attachment_names = [
            getattr(attachment, 'longFilename', None),
            getattr(attachment, 'shortFilename', None),
            getattr(attachment, 'displayName', None),
        ]
        
        for name in attachment_names:
            if name and cid_filename.lower() in name.lower():
                return attachment
                
    return None
def _get_image_mimetype(filename: str) -> str:
    """
    Determine MIME type based on file extension.
    """
    ext = filename.lower().split('.')[-1] if '.' in filename else ''
    mime_map = {
        'png': 'image/png',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'gif': 'image/gif',
        'bmp': 'image/bmp',
        'webp': 'image/webp',
    }
    return mime_map.get(ext, 'image/png')
def _process_attachments_with_llm(
    markdown_content: str, 
    msg, 
    llm_client,
    llm_model,
    llm_prompt
) -> tuple[str, list]:
    """
    Process attachments (image) in markdown content using LLM for captioning.
    """
    try:
        # Extract CID references from markdown
        attachment_base64s = []
        cid_map = _extract_cid_from_markdown(markdown_content)
        if not cid_map:
            return markdown_content
            
        # Get attachments
        attachments = getattr(msg, 'attachments', [])
        if not attachments:
            return markdown_content
            
        processed_content = markdown_content
        
        for idx,(cid_filename, markdown_syntax) in enumerate(cid_map.items()):
            try:
                # Find matching attachment
                attachment = _find_matching_attachment(cid_filename, attachments)
                if not attachment:
                    print(f"No matching attachment found for CID: {cid_filename}")
                    continue
                    
                # Get attachment data
                img_data = attachment.data
                if not img_data:
                    print(f"No data found for attachment: {cid_filename}")
                    continue
                    
                # Prepare image for LLM
                img_io = io.BytesIO(img_data)
                img_stream_info = StreamInfo(
                    extension=f".{cid_filename.split('.')[-1]}" if '.' in cid_filename else ".png",
                    mimetype=_get_image_mimetype(cid_filename),
                )

                # Generate caption using LLM
                img_description, imgbase64 = llm_caption(
                    file_stream=io.BytesIO(img_io.getvalue()),
                    stream_info=img_stream_info,
                    client=llm_client,
                    model=llm_model,
                    prompt=llm_prompt,
                )
                
                # Replace the markdown image syntax with description
                if img_description:
                    # You can choose different replacement formats:
                    # Option 1: Replace with description only
                    replacement = f"[Image at {idx}] with content: {img_description}\n\n"
                    
                    processed_content = processed_content.replace(markdown_syntax, replacement)
                    attachment_base64s.append(imgbase64)
                    print(f"Processed image: {cid_filename} -> {img_description[:50]}...")
                
            except Exception as e:
                print(f"Error processing image {cid_filename}: {str(e)}")
                continue
                
        return processed_content,attachment_base64s
        
    except Exception as e:
        print(f"Error in image processing: {str(e)}")
        return markdown_content,attachment_base64s
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
        llm_client = kwargs.get("llm_client")
        llm_model = kwargs.get("llm_model")
        llm_prompt=kwargs.get("llm_prompt")
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
            images= []
            
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
                print("Using HTML converter")
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
                print("Using base plain converter")
                body_md = plain_body
            else:
                print("No content found")
                body_md = "(No content)"
            # Process images with LLM if available
            if all([llm_client, llm_model]):
                print("Processing images with LLM...")
                body_md, images = _process_attachments_with_llm(
                    body_md, msg, llm_client, llm_model, llm_prompt
                )
            # Compose final markdown
            md_content = "# Email Message\n\n"
            for key, value in headers.items():
                if value:
                    md_content += f"**{key}:** {value}\n"
            md_content += "\n## Content\n\n"
            md_content += body_md.strip().replace('\r\n', '\n').replace('\r', '\n')
            
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
                base64_images=images
            )
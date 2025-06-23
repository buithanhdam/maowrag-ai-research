from typing import BinaryIO, Union, Tuple
from .._stream_info import StreamInfo
import base64
import mimetypes
from google import genai
from google.genai import types
from openai import OpenAI
from PIL import Image 

def llm_caption(
    file_stream: BinaryIO,
    stream_info: StreamInfo,
    *,
    client: genai.Client|OpenAI,
    model,
    prompt=None,
) -> Tuple[Union[None, str], Union[None, str]]:
    if prompt is None or prompt.strip() == "":
        prompt = "Write a detailed caption or Extract structured content from the image and write it clearly in Markdown format using headers, bullet points, and tables where applicable. Do not include code fences or additional explanations."

    # Get the content type
    content_type = stream_info.mimetype
    if not content_type:
        content_type, _ = mimetypes.guess_type(
            "_dummy" + (stream_info.extension or "")
        )
    
    if not content_type:
        try:
            # Save current position
            current_pos = file_stream.tell()
            # Try to guess MIME type using Pillow if available
            img = Image.open(file_stream)
            content_type = Image.MIME.get(img.format, "image/jpeg")
            # Reset stream position after opening with Pillow
            file_stream.seek(current_pos)
        except Exception:
            content_type = "image/jpeg"  # Default fallback
    
    if not content_type:
        content_type = "application/octet-stream"
            
    # Convert to base64
    cur_pos = file_stream.tell()
    try:
        image_bytes = file_stream.read()
        if not image_bytes:
            return None, None
            
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        data_uri = f"data:{content_type};base64,{base64_image}"
        
        # Call LLM for caption
        if isinstance(client, genai.Client):
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type=content_type,
                    ),
                    prompt
                ]
            )
            response = response.text
        elif isinstance(client, OpenAI):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_uri,
                            },
                        },
                    ],
                }
            ]
            # Call the OpenAI API
            response = client.chat.completions.create(model=model, messages=messages)
            response = response.choices[0].message.content
        
        return response, data_uri
        
    except Exception as e:
        print(f"Error in llm_caption: {e}")
        return None, None
    finally:
        # Always reset stream position
        try:
            file_stream.seek(cur_pos)
        except Exception:
            pass  # File might be closed already
# test with old google-generative ai vs google-genai
from dotenv import load_dotenv
load_dotenv()
import os
# from google import generativeai
from google import genai
from google.genai import types

if __name__ == "__main__":
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    with open('tests/test_data/t3qWG.png', 'rb') as f:
      img_bytes = f.read()

    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=[
        types.Part.from_bytes(
            data=img_bytes,
            mime_type='image/jpg',
        ),
        'Write a detailed caption or OCR Text if needed for this image.'
        ]
    )

    print(response.text)
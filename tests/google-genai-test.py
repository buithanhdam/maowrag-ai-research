# test with old google-generative ai vs google-genai
from dotenv import load_dotenv
load_dotenv()
import os
# from google import generativeai
from google import genai
from google.genai import types
from llama_index.llms.gemini import Gemini
from llama_index.core.llms import ChatMessage

if __name__ == "__main__":
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    with open('tests/test_data/t3qWG.png', 'rb') as f:
      img_bytes = f.read()

    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=[
        # types.Part.from_bytes(
        #     data=img_bytes,
        #     mime_type='image/jpg',
        # ),
        # types.SafetySetting(
        #     category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        #     threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        # ),
        'Write a detailed caption or OCR Text if needed for this image.'
        ]
    )
    model = client.models
    
    model.send_message
    # usage_metadata = response.usage_metadata
    # input_token_count=usage_metadata.prompt_token_count
    # output_token_count=usage_metadata.candidates_token_count
    # thoughts_token_count=usage_metadata.thoughts_token_count
    # tool_use_prompt_token_count=usage_metadata.tool_use_prompt_token_count
    # total_token_count=usage_metadata.total_token_count
    # print(response.usage_metadata)
    
    # model = Gemini(model_name='models/gemini-2.0-flash',api_key=os.environ.get("GOOGLE_API_KEY"))
    # response = model.chat(messages=[ChatMessage(content="who is messi")])
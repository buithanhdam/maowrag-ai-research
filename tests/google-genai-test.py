# test with old google-generative ai vs google-genai
import os
from dotenv import load_dotenv
load_dotenv()
from llama_index.core.llms import ChatMessage
from src.config import get_llm_config, LLMProviderType
def test_generative_ai():
    import google.generativeai as genai
    gemini_config = get_llm_config(LLMProviderType.GOOGLE)
    genai.configure(api_key=gemini_config.api_key)

    model = "models/gemini-2.5-flash-preview-04-17"
    model_meta = genai.get_model(model)
    genai_model = genai.GenerativeModel(
        model_name=model,
    )
    chat = genai_model.start_chat()
    response = chat.send_message("hello")
    print(model_meta.output_token_limit)
    print(response)
def test_llamaindex_gemini():
    from llama_index.llms.gemini import Gemini
    gemini_config = get_llm_config(LLMProviderType.GOOGLE)
    gemini = Gemini(
        api_key=gemini_config.api_key,
        model=gemini_config.model_id,
        temperature=gemini_config.temperature,
        max_tokens=gemini_config.max_tokens,
    )
    assistant_prompt =  ChatMessage(
                    role="assistant",
                    content="I understand and will follow these instructions.",
                )
    system_prompt=ChatMessage(role="system", content="You are a helpful assistant. use vietnam language to response")
    response = gemini.chat([system_prompt,assistant_prompt,ChatMessage(content="hello", role="user")])
    print(response.raw)
def test_genai():
    from google import genai
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    # with open('tests/test_data/t3qWG.png', 'rb') as f:
    #   img_bytes = f.read()

    response = client.models.generate_content(
        model='gemini-2.0-flash',
        config={
        'response_logprobs': True
        },
        contents=[
        # types.Part.from_bytes(
        #     data=img_bytes,
        #     mime_type='image/jpg',
        # ),
        # types.SafetySetting(
        #     category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        #     threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        # ),
        """what is biggest thing about nanidbt logistic company""",
        ]
    )
    # model = client.models
    
    # model.send_message
    # usage_metadata = response.usage_metadata
    # input_token_count=usage_metadata.prompt_token_count
    # output_token_count=usage_metadata.candidates_token_count
    # thoughts_token_count=usage_metadata.thoughts_token_count
    # tool_use_prompt_token_count=usage_metadata.tool_use_prompt_token_count
    # total_token_count=usage_metadata.total_token_count
    print(response)
    
    # model = Gemini(model_name='models/gemini-2.0-flash',api_key=os.environ.get("GOOGLE_API_KEY"))
    # response = model.chat(messages=[ChatMessage(content="who is messi")])
if __name__ == "__main__":
    test_genai()
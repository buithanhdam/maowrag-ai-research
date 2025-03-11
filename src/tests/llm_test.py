import asyncio
from src.agents.llm import UnifiedLLM
from llama_index.core.llms import ChatMessage

async def test_gemini_achat():
    try:
        llm = UnifiedLLM(model_name="gemini")
        response = llm.chat("Xin chào!")
        print("=== Chat đơn giản ===")
        print(f"Response: {response}")
        
        history = [
            ChatMessage(role="user", content="Bạn là ai?"),
            ChatMessage(role="assistant", content="Tôi là trợ lý AI.")
        ]
        response = await llm.achat("Rất vui được gặp bạn!", chat_history=history)
        print("\n=== Chat với history ===")
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

async def test_gemini_async():
    try:
        llm = UnifiedLLM(model_name="gemini")
        
        # response = await llm.achat("Xin chào!")
        # print("\n=== Async chat ===")
        # print(f"Response: {response}")
        
        print("\n=== Async stream chat ===")
        try:
            async for chunk in llm.astream_chat("hello"):
                print(chunk, end="", flush=True)
            print()  # New line after story
        except Exception as e:
            print(f"\nError in stream: {str(e)}")
            
    except Exception as e:
        print(f"Error in async test: {str(e)}")
        raise

if __name__ == "__main__":
    # test_gemini_sync()
    asyncio.run(test_gemini_async())
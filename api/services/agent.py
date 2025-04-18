import logging
from typing import Optional, List, AsyncGenerator, Generator
import asyncio

from src.agents import (
    ReflectionAgent,
    ReActAgent,
    AgentOptions,
)
from src.tools import get_weather_tool, search_web_tool, rag_retriever_tool,search_paper_tool
from src.llm import UnifiedLLM
from llama_index.core.llms import ChatMessage
from src.logger import get_formatted_logger
logger = get_formatted_logger(__name__)
class AgentService:
    def __init__(self):
        # Initialize LLM
        self.llm = UnifiedLLM(model_name="gemini")
        # Initialize specialized agents
        self.reflection_agent = ReflectionAgent(
            self.llm,
            AgentOptions(
                id="reflection",
                name="Reflection Assistant",
                description="Helps with information generation and refinement about football"
            ),
            system_prompt="Bạn là 1 trợ lý AI hữu ích, thân thiện và có hiểu biết sâu rộng."
        )
        
        self.planning_agent = ReActAgent(
            self.llm,
            AgentOptions(
                id="react",
                name="React Assistant",
                description="Assists can reason and act with project planning, task breakdown, and using weather tool"
            ),
            system_prompt="Bạn là 1 trợ lý AI hữu ích, thân thiện và có hiểu biết sâu rộng.",
            tools=[get_weather_tool, search_web_tool,rag_retriever_tool,search_paper_tool]
        )
        
        # Chat history to provide context
        self.chat_history: List[ChatMessage] = []
    
    async def get_response(self, user_input: str, verbose: bool = True) -> str:
        """
        Process user input by routing to appropriate agent
        
        Args:
            user_input (str): User's query
            verbose (bool): Whether to log detailed information
            
        Returns:
            str: Agent's response
        """
        try:
            # Process the input and get a response
            response = await self.planning_agent.achat(
                query=user_input,
                chat_history=self.chat_history[:-1],
                verbose=verbose
            )
            
            # Update chat history
            self.chat_history.append(ChatMessage(role="user", content=user_input))
            self.chat_history.append(ChatMessage(role="assistant", content=response))
            
            # Trim chat history to last 10 messages to prevent context overflow
            self.chat_history = self.chat_history[-10:]
            
            return response
            
        except Exception as e:
            logger.error(f"Error in get_response: {e}")
            return "I'm sorry, I encountered an error processing your request."
    
    async def stream_response(self, user_input: str, verbose: bool = True) -> AsyncGenerator[str, None]:
        """
        Process user input and stream the response from the appropriate agent
        
        Args:
            user_input (str): User's query
            verbose (bool): Whether to log detailed information
            
        Returns:
            AsyncGenerator[str, None]: Stream of agent's response chunks
        """
        try:
            # First add the user message to history
            logger.info(f"User input: {user_input}")
            self.chat_history.append(ChatMessage(role="user", content=user_input))
            
            # Get streaming response from planning agent
            full_response = ""
            async for chunk in self.planning_agent.astream_chat(
                query=user_input,
                chat_history=self.chat_history[:-1],  # Exclude the user message we just added
                verbose=verbose
            ):
                full_response += chunk
                yield chunk
            
            # After streaming is complete, update chat history with the full response
            self.chat_history.append(ChatMessage(role="assistant", content=full_response))
            logger.info(f"Final response: {full_response}")
            # Trim chat history to last 10 messages
            self.chat_history = self.chat_history[-10:]
            
        except Exception as e:
            logger.error(f"Error in stream_response: {e}")
            error_msg = "I'm sorry, I encountered an error processing your request."
            yield error_msg
            
            # Add error message to chat history
            self.chat_history.append(ChatMessage(role="assistant", content=error_msg))
    
    def reset_chat(self):
        """Reset the chat history"""
        self.chat_history = []
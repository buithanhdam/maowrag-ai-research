from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any
import asyncio

from api.services.agent import AgentService

# Create router
agent_router = APIRouter(prefix="/agent", tags=["agent"])

# Request model
class ChatRequest(BaseModel):
    query: str

# Response model
class ChatResponse(BaseModel):
    response: str

# Response model for reset endpoint
class ResetResponse(BaseModel):
    status: str
    message: str

# Initialize agent chat globally
agent_chat = AgentService()

@agent_router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint for agent chat interaction
    
    Args:
        request (ChatRequest): Contains the user query
        
    Returns:
        ChatResponse: Agent's response to the query
    """
    try:
        # Use the existing get_response method from AgentChat
        response = await agent_chat.get_response(request.query)
        return ChatResponse(response=response)
    except Exception as e:
        # Handle any potential errors
        raise HTTPException(status_code=500, detail=str(e))

@agent_router.post("/stream")
async def stream_chat_endpoint(request: ChatRequest):
    """
    Endpoint for streaming agent chat interaction
    
    Args:
        request (ChatRequest): Contains the user query
        
    Returns:
        StreamingResponse: Stream of agent's response chunks
    """
    try:
        # Create an async generator for streaming the response
        async def response_generator():
            async for chunk in agent_chat.stream_response(request.query):
                # Yield each chunk as a server-sent event
                yield chunk
            # Signal the end of the stream
            yield "[DONE]"
        
        # Return a streaming response with text/event-stream content type
        return StreamingResponse(
            response_generator(),
            media_type="text/event-stream"
        )
    except Exception as e:
        # Handle any potential errors
        raise HTTPException(status_code=500, detail=str(e))

@agent_router.post("/reset", response_model=ResetResponse)
async def reset_chat_endpoint():
    """
    Endpoint to reset the chat history
    
    Returns:
        ResetResponse: Confirmation of chat history reset
    """
    try:
        # Call the reset_chat method
        agent_chat.reset_chat()
        return ResetResponse(
            status="success", 
            message="Chat history has been successfully reset"
        )
    except Exception as e:
        # Handle any potential errors
        raise HTTPException(status_code=500, detail=str(e))
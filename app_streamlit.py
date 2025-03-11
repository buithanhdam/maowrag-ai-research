import os
import streamlit as st
import requests
import json
from typing import List, Dict, Iterator, Optional
import time
from dotenv import load_dotenv

load_dotenv()
# Get API URL from environment variable, with a default fallback
API_URL = os.getenv('API_URL', 'http://localhost:8000')

def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []

def add_message(role: str, content: str):
    st.session_state.messages.append({"role": role, "content": content})

def get_agent_response(prompt: str) -> str:
    """
    Send request to FastAPI backend and get agent response
    
    Args:
        prompt (str): User input query
    
    Returns:
        str: Agent's response
    """
    try:
        # Use the environment variable for the API URL
        full_url = f"{API_URL}/agent/chat"
        
        response = requests.post(full_url, json={"query": prompt})
        response.raise_for_status()  # Raise an error for bad responses
        
        return response.json()['response']
    except requests.RequestException as e:
        st.error(f"Error communicating with agent: {e}")
        return "Sorry, there was an error processing your request."

def stream_agent_response(prompt: str) -> Iterator[str]:
    """
    Stream response from agent API
    
    Args:
        prompt (str): User input query
    
    Returns:
        Iterator[str]: Stream of response chunks
    """
    try:
        # Use the environment variable for the API URL
        full_url = f"{API_URL}/agent/stream"
        
        # Make a streaming request to the server
        with requests.post(full_url, json={"query": prompt}, stream=True) as response:
            response.raise_for_status()
            
            # Process server-sent events
            for line in response.iter_lines():
                if line:
                    # SSE format: "data: {chunk}"
                    chunk = line.decode('utf-8')
                    if chunk == "[DONE]":
                        break
                    yield chunk
    except requests.RequestException as e:
        st.error(f"Error streaming from agent: {e}")
        yield "Sorry, there was an error processing your request."

def reset_chat_history():
    """
    Send request to reset chat history via API
    """
    try:
        full_url = f"{API_URL}/agent/reset"
        
        response = requests.post(full_url)
        response.raise_for_status()
        
        # Clear local session state messages
        st.session_state.messages = []
        
        st.success("Chat history has been reset successfully!")
    except requests.RequestException as e:
        st.error(f"Error resetting chat history: {e}")

def main():
    st.title("🤖 Multi-Agent Chat Interface")

    # Initialize session state
    initialize_session_state()

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        add_message("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response with streaming
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # Stream the response
            for chunk in stream_agent_response(prompt):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
                time.sleep(0.01)  # Small delay for smoother animation
            
            # Update with the final response (without cursor)
            response_placeholder.markdown(full_response)
            
            # Add completed response to chat history
            add_message("assistant", full_response)

    # Sidebar with agent information and settings
    with st.sidebar:
        st.subheader("Available Agents")
        st.info("📝 Reflection Assistant\n\nHelps with information based on LLM")
        st.info("📋 Planning Assistant\n\nAssists with project planning, task breakdown, and weather information")

        # Use expander for settings to keep the UI clean
        with st.expander("⚙️ Chat Settings"):
            # Add a reset button with confirmation
            if st.button("🔄 Reset Entire Chat", type="primary"):
                reset_confirmation = st.checkbox("Are you sure you want to reset the chat?", key="confirm_reset")
                if reset_confirmation:
                    reset_chat_history()
                    st.rerun()

if __name__ == "__main__":
    main()
from typing import Callable, Optional
from llama_index.core.tools import FunctionTool

def create_function_tool(
    func: Callable,
    name: Optional[str] = None,
    description: Optional[str] = None
) -> FunctionTool:
    """Helper function to create a FunctionTool with proper metadata"""
    return FunctionTool.from_defaults(
        fn=func,
        name=name or func.__name__,
        description=description or func.__doc__ or "No description provided"
    )
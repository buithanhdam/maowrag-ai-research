from typing import Callable, Optional
from llama_index.core.tools import FunctionTool
from .calculator_tool import (
    sum,
    subtract,
    multiply,
    divide,
    square_root,
    power,
    abs,convert_currency_to_USD
)
from .weather_tool import get_weather, WEATHER_DESCRIPTION
from .search_tool import search_web, search_wiki
from .paper_search_tool import search_paper
from .rag_tool import retrieve_documents

class ToolManager:
    tools = []
    """A class to manage and create tools for the application."""
    @staticmethod
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
    @staticmethod
    def get_rag_tools() -> list[FunctionTool]:
        """Get the RAG tool"""
        return [ToolManager.create_function_tool(
            func=retrieve_documents,
            name=retrieve_documents.__name__,
            description=retrieve_documents.__doc__,
        )]
    @staticmethod
    def get_search_tools() -> list[FunctionTool]:
        """Get the search tools"""
        search_web_by_tavily_fnc = search_web()
        search_wiki_by_wikipedia_fnc = search_wiki()
        search_paper_fnc = search_paper()
        return [
            ToolManager.create_function_tool(
                func=search_web_by_tavily_fnc,
                name=search_web_by_tavily_fnc.__name__,
                description=search_web_by_tavily_fnc.__doc__,
            ),
            ToolManager.create_function_tool(
                func=search_wiki_by_wikipedia_fnc,
                name=search_wiki_by_wikipedia_fnc.__name__,
                description=search_wiki_by_wikipedia_fnc.__doc__,
            ),
            ToolManager.create_function_tool(
                func=search_paper_fnc,
                name=search_paper_fnc.__name__,
                description=search_paper_fnc.__doc__,
            )
        ]
    @staticmethod
    def get_calculator_tools() -> list[FunctionTool]:
        """Get the calculator tools"""
        return [
            ToolManager.create_function_tool(
                func=sum,
                name=sum.__name__,
                description=sum.__doc__,
            ),
            ToolManager.create_function_tool(
                func=subtract,
                name=subtract.__name__,
                description=subtract.__doc__,
            ),
            ToolManager.create_function_tool(
                func=multiply,
                name=multiply.__name__,
                description=multiply.__doc__,
            ),
            ToolManager.create_function_tool(
                func=divide,
                name=divide.__name__,
                description=divide.__doc__,
            ),
            ToolManager.create_function_tool(
                func=square_root,
                name=square_root.__name__,
                description=square_root.__doc__,
            ),
            ToolManager.create_function_tool(
                func=power,
                name=power.__name__,
                description=power.__doc__,
            ),
            ToolManager.create_function_tool(
                func=abs,
                name=abs.__name__,
                description=abs.__doc__,
            ),
            ToolManager.create_function_tool(
                func=convert_currency_to_USD,
                name=convert_currency_to_USD.__name__,
                description=convert_currency_to_USD.__doc__,
            )
        ]
    @staticmethod
    def get_weather_tools() -> list[FunctionTool]:
        """Get the weather tool"""
        return [ToolManager.create_function_tool(
            func=get_weather,
            name=get_weather.__name__,
            description=WEATHER_DESCRIPTION,
        )]
    @staticmethod
    def get_all_tools() -> list[FunctionTool]:
        """Get all tools"""
        return (
            ToolManager.get_search_tools()
            + ToolManager.get_calculator_tools()
            + ToolManager.get_weather_tools()+ ToolManager.get_rag_tools()
        )
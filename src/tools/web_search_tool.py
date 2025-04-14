from .base import create_function_tool
from src.search_engine import TavilyEngine

tavily_client = TavilyEngine()
def search_web(query: str) -> dict:
    """Search the web for a given query and return results"""
    # Implementation here
    search_results=tavily_client.search(
        query=query,
        max_results=5,
    )
    return search_results["data"]["results"]
search_web_tool = create_function_tool(
    func=search_web,
    name="SearchWeb",
    description="Search the web for a given query and return results",
)
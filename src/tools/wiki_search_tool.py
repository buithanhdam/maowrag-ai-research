from .base import create_function_tool
from src.search_engine import WikipediaSearchEngine

WIKIPEDIA_DESCRIPTION = """The Wikipedia Search tool provides access to a vast collection of articles covering a wide range of topics.
Can query specific keywords or topics to retrieve accurate and comprehensive information.
"""
wikipedia_client = WikipediaSearchEngine(top_k_results=2)

def search_wiki(query: str) -> dict:
    """Search the content from Wikipedia for a given query and return results"""
    # Implementation here
    search_results=wikipedia_client.search(
        query=query
    )
    return search_results["results"]
search_wiki_tool = create_function_tool(
    func=search_wiki,
    name="search_wiki",
    description=WIKIPEDIA_DESCRIPTION,
)
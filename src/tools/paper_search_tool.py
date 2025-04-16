from .base import create_function_tool
from src.search_engine import ArXivSearchEngine

arxiv_client = ArXivSearchEngine(top_k_results=5)
def search_paper(query: str) -> dict:
    """Search the research technology paper content from arxiv for a given query and return results"""
    # Implementation here
    search_results=arxiv_client.search(
        query=query
    )
    return search_results["results"]
search_paper_tool = create_function_tool(
    func=search_paper,
    name="search_paper_by_arxiv",
    description="Search the research technology paper content from arxiv for a given query and return results",
)
from src.search_engine import ArXivSearchEngine
def search_paper():
    arxiv_client = ArXivSearchEngine(top_k_results=5)
    def search_paper_by_arxiv(query: str) -> dict:
        """The Arxiv Search tool provides access to a vast collection of academic, technology papers covering a wide range of topics."""
        # Implementation here
        search_results=arxiv_client.search(
            query=query
        )
        return search_results["results"]
    return search_paper_by_arxiv
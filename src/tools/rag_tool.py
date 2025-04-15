from src.rag.rag_manager import RAGManager
from src.config import RAGType
from src.config import Config
from .base import create_function_tool
def retrieve_documents(query: str) -> str:
    """
    Search through knowledge base return relevant information
            
    Args:
        query: Search query
    """
    settings = Config()
    rag_manager = RAGManager.create_rag(
            rag_type=RAGType.HYBRID,
            qdrant_url=settings.QDRANT_URL,
            gemini_api_key=settings.GEMINI_CONFIG.api_key,
        )
    def search_documents(query: str, collection_name: str = "test_collection", limit: int = 5) -> str:
        return rag_manager.search(query=query, collection_name=collection_name,limit=limit)
    
    return search_documents(query=query)

rag_retriever_tool = create_function_tool(
    func=retrieve_documents,
    name="rag_retriever_tool",
    description="Search through knowledge base return relevant information",
)
from src.rag.rag_manager import RAGManager
from src.config import RAGType
from src.config import global_config, LLMProviderType

def retrieve_documents(query: str) -> str:
    """
    Search through business knowledge base return relevant business information  
    Args:
        query: Search query
    """
    rag_manager = RAGManager.create_rag(
            rag_type=RAGType.HYBRID,
            vector_db_url=global_config.QDRANT_URL,
            llm_type=LLMProviderType.GOOGLE,
        )
    def search_documents(query: str, collection_name: str = "test_collection", limit: int = 5) -> str:
        return rag_manager.search(query=query, collection_name=collection_name,limit=limit)
    
    return search_documents(query=query)

from src.readers import FileExtractor, parse_multiple_files
from src.rag.rag_manager import RAGManager
from src.config import RAGType, LLMProviderType, global_config
from src.db.qdrant import QdrantVectorDatabase
import dotenv
import os 
import uuid
from datetime import datetime
dotenv.load_dotenv()
if __name__ == "__main__":
    # Example usage
    collection_name="test_collection"
    file = "tests/test_data/test-vlc.pdf"
    file_id = str(uuid.uuid4())
    
    file_extractor = FileExtractor()
    extractor = file_extractor.get_extractor_for_file(file)
    print(extractor)
    
    # Parse the files
    documents = parse_multiple_files(file, extractor)
    
    # Initialize Qdrant vector database
    qdrant_db = QdrantVectorDatabase(
        url=global_config.QDRANT_URL,
    )
    
    # Initialize RAG manager
    rag_manager = RAGManager.create_rag(
        rag_type=RAGType.NAIVE,
        vector_db_url=global_config.QDRANT_URL,
        llm_type=LLMProviderType.GOOGLE,
    )
    
    qdrant_db.create_collection(collection_name=collection_name)
    for document in documents:
        chunks = rag_manager.process_document(
            document=document,
            document_id=file_id,
            collection_name=collection_name,
            metadata={
                **document.metadata,
                "file_path": file,
                "created_at": datetime.now().isoformat(),
            }
        )
    
    
    
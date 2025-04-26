from src.readers import FileExtractor, parse_multiple_files
from src.rag.rag_manager import RAGManager
from src.config import RAGType, LLMType, global_config
from src.db.qdrant import QdrantVectorDatabase
import dotenv
import os 
import uuid
from datetime import datetime
dotenv.load_dotenv()
if __name__ == "__main__":
    # Example usage
    collection_name="test_collection"
    # file = "tests/test_data/Free AI Podcast for Nonprofits Amplify.mp3"
    file = "tests/test_data/test-vlc.pdf"
    file_id = str(uuid.uuid4())
    
    file_extractor = FileExtractor()
    extractor = file_extractor.get_extractor_for_file(file)
    print(extractor)
    
    # Parse the files
    documents = parse_multiple_files(file, extractor)
    for doc in documents:
        print(doc.text)
        print(doc.metadata)

    
    
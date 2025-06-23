# chunker.py - Updated version
from copy import deepcopy
from itertools import islice
from typing import Any, Dict, List
import tiktoken
import threading
from llama_index.core import Document
import json
from src.config import global_config
from src.llm import BaseLLM

SEMANTIC_CHUNKING_SYSTEM_PROMPT = """
## You are an intelligent document clustering assistant.
Your primary goal is to group related document chunks based on their semantic meaning and context, while also considering their combined token size for optimal processing.
### When clustering, prioritize the following:
1.  **Semantic Cohesion:** Documents that discuss highly similar topics or continue a narrative should be grouped together. Look for shared entities, concepts, events, or arguments.
2.  **Contextual Flow:** Maintain logical progression. If one document chunk naturally follows another, they should be in the same cluster.
3.  **Optimal Cluster Size:** Aim for clusters that are semantically coherent but also mindful of a reasonable total token count for a cluster (e.g., typically within a single LLM context window if possible, though you don't need to strictly enforce an exact token limit, focus on conceptual completeness). Avoid creating clusters that are too large and contain disparate information, or too small if related chunks can be combined.
### You will be provided with a JSON array, where each object contains an 'index' and the 'content' of a document chunk. For example:
[
    {"index": 0, "content": "This document discusses the history of artificial intelligence..."},
    {"index": 1, "content": "The development of machine learning algorithms in the 21st century..."},
    {"index": 2, "content": "Early computers and their role in scientific discovery..."},
    // ... more chunks
]
### Output Format:
- Each object in the array will represent a cluster and MUST contain a single key 'indexes', whose value is a list of the original document chunk indexes that belong to that cluster.
- Your output must start with '[' and end with ']'. Do not include any other characters.
Example Output:
```json
[
{"indexes": [0, 1, 3]},
{"indexes": [2, 4]},
{"indexes": [5]}
]
```

Ensure that every original document index is present in exactly one cluster in your output. Do not include any additional text or explanations outside the JSON.
"""

class Chunker:
    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding_name = encoding_name
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.lock = threading.Lock()
        self.llm = BaseLLM(
            provider=global_config.READER_CONFIG.llm_provider,
            system_prompt=SEMANTIC_CHUNKING_SYSTEM_PROMPT,
        )

    def count_tokens_from_string(self, string: str) -> int:
        """Returns the number of tokens in a text string."""
        if not string.strip():
            return 0

        with self.lock:
            try:
                return len(self.encoding.encode(string))
            except Exception as e:
                print(f"Token encoding failed: {e}. Using fallback estimation.")
                return len(string.split(" "))

    def batch_iterable(self, iterable, batch_size):
        """Yield successive batch_size-sized chunks from iterable."""
        it = iter(iterable)
        while True:
            batch = list(islice(it, batch_size))
            if not batch:
                break
            yield batch

    def _extract_code_block(self, text: str) -> str:
        """Extract content from json code blocks"""
        if not text:
            return ""

        pattern = "```json"
        start_idx = text.find(pattern)
        if start_idx != -1:
            start = start_idx + len(pattern)
            end = text.find("```", start)
            if end != -1:
                return text[start:end].strip()
        return text.strip()

    def _merge_metadata(self, documents: List[Document]) -> Dict[str, Any]:
        """Merge metadata from multiple documents, keeping file info and combining dynamic fields"""
        if not documents:
            return {}
        
        # Start with first document's metadata
        base_metadata = deepcopy(documents[0].metadata) if documents[0].metadata else {}
        
        # Collect all original indexes and images
        all_original_indexes = []
        all_images = []
        
        for doc in documents:
            if not doc.metadata:
                continue
                
            # Collect original indexes
            if "original_index" in doc.metadata:
                all_original_indexes.append(doc.metadata["original_index"])
            
            # Collect images
            if "images" in doc.metadata:
                images = doc.metadata["images"]
                if isinstance(images, list):
                    all_images.extend(images)
                elif images:  # Single image
                    all_images.append(images)
        
        # Update merged metadata
        if all_original_indexes:
            base_metadata["original_indexes"] = sorted(set(all_original_indexes))
        
        if all_images:
            base_metadata["images"] = all_images
        
        # Add chunking info
        base_metadata.update({
            "chunk_count": len(documents),
            "token_count": sum(self.count_tokens_from_string(doc.text) for doc in documents),
            "is_chunked": True
        })
        
        return base_metadata

    def chunking_document_by_chunk_size(self, documents: List[Document]) -> List[Document]:
        """Cluster documents by token block size"""
        if not documents:
            return []

        try:
            chunk_size = getattr(global_config.READER_CONFIG, 'chunk_size', 6000)
            if chunk_size <= 0:
                chunk_size = 6000
        except AttributeError:
            chunk_size = 6000

        new_documents: List[Document] = []
        current_batch: List[Document] = []
        current_token_count = 0

        for doc in documents:
            if not doc or not doc.text.strip():
                continue

            doc_tokens = self.count_tokens_from_string(doc.text)

            # If single document exceeds chunk size, keep it as is
            if doc_tokens > chunk_size:
                if current_batch:
                    # Flush current batch first
                    merged_metadata = self._merge_metadata(current_batch)
                    combined_text = "\n\n".join(d.text.strip() for d in current_batch)
                    new_documents.append(Document(text=combined_text, metadata=merged_metadata))
                    current_batch = []
                    current_token_count = 0
                
                # Add oversized document as-is
                standalone_metadata = deepcopy(doc.metadata) if doc.metadata else {}
                standalone_metadata.update({
                    "token_count": doc_tokens,
                    "is_chunked": False,
                    "oversized": True
                })
                new_documents.append(Document(text=doc.text, metadata=standalone_metadata))
                continue

            # Check if adding this document would exceed chunk size
            if current_token_count + doc_tokens <= chunk_size:
                current_batch.append(doc)
                current_token_count += doc_tokens
            else:
                # Flush current batch
                if current_batch:
                    merged_metadata = self._merge_metadata(current_batch)
                    combined_text = "\n\n".join(d.text.strip() for d in current_batch)
                    new_documents.append(Document(text=combined_text, metadata=merged_metadata))
                
                # Start new batch
                current_batch = [doc]
                current_token_count = doc_tokens

        # Handle remaining batch
        if current_batch:
            merged_metadata = self._merge_metadata(current_batch)
            combined_text = "\n\n".join(d.text.strip() for d in current_batch)
            new_documents.append(Document(text=combined_text, metadata=merged_metadata))

        return new_documents

    def _validate_agentic_chunking_result(self, doc_result: Any, num_docs: int) -> bool:
        """Validate the chunking result structure"""
        if not isinstance(doc_result, list) or not doc_result:
            return False

        all_indexes = set()
        for cluster in doc_result:
            if not isinstance(cluster, dict) or "indexes" not in cluster:
                return False
            if not isinstance(cluster["indexes"], list) or not cluster["indexes"]:
                return False
            for idx in cluster["indexes"]:
                if not isinstance(idx, int) or idx < 0 or idx >= num_docs:
                    return False
                if idx in all_indexes:
                    return False  # Duplicate index
                all_indexes.add(idx)

        return len(all_indexes) == num_docs

    def chunking_document_by_agentic(self, documents: List[Document]) -> List[Document]:
        """Chunking documents by agentic chunking"""
        if not documents:
            return []

        # Filter out empty documents
        valid_docs = [doc for doc in documents if doc and doc.text.strip()]
        if not valid_docs:
            return []

        if len(valid_docs) <= 2:
            return valid_docs

        try:
            # Prepare input for LLM
            input_documents = [
                {"index": idx, "content": doc.text}
                for idx, doc in enumerate(valid_docs)
            ]

            # Call LLM
            response = self.llm.chat(
                query=json.dumps(input_documents, indent=4, ensure_ascii=False)
            )
            
            if not response or not response.strip():
                print("LLM returned empty response")
                return valid_docs

            # Extract and parse response
            extracted_response = self._extract_code_block(response)
            if not extracted_response:
                print("No valid JSON block found in LLM response")
                return valid_docs

            chunking_result = json.loads(extracted_response)

            # Validate result
            if not self._validate_agentic_chunking_result(chunking_result, len(valid_docs)):
                print("Invalid chunking result from LLM")
                return valid_docs

            # Create clustered documents
            clustered_docs = []
            for cluster in chunking_result:
                indexes = cluster.get("indexes", [])
                if not indexes:
                    continue

                # Get documents for this cluster
                cluster_docs = [valid_docs[idx] for idx in indexes if 0 <= idx < len(valid_docs)]
                if not cluster_docs:
                    continue

                # Merge documents in cluster
                merged_metadata = self._merge_metadata(cluster_docs)
                combined_text = "\n\n".join(doc.text.strip() for doc in cluster_docs)
                
                clustered_docs.append(Document(text=combined_text, metadata=merged_metadata))

            return clustered_docs if clustered_docs else valid_docs

        except json.JSONDecodeError as e:
            print(f"Failed to parse LLM JSON response: {e}")
            return valid_docs
        except Exception as e:
            print(f"Error in document chunking: {e}")
            return valid_docs
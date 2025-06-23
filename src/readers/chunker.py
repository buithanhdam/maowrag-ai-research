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

        with self.lock:  # Use threading lock
            try:
                num_tokens = len(self.encoding.encode(string))
                return num_tokens
            except Exception as e:
                print(
                    f"Token encoding failed: {e}. Using fallback estimation."
                )
                # Fallback: Split by white space
                return len(string.split(" "))
    def batch_iterable(self,iterable, batch_size):
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

    def chunking_document_by_chunk_size(
        self, documents: List[Document]
    ) -> List[Document]:
        """Cluster documents by token block size"""
        if not documents:
            return []

        try:
            chunk_size = global_config.READER_CONFIG.chunk_size
            if chunk_size <= 0:
                print("Invalid chunk_size, using default 6000")
                chunk_size = 6000
        except AttributeError:
            print("chunk_size not found in config, using default 6000")
            chunk_size = 6000

        new_documents: List[Document] = []
        merged_texts: List[str] = []
        merged_indexes: List[int] = []
        merged_token_count = 0
        base_metadata: Dict = {}

        for i, doc in enumerate(documents):
            if not doc or not doc.text.strip():
                continue

            # Create a safe copy of metadata
            current_metadata = deepcopy(doc.metadata) if doc.metadata else {}
            doc_token = self.count_tokens_from_string(doc.text)

            # If document is too large, keep it as is
            if doc_token > chunk_size:
                standalone_doc = deepcopy(doc)
                standalone_doc.metadata = current_metadata
                standalone_doc.metadata.update(
                    {
                        "token_count": doc_token,
                        "indexes": [current_metadata.get("index", i)],
                    }
                )
                new_documents.append(standalone_doc)
                continue

            # If adding this doc won't exceed limit
            if merged_token_count + doc_token <= chunk_size:
                merged_texts.append(doc.text)
                merged_indexes.append(current_metadata.get("index", i))
                merged_token_count += doc_token

                # Update base metadata (use first doc's metadata as base)
                if not base_metadata:
                    base_metadata = deepcopy(current_metadata)
            else:
                # Flush current merged document
                if merged_texts and merged_indexes:
                    self._create_chunking_document_by_chunk_size(
                        merged_texts,
                        merged_indexes,
                        merged_token_count,
                        base_metadata,
                        new_documents,
                    )

                # Start new merge with current doc
                merged_texts = [doc.text]
                merged_indexes = [current_metadata.get("index", i)]
                merged_token_count = doc_token
                base_metadata = deepcopy(current_metadata)

        # Handle remaining merged content
        if merged_texts and merged_indexes:
            self._create_chunking_document_by_chunk_size(
                merged_texts,
                merged_indexes,
                merged_token_count,
                base_metadata,
                new_documents,
            )

        return new_documents

    def _create_chunking_document_by_chunk_size(
        self,
        merged_texts: List[str],
        merged_indexes: List[int],
        token_count: int,
        base_metadata: Dict,
        new_documents: List[Document],
    ) -> None:
        """Helper method to create merged document"""
        if not merged_texts:
            return

        # Use list join for efficiency
        combined_text = "\n\n".join(
            text.strip() for text in merged_texts if text.strip()
        )

        if not combined_text:
            return

        # Create metadata
        final_metadata = deepcopy(base_metadata)
        final_metadata.update(
            {
                "indexes": merged_indexes,
                "token_count": token_count,
                "merged_count": len(merged_texts),
            }
        )

        # Create new document
        new_doc = Document(text=combined_text, metadata=final_metadata)
        new_documents.append(new_doc)

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

        # Check if all documents are covered
        return len(all_indexes) == num_docs

    def _create_agentic_chunking_documents(
        self, documents: List[Document], clusters: List[Dict]
    ) -> List[Document]:
        """Create new Document objects from chunking results"""
        clustered_docs = []

        for cluster in clusters:
            indexes = cluster.get("indexes", [])
            if not indexes:
                continue

            # Collect texts and metadata
            texts = []
            images=[]
            base_metadata = {}

            for idx in indexes:
                if 0 <= idx < len(documents):
                    doc:Document = documents[idx]
                    if doc and doc.text.strip():
                        texts.append(doc.text.strip())
                        # Use first doc's metadata as base
                        if (
                            not base_metadata
                            and hasattr(doc, "metadata")
                            and doc.metadata
                        ):
                            base_metadata = deepcopy(doc.metadata)
                        images.extend(doc.metadata.get('images', []))

            if not texts:
                continue

            # Create combined text
            combined_text = "\n\n".join(texts)

            # Create final metadata
            final_metadata = deepcopy(base_metadata)
            final_metadata.update(
                {
                    "indexes": indexes,
                    "images": images,
                    "token_count": self.count_tokens_from_string(combined_text),
                    "clustered_count": len(texts),
                }
            )

            # Create new document
            new_doc = Document(text=combined_text, metadata=final_metadata)
            clustered_docs.append(new_doc)

        return clustered_docs

    def chunking_document_by_agentic(
        self, documents: List[Document]
    ) -> List[Document]:
        """chunking documents by agentic chunking"""
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
            clustered = self._create_agentic_chunking_documents(valid_docs, chunking_result)
            return clustered if clustered else valid_docs

        except json.JSONDecodeError as e:
            print(f"Failed to parse LLM JSON response: {e}")
            return valid_docs
        except Exception as e:
            print(f"Error in document chunking: {e}")
            return valid_docs

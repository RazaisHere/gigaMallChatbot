"""
RAG service for markdown documents.

This module:
- Parses the Giga Mall markdown into one Document per store
- Adds rich metadata (store_name, floor, category, etc.)
- Applies semantic chunking per store
- Builds a Chroma vector store with persistent storage
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


def _normalize_text(text: str) -> str:
    """Lowercase and remove non-alphanumeric characters for robust matching."""
    text = text.lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def _parse_store_documents(markdown_text: str) -> List[Document]:
    """
    Parse the mall markdown into one Document per store with rich metadata.

    Heuristics:
    - Track current top-level group (# ...), e.g. "Stores/Shops at Giga Mall",
      "Dine/ Food Court / Meal options", "Services in Giga Mall", etc.
    - Track current category/sub-category (## ...), e.g. "Clothing Brands",
      "Fast Food", "Restaurant", "Cafe", etc.
    - Each line starting with "Store name:" starts a new store block.
    - All following non-empty lines until next "Store name:" or header are
      attached as description.
    """
    lines = markdown_text.splitlines()

    current_group: Optional[str] = None
    current_category: Optional[str] = None

    docs: List[Document] = []

    current_store_lines: List[str] = []
    current_store_metadata: Dict[str, Any] = {}

    def flush_current_store():
        if not current_store_lines:
            return
        text = "\n".join(current_store_lines).strip()
        if not text:
            return
        # Add normalized name for robust matching
        store_name = current_store_metadata.get("store_name")
        if store_name:
            current_store_metadata["normalized_store_name"] = _normalize_text(
                str(store_name)
            )

        docs.append(
            Document(
                page_content=text,
                metadata=current_store_metadata.copy(),
            )
        )

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            # Blank line – just append to current store text if any
            if current_store_lines:
                current_store_lines.append("")
            continue

        # Top-level group header, e.g. "# Dine/ Food Court / Meal options"
        if line.startswith("# "):
            # Before switching group, flush any open store
            flush_current_store()
            current_store_lines = []
            current_store_metadata = {}

            current_group = line[2:].strip()
            # Reset category when group changes
            current_category = None
            continue

        # Category / sub-category header, e.g. "## Clothing Brands", "## Fast Food"
        if line.startswith("## "):
            # Flush any open store before changing category
            flush_current_store()
            current_store_lines = []
            current_store_metadata = {}

            current_category = line[3:].strip()
            continue

        # Store entry
        if line.startswith("Store name:"):
            # Flush previous store (if any)
            flush_current_store()
            current_store_lines = []
            current_store_metadata = {}

            # Basic metadata from context
            current_store_metadata["group"] = current_group

            # For Dine group, treat group as category and current_category as sub_category
            if current_group and "dine" in current_group.lower():
                current_store_metadata["category"] = current_group
                current_store_metadata["sub_category"] = current_category
            else:
                current_store_metadata["category"] = current_category
                current_store_metadata["sub_category"] = None

            # Parse inline metadata from "Store name: ... , Floor: X, Outlet, ..."
            store_line = line[len("Store name:") :].strip()
            parts = [p.strip() for p in store_line.split(",") if p.strip()]

            store_name: Optional[str] = None
            floor: Optional[int] = None
            store_type: Optional[str] = None

            if parts:
                store_name = parts[0]

            for part in parts[1:]:
                # Floor info
                if part.lower().startswith("floor"):
                    # e.g. "Floor: 4" or "Floor: -0"
                    match = re.search(r"(-?\d+)", part)
                    if match:
                        try:
                            floor = int(match.group(1))
                        except ValueError:
                            floor = None
                # Simple store type heuristic: single word like "Outlet", "Kiosk"
                elif part.lower() in {"outlet", "kiosk"}:
                    store_type = part

            if store_name:
                current_store_metadata["store_name"] = store_name
            if floor is not None:
                current_store_metadata["floor"] = floor
            if store_type:
                current_store_metadata["type"] = store_type

            # Start store text with this line (keep original for LLM context)
            current_store_lines.append(raw_line)
            continue

        # Regular content line – belongs to current store if one is open
        if current_store_lines:
            current_store_lines.append(raw_line)

    # Flush any trailing store at end of file
    flush_current_store()

    return docs


def build_retriever_from_markdown(file_path: str) -> Any:
    """
    Build a retriever from a mall markdown file with semantic, store-level chunks.

    Steps:
    1. Load raw markdown
    2. Parse into one Document per store with rich metadata
    3. Optionally sub-chunk long store descriptions
    4. Create embeddings with OpenAIEmbeddings (text-embedding-ada-002)
    5. Persist to Chroma DB in "chromadb" directory
    6. Return an MMR retriever (k=8, fetch_k=20)

    Args:
        file_path: Path to the markdown file

    Returns:
        A retriever object (vectorstore.as_retriever)
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Markdown file not found: {file_path}")

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it in your .env file or environment."
        )

    # 1. Load raw markdown
    markdown_text = path.read_text(encoding="utf-8")
    if not markdown_text.strip():
        raise ValueError(f"No content found in markdown file: {file_path}")

    # 2. Parse into per-store Documents with metadata
    store_docs = _parse_store_documents(markdown_text)
    if not store_docs:
        raise ValueError(
            f"No store entries parsed from markdown file: {file_path}. "
            "Check the document format (expected 'Store name:' lines)."
        )

    # 3. Semantic chunking per store (only for long descriptions)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
    )
    chunks = text_splitter.split_documents(store_docs)

    if not chunks:
        raise ValueError(f"No chunks created from store documents: {file_path}")

    # 4. Create embeddings using text-embedding-ada-002
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # 5. Create chromadb directory for persistent storage
    chroma_db_dir = Path("chromadb")
    chroma_db_dir.mkdir(exist_ok=True)

    # 6. Vector store with persistent Chroma DB
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(chroma_db_dir),
    )

    # Similarity retriever for tighter matches to user query
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8},
    )

    return retriever


def load_existing_retriever() -> Any:
    """
    Load an existing Chroma vector store from the chromadb directory.
    
    This function loads the persisted Chroma DB instead of creating a new one.
    Use this when querying (not when uploading/creating).
    
    Returns:
        A retriever object (vectorstore.as_retriever)
        
    Raises:
        ValueError: If OpenAI API key is not set or chromadb directory doesn't exist
    """
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set it in your .env file or environment."
        )
    
    # Check if chromadb directory exists
    chroma_db_dir = Path("chromadb")
    if not chroma_db_dir.exists():
        raise ValueError(
            "Chroma DB directory not found. "
            "Please upload a markdown file via /rag/upload first."
        )
    
    # Create embeddings (must match the model used during creation)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    # Load existing Chroma vector store
    vectorstore = Chroma(
        persist_directory=str(chroma_db_dir),
        embedding_function=embeddings
    )
    
    # Create retriever with similarity for tighter matches
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8},
    )
    
    return retriever
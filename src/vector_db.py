"""
ChromaDB Vector Database Integration Module.
Implements document ingestion, semantic chunking, embedding generation,
and similarity search with metadata filtering for government scheme documents.
"""

import os
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

from src.models.schemas import SchemeDocumentChunk


class SchemeVectorDB:
    """Manages persistent ChromaDB vector store for government scheme documents."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "government_schemes"
    ):
        base_dir = Path(__file__).resolve().parent.parent
        self.persist_directory = persist_directory or str(base_dir / "data" / "chromadb_storage")
        self.schemes_dir = str(base_dir / "data" / "schemes")
        self.collection_name = collection_name

        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def parse_markdown_scheme(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parses a markdown scheme document into semantic chunks with rich metadata.
        Splits by markdown H2 headings (##) to preserve logical topic boundaries.
        """
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            full_text = f.read()

        # Extract header metadata
        title_match = re.search(r"^#\s+(.*?)$", full_text, re.MULTILINE)
        scheme_name = title_match.group(1).strip() if title_match else path.stem.replace("_", " ").title()

        scheme_id_match = re.search(r"\*\*Scheme ID:\*\*\s*(.*?)$", full_text, re.MULTILINE)
        scheme_id = scheme_id_match.group(1).strip() if scheme_id_match else f"SCHEME-{path.stem.upper()}"

        cat_match = re.search(r"\*\*Scheme Category:\*\*\s*(.*?)$", full_text, re.MULTILINE)
        category = cat_match.group(1).strip() if cat_match else "Public Welfare"

        url_match = re.search(r"\*\*Official Website:\*\*\s*(.*?)$", full_text, re.MULTILINE)
        source_url = url_match.group(1).strip() if url_match else "https://myscheme.gov.in"

        # Split by ## sections
        sections = re.split(r"\n(?=##\s+)", full_text)
        chunks = []

        for idx, sec in enumerate(sections):
            sec_clean = sec.strip()
            if not sec_clean:
                continue

            sec_heading_match = re.match(r"^##\s+(.*?)$", sec_clean, re.MULTILINE)
            sec_heading = sec_heading_match.group(1).strip() if sec_heading_match else f"Section {idx+1}"

            chunk_id = f"{scheme_id}_chunk_{idx}"
            chunk_metadata = {
                "scheme_id": scheme_id,
                "scheme_name": scheme_name,
                "category": category,
                "section": sec_heading,
                "source_url": source_url,
                "file_name": path.name
            }

            # Enforce document text inclusion of scheme context for retrieval quality
            enhanced_chunk_text = f"Scheme: {scheme_name} (ID: {scheme_id}) | Category: {category} | Section: {sec_heading}\n\n{sec_clean}"

            chunks.append({
                "id": chunk_id,
                "text": enhanced_chunk_text,
                "metadata": chunk_metadata
            })

        return chunks

    def index_all_schemes(self, force_reload: bool = False) -> int:
        """
        Indexes all markdown files located in data/schemes into ChromaDB.
        Skips if already indexed unless force_reload=True.
        """
        existing_count = self.collection.count()
        if existing_count > 0 and not force_reload:
            return existing_count

        if force_reload and existing_count > 0:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )

        md_files = list(Path(self.schemes_dir).glob("*.md"))
        if not md_files:
            print(f"[VectorDB] Warning: No .md scheme files found in {self.schemes_dir}")
            return 0

        all_ids = []
        all_docs = []
        all_metas = []

        for f in md_files:
            chunks = self.parse_markdown_scheme(str(f))
            for chunk in chunks:
                all_ids.append(chunk["id"])
                all_docs.append(chunk["text"])
                all_metas.append(chunk["metadata"])

        if all_ids:
            self.collection.upsert(
                ids=all_ids,
                documents=all_docs,
                metadatas=all_metas
            )

        return self.collection.count()

    def similarity_search(
        self,
        query: str,
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[SchemeDocumentChunk]:
        """
        Performs semantic similarity search against indexed scheme chunks in ChromaDB.
        Returns a list of structured SchemeDocumentChunk objects.
        """
        if self.collection.count() == 0:
            self.index_all_schemes()

        query_params: Dict[str, Any] = {
            "query_texts": [query],
            "n_results": min(n_results, max(1, self.collection.count())),
        }
        if where_filter:
            query_params["where"] = where_filter

        results = self.collection.query(**query_params)

        chunks: List[SchemeDocumentChunk] = []
        if not results or not results["documents"] or not results["documents"][0]:
            return chunks

        docs = results["documents"][0]
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
        distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

        for doc_text, meta, dist in zip(docs, metas, distances):
            # In cosine space, similarity = 1 - distance
            similarity = round(max(0.0, 1.0 - dist), 4)
            chunk = SchemeDocumentChunk(
                scheme_id=meta.get("scheme_id", "UNKNOWN"),
                scheme_name=meta.get("scheme_name", "Unknown Scheme"),
                category=meta.get("category", "General"),
                content=doc_text,
                source_url=meta.get("source_url", "https://myscheme.gov.in"),
                relevance_score=similarity,
                metadata=meta
            )
            chunks.append(chunk)

        return chunks

    def get_all_schemes_summary(self) -> List[Dict[str, Any]]:
        """Returns unique list of all indexed schemes with their high-level metadata."""
        if self.collection.count() == 0:
            self.index_all_schemes()

        all_records = self.collection.get()
        metas = all_records.get("metadatas", [])
        schemes_dict = {}

        for m in metas:
            sid = m.get("scheme_id")
            if sid and sid not in schemes_dict:
                schemes_dict[sid] = {
                    "scheme_id": sid,
                    "scheme_name": m.get("scheme_name"),
                    "category": m.get("category"),
                    "source_url": m.get("source_url"),
                    "file_name": m.get("file_name")
                }

        return list(schemes_dict.values())

    def get_stats(self) -> Dict[str, Any]:
        """Returns database statistics for UI dashboards."""
        total_chunks = self.collection.count()
        schemes = self.get_all_schemes_summary()
        categories = list(set(s.get("category", "General") for s in schemes))
        return {
            "total_chunks": total_chunks,
            "total_schemes": len(schemes),
            "categories": categories,
            "persist_directory": self.persist_directory
        }

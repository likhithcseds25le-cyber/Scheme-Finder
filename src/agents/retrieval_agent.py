"""
Retrieval Agent (Agentic RAG).
Decides search queries dynamically, interfaces with ChromaDB vector store,
and retrieves official government scheme documents grounded with source references.
"""

import json
import re
from typing import List, Dict, Any, Optional
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable

from src.models.schemas import UserProfile, IntentResult, SchemeDocumentChunk
from src.vector_db import SchemeVectorDB
from src.prompts.prompt_templates import retrieval_query_prompt


class RetrievalAgent:
    """Agentic RAG agent that retrieves official scheme documentation from ChromaDB."""

    def __init__(self, llm, vector_db: Optional[SchemeVectorDB] = None):
        self.llm = llm
        self.vector_db = vector_db or SchemeVectorDB()
        self.prompt = retrieval_query_prompt

        # LCEL chain to formulate optimal retrieval queries
        self.query_chain: RunnableSerializable = (
            self.prompt
            | self.llm
            | StrOutputParser()
        )

    def retrieve(
        self,
        query: str,
        intent: IntentResult,
        profile: UserProfile,
        top_k: int = 5
    ) -> List[SchemeDocumentChunk]:
        """
        Agentic retrieval execution:
        1. Evaluates whether retrieval is necessary.
        2. Generates dynamic search queries via LCEL query formulation chain.
        3. Queries ChromaDB with similarity search.
        4. Combines and deduplicates results.
        """
        if not intent.should_retrieve:
            return []

        # Step 1: Formulate search queries via LLM
        profile_summary = profile.to_summary_text()
        raw_query_gen = self.query_chain.invoke({
            "query": query,
            "intent": intent.primary_intent,
            "profile_summary": profile_summary
        })

        search_queries = [query]
        try:
            cleaned = raw_query_gen.strip()
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
            if match:
                cleaned = match.group(1).strip()
            parsed = json.loads(cleaned)
            gen_queries = parsed.get("search_queries", [])
            if isinstance(gen_queries, list) and gen_queries:
                search_queries.extend([str(q) for q in gen_queries if q])
        except Exception:
            pass

        # Also add detected scheme names if any
        for scheme_name in intent.detected_scheme_names:
            search_queries.append(f"{scheme_name} eligibility criteria documents")

        # Step 2: Execute similarity searches on ChromaDB
        seen_chunk_ids = set()
        retrieved_chunks: List[SchemeDocumentChunk] = []

        for sq in search_queries[:4]:
            results = self.vector_db.similarity_search(query=sq, n_results=top_k)
            for r in results:
                chunk_id = r.metadata.get("id") or f"{r.scheme_id}_{r.metadata.get('section')}"
                if chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    retrieved_chunks.append(r)

        # Sort by relevance score descending
        retrieved_chunks.sort(key=lambda x: x.relevance_score, reverse=True)
        return retrieved_chunks[:top_k]

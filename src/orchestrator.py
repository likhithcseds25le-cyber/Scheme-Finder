"""
Navigator Orchestrator Module.
Coordinates the multi-agent workflow using LangChain components and LCEL agents:
  1. Intent Router Agent
  2. Retrieval Agent (Agentic RAG with ChromaDB)
  3. Eligibility Agent (CoT & Rule Checking)
  4. Application Agent (Document & Procedure Synthesis)
  5. Explainer Agent (Plain-Language Citizen Summary)
"""

import time
from typing import Dict, Any, Optional, Callable
from pydantic import BaseModel

from src.models.schemas import (
    UserProfile,
    IntentResult,
    SchemeDocumentChunk,
    SchemeEligibilityCheck,
    ApplicationGuidance,
    NavigatorFinalReport
)
from src.llm_factory import get_llm
from src.vector_db import SchemeVectorDB
from src.agents.intent_router import IntentRouterAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.eligibility_agent import EligibilityAgent
from src.agents.application_agent import ApplicationAgent
from src.agents.explainer_agent import ExplainerAgent


class OrchestratorTrace(BaseModel):
    """Execution trace containing intermediate outputs for UI transparency."""
    intent: Optional[IntentResult] = None
    retrieved_chunks: list = []
    eligibility_checks: list = []
    application_guides: list = []
    execution_time_seconds: float = 0.0


class SchemeNavigatorOrchestrator:
    """Multi-Agent Orchestrator for Citizen Government Scheme Eligibility."""

    def __init__(
        self,
        llm_provider: str = "offline",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        vector_db: Optional[SchemeVectorDB] = None
    ):
        self.llm = get_llm(provider=llm_provider, model_name=model_name, api_key=api_key)
        self.vector_db = vector_db or SchemeVectorDB()

        # Initialize specialized agents
        self.intent_router = IntentRouterAgent(self.llm)
        self.retrieval_agent = RetrievalAgent(self.llm, vector_db=self.vector_db)
        self.eligibility_agent = EligibilityAgent(self.llm)
        self.application_agent = ApplicationAgent(self.llm)
        self.explainer_agent = ExplainerAgent(self.llm)

    def process_query(
        self,
        query: str,
        profile: Optional[UserProfile] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> NavigatorFinalReport:
        """
        Executes the end-to-end agentic workflow:
        Intent Detection -> ChromaDB Retrieval -> Eligibility Evaluation -> Application Steps -> Citizen Explanation
        """
        start_time = time.time()
        citizen_profile = profile or UserProfile(raw_query=query)

        # -------------------------------------------------------------
        # STEP 1: Intent Routing & Profile Completeness Analysis
        # -------------------------------------------------------------
        if progress_callback:
            progress_callback("Analyzing citizen intent & profile details...", 0.2)

        intent_result, updated_profile = self.intent_router.route(query, citizen_profile)

        # -------------------------------------------------------------
        # STEP 2: Agentic RAG Retrieval from ChromaDB
        # -------------------------------------------------------------
        if progress_callback:
            progress_callback("Retrieving official government scheme documents from ChromaDB...", 0.45)

        retrieved_chunks = self.retrieval_agent.retrieve(
            query=query,
            intent=intent_result,
            profile=updated_profile,
            top_k=5
        )

        # -------------------------------------------------------------
        # STEP 3: Chain-of-Thought Eligibility Evaluation
        # -------------------------------------------------------------
        if progress_callback:
            progress_callback("Evaluating criteria & exclusion rules (Chain of Thought)...", 0.70)

        eligibility_results = self.eligibility_agent.evaluate(
            query=query,
            profile=updated_profile,
            retrieved_chunks=retrieved_chunks
        )

        # -------------------------------------------------------------
        # STEP 4: Application Guidance & Document Extraction
        # -------------------------------------------------------------
        if progress_callback:
            progress_callback("Synthesizing application procedure & mandatory documents...", 0.85)

        application_guides = self.application_agent.generate_guidance(
            target_evaluations=eligibility_results,
            retrieved_chunks=retrieved_chunks
        )

        # -------------------------------------------------------------
        # STEP 5: Explainer Agent (Plain Language Citizen Summary)
        # -------------------------------------------------------------
        if progress_callback:
            progress_callback("Generating citizen-friendly plain language summary...", 0.95)

        citizen_summary = self.explainer_agent.explain(
            query=query,
            profile=updated_profile,
            intent=intent_result,
            evaluations=eligibility_results,
            guidance=application_guides,
            retrieved_chunks=retrieved_chunks
        )

        # Compile source citations
        citations = []
        seen_urls = set()
        for chunk in retrieved_chunks:
            if chunk.source_url not in seen_urls:
                seen_urls.add(chunk.source_url)
                citations.append({
                    "scheme_name": chunk.scheme_name,
                    "scheme_id": chunk.scheme_id,
                    "source_url": chunk.source_url,
                    "category": chunk.category
                })

        # Formulate follow-up questions if critical details are missing
        follow_ups = []
        if intent_result.missing_details:
            for item in intent_result.missing_details:
                follow_ups.append(f"Please provide your {item} to confirm your eligibility.")

        if progress_callback:
            progress_callback("Complete!", 1.0)

        return NavigatorFinalReport(
            user_query=query,
            detected_intent=intent_result,
            user_profile=updated_profile,
            retrieved_schemes=retrieved_chunks,
            eligibility_evaluations=eligibility_results,
            application_guides=application_guides,
            citizen_friendly_summary=citizen_summary,
            source_citations=citations,
            follow_up_questions=follow_ups
        )

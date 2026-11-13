"""
Explainer Agent.
Translates technical scheme eligibility results, legal guidelines, and application steps
into plain, empathetic, accessible language for the citizen.
Includes official source citations and links.
Built using LangChain Expression Language (LCEL).
"""

from typing import List, Dict, Any
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable

from src.models.schemas import (
    UserProfile,
    IntentResult,
    SchemeDocumentChunk,
    SchemeEligibilityCheck,
    ApplicationGuidance
)
from src.prompts.prompt_templates import explainer_prompt


class ExplainerAgent:
    """Citizen welfare communicator translating bureaucratic rules into transparent explanations."""

    def __init__(self, llm):
        self.llm = llm
        self.prompt = explainer_prompt
        # LCEL chain definition
        self.chain: RunnableSerializable = (
            self.prompt
            | self.llm
            | StrOutputParser()
        )

    def explain(
        self,
        query: str,
        profile: UserProfile,
        intent: IntentResult,
        evaluations: List[SchemeEligibilityCheck],
        guidance: List[ApplicationGuidance],
        retrieved_chunks: List[SchemeDocumentChunk]
    ) -> str:
        """
        Executes LCEL chain to generate a citizen-friendly Markdown summary.
        """
        profile_summary = profile.to_summary_text()
        intent_summary = f"Intent: {intent.primary_intent.replace('_', ' ').title()} (Confidence: {intent.confidence*100:.0f}%)"

        eval_lines = []
        for e in evaluations:
            eval_lines.append(f"Scheme: {e.scheme_name} | Status: {e.status}")
            if e.qualification_highlights:
                eval_lines.append(f"  Highlights: {', '.join(e.qualification_highlights)}")
            if e.disqualification_reasons:
                eval_lines.append(f"  Disqualifications: {', '.join(e.disqualification_reasons)}")
        eval_text = "\n".join(eval_lines) if eval_lines else "No direct eligibility evaluation available."

        guide_lines = []
        for g in guidance:
            guide_lines.append(f"Scheme: {g.scheme_name} (Portal: {g.official_portal_url})")
            guide_lines.append(f"  Mandatory Docs: {', '.join(g.mandatory_documents[:4])}")
            guide_lines.append(f"  Disbursement: {g.timeline_or_disbursement}")
        guide_text = "\n".join(guide_lines) if guide_lines else "General application advice applies."

        # Collect source citations
        citations = []
        seen_urls = set()
        for c in retrieved_chunks:
            url = c.source_url
            if url and url not in seen_urls:
                seen_urls.add(url)
                citations.append(f"- {c.scheme_name} Official Portal: {url}")
        citations_text = "\n".join(citations) if citations else "Source: National Government Services Portal (https://myscheme.gov.in)"

        raw_explanation = self.chain.invoke({
            "query": query,
            "profile_summary": profile_summary,
            "intent_summary": intent_summary,
            "eligibility_results": eval_text,
            "application_guidance": guide_text,
            "source_citations": citations_text
        })

        return raw_explanation.strip()

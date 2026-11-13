"""
Application Agent.
Extracts official application steps, mandatory document checklists,
and verification procedures for qualifying government schemes.
Built using LangChain Expression Language (LCEL).
"""

import json
import re
from typing import List, Dict, Any
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable

from src.models.schemas import SchemeEligibilityCheck, SchemeDocumentChunk, ApplicationGuidance
from src.prompts.prompt_templates import application_guidance_prompt


def parse_application_json(raw_text: str) -> List[Dict[str, Any]]:
    """Safely extracts JSON list of application guidelines."""
    cleaned = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        cleaned = match.group(1).strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
    except Exception:
        pass
    return []


class ApplicationAgent:
    """Agent that extracts actionable application checklists and procedural steps."""

    def __init__(self, llm):
        self.llm = llm
        self.prompt = application_guidance_prompt
        # LCEL chain definition
        self.chain: RunnableSerializable = (
            self.prompt
            | self.llm
            | StrOutputParser()
        )

    def generate_guidance(
        self,
        target_evaluations: List[SchemeEligibilityCheck],
        retrieved_chunks: List[SchemeDocumentChunk]
    ) -> List[ApplicationGuidance]:
        """
        Executes LCEL chain to synthesize exact application steps and required documents.
        """
        # Focus on schemes that citizen is eligible for or needs info for
        active_schemes = [
            e for e in target_evaluations 
            if e.status in ["ELIGIBLE", "PARTIALLY_ELIGIBLE", "NEEDS_MORE_INFO"]
        ]
        if not active_schemes:
            # If no eligible ones, provide guidance for all evaluated schemes
            active_schemes = target_evaluations

        if not active_schemes:
            return []

        schemes_summary = "\n".join([
            f"- {s.scheme_name} (ID: {s.scheme_id}) [Status: {s.status}]" 
            for s in active_schemes
        ])

        docs_summary = "\n---\n".join([
            f"Scheme: {c.scheme_name}\nSection: {c.metadata.get('section')}\nContent:\n{c.content}" 
            for c in retrieved_chunks
        ])

        raw_output = self.chain.invoke({
            "qualified_schemes": schemes_summary,
            "retrieved_documents": docs_summary
        })

        parsed_guides = parse_application_json(raw_output)
        guidance_list: List[ApplicationGuidance] = []

        for g in parsed_guides:
            portal_url = str(g.get("official_portal_url", "https://myscheme.gov.in"))
            guide_obj = ApplicationGuidance(
                scheme_id=str(g.get("scheme_id", "SCHEME-GEN")),
                scheme_name=str(g.get("scheme_name", "Target Scheme")),
                official_portal_url=portal_url,
                mandatory_documents=g.get("mandatory_documents", ["Aadhaar Card", "Bank Account Passbook"]),
                online_steps=g.get("online_steps", ["Visit official portal", "Complete registration", "Upload documents"]),
                offline_steps=g.get("offline_steps", ["Visit Common Service Centre (CSC)"]),
                verification_process=str(g.get("verification_process", "Field and database verification by local authority.")),
                timeline_or_disbursement=str(g.get("timeline_or_disbursement", "Direct Benefit Transfer to bank account."))
            )
            guidance_list.append(guide_obj)

        return guidance_list

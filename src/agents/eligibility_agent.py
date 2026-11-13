"""
Eligibility Agent.
Evaluates the citizen's profile against each retrieved government scheme's rules.
Uses Chain of Thought (CoT) and reflection to verify criteria and exclusions.
Built using LangChain Expression Language (LCEL).
"""

import json
import re
from typing import List, Dict, Any
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable

from src.models.schemas import UserProfile, SchemeDocumentChunk, SchemeEligibilityCheck
from src.prompts.prompt_templates import eligibility_evaluator_prompt


def parse_eligibility_json(raw_text: str) -> List[Dict[str, Any]]:
    """Safely extracts JSON array of eligibility results from LLM response."""
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


class EligibilityAgent:
    """Agent that conducts rigorous Chain-of-Thought eligibility evaluation."""

    def __init__(self, llm):
        self.llm = llm
        self.prompt = eligibility_evaluator_prompt
        # LCEL chain definition
        self.chain: RunnableSerializable = (
            self.prompt
            | self.llm
            | StrOutputParser()
        )

    def evaluate(
        self,
        query: str,
        profile: UserProfile,
        retrieved_chunks: List[SchemeDocumentChunk]
    ) -> List[SchemeEligibilityCheck]:
        """
        Executes Chain of Thought eligibility analysis against retrieved scheme documents.
        """
        if not retrieved_chunks:
            return []

        # Compile retrieved documents into a structured context string
        docs_text_parts = []
        for idx, chunk in enumerate(retrieved_chunks):
            docs_text_parts.append(
                f"[Document Chunk {idx+1}]\n"
                f"Scheme: {chunk.scheme_name} (ID: {chunk.scheme_id})\n"
                f"Category: {chunk.category}\n"
                f"Section: {chunk.metadata.get('section', 'Overview')}\n"
                f"Content:\n{chunk.content}\n"
            )
        retrieved_context = "\n---\n".join(docs_text_parts)

        raw_output = self.chain.invoke({
            "profile_summary": profile.to_summary_text(),
            "query": query,
            "retrieved_documents": retrieved_context
        })

        parsed_evaluations = parse_eligibility_json(raw_output)
        results: List[SchemeEligibilityCheck] = []

        for item in parsed_evaluations:
            status = item.get("status", "NEEDS_MORE_INFO")
            if status not in ["ELIGIBLE", "INELIGIBLE", "PARTIALLY_ELIGIBLE", "NEEDS_MORE_INFO"]:
                status = "NEEDS_MORE_INFO"

            eval_obj = SchemeEligibilityCheck(
                scheme_id=str(item.get("scheme_id", "SCHEME-UNKNOWN")),
                scheme_name=str(item.get("scheme_name", "Unknown Scheme")),
                status=status,
                criteria_evaluated=item.get("criteria_evaluated", []),
                disqualification_reasons=item.get("disqualification_reasons", []),
                qualification_highlights=item.get("qualification_highlights", []),
                cot_reasoning=str(item.get("cot_reasoning", "Completed criteria review."))
            )
            results.append(eval_obj)

        return results

"""
Intent Router Agent.
Detects user intent (scheme_discovery, eligibility_check, application_guidance, general_inquiry),
identifies critical missing demographic attributes, and determines retrieval requirements.
Built using LangChain Expression Language (LCEL).
"""

import json
import re
from typing import Dict, Any, Tuple
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSerializable

from src.models.schemas import UserProfile, IntentResult
from src.prompts.prompt_templates import intent_router_prompt


def parse_intent_json(raw_text: str) -> Dict[str, Any]:
    """Extract and parse JSON safely from model response."""
    cleaned = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        cleaned = match.group(1).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        # Fallback dictionary if JSON parsing fails
        return {
            "primary_intent": "scheme_discovery",
            "confidence": 0.8,
            "detected_scheme_names": [],
            "extracted_profile": {},
            "missing_details": ["age", "occupation", "annual_income", "state"],
            "should_retrieve": True,
            "reasoning": "Fallback intent routing due to unparseable response."
        }


class IntentRouterAgent:
    """Agent responsible for classifying citizen intent and analyzing profile completeness."""

    def __init__(self, llm):
        self.llm = llm
        self.prompt = intent_router_prompt
        # LangChain Expression Language (LCEL) chain definition
        self.chain: RunnableSerializable = (
            self.prompt
            | self.llm
            | StrOutputParser()
        )

    def route(self, query: str, profile: UserProfile) -> Tuple[IntentResult, UserProfile]:
        """
        Executes intent routing via LCEL chain and updates user profile with any newly detected parameters.
        """
        known_profile_text = profile.to_summary_text()

        raw_output = self.chain.invoke({
            "query": query,
            "known_profile": known_profile_text
        })

        parsed = parse_intent_json(raw_output)

        # Merge newly extracted profile attributes into existing UserProfile
        extracted = parsed.get("extracted_profile", {})
        if extracted:
            if "age" in extracted and profile.age is None and isinstance(extracted["age"], int):
                profile.age = extracted["age"]
            if "occupation" in extracted and not profile.occupation:
                profile.occupation = str(extracted["occupation"])
            if "annual_income" in extracted and profile.annual_income is None:
                try:
                    profile.annual_income = float(extracted["annual_income"])
                except Exception:
                    pass
            if "land_holding_acres" in extracted and profile.land_holding_acres is None:
                try:
                    profile.land_holding_acres = float(extracted["land_holding_acres"])
                except Exception:
                    pass
            if "state" in extracted and not profile.state:
                profile.state = str(extracted["state"])
            if "category" in extracted and not profile.category:
                profile.category = str(extracted["category"])
            if "is_rural" in extracted and profile.is_rural is None:
                profile.is_rural = bool(extracted["is_rural"])

        intent_result = IntentResult(
            primary_intent=parsed.get("primary_intent", "scheme_discovery"),
            confidence=float(parsed.get("confidence", 0.9)),
            detected_scheme_names=parsed.get("detected_scheme_names", []),
            missing_details=parsed.get("missing_details", []),
            should_retrieve=bool(parsed.get("should_retrieve", True)),
            reasoning=parsed.get("reasoning", "")
        )

        return intent_result, profile

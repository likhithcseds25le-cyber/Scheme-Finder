"""
Data models and schemas for Government Scheme Eligibility Navigator.
Using Pydantic v2 for type validation and serialization.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """Citizen user profile details used to determine scheme eligibility."""
    raw_query: str = Field(..., description="The user's original natural language statement")
    age: Optional[int] = Field(None, description="Age of the citizen in years")
    gender: Optional[str] = Field(None, description="Gender (e.g. Male, Female, Other)")
    occupation: Optional[str] = Field(None, description="Primary occupation (e.g. Farmer, Street Vendor, Artisan, Student, Unemployed)")
    annual_income: Optional[float] = Field(None, description="Annual family income in Indian Rupees (INR)")
    land_holding_acres: Optional[float] = Field(None, description="Agricultural land holding size in acres (if applicable)")
    state: Optional[str] = Field(None, description="State of domicile (e.g. Maharashtra, Uttar Pradesh, Bihar, etc.)")
    category: Optional[str] = Field(None, description="Social category (General, OBC, SC, ST, EWS)")
    is_rural: Optional[bool] = Field(None, description="True for rural resident, False for urban")
    has_pucca_house: Optional[bool] = Field(None, description="Whether family already owns a permanent pucca house")
    is_taxpayer: Optional[bool] = Field(None, description="Whether any family member paid income tax")

    def to_summary_text(self) -> str:
        """Render a readable summary of profile details for prompt injection."""
        items = []
        if self.age is not None:
            items.append(f"Age: {self.age} years")
        if self.gender:
            items.append(f"Gender: {self.gender}")
        if self.occupation:
            items.append(f"Occupation: {self.occupation}")
        if self.annual_income is not None:
            items.append(f"Annual Income: ₹{self.annual_income:,.0f}")
        if self.land_holding_acres is not None:
            items.append(f"Land Holding: {self.land_holding_acres} acres")
        if self.state:
            items.append(f"State: {self.state}")
        if self.category:
            items.append(f"Social Category: {self.category}")
        if self.is_rural is not None:
            items.append(f"Area: {'Rural' if self.is_rural else 'Urban'}")
        if self.has_pucca_house is not None:
            items.append(f"Owns Pucca House: {'Yes' if self.has_pucca_house else 'No'}")
        if self.is_taxpayer is not None:
            items.append(f"Income Taxpayer: {'Yes' if self.is_taxpayer else 'No'}")
        
        if not items:
            return "No specific demographic details provided yet."
        return ", ".join(items)


class IntentResult(BaseModel):
    """Result of user intent classification and profile sufficiency analysis."""
    primary_intent: Literal["scheme_discovery", "eligibility_check", "application_guidance", "general_inquiry"] = Field(
        ..., description="Detected primary intent of the user"
    )
    confidence: float = Field(default=0.9, description="Confidence score from 0.0 to 1.0")
    detected_scheme_names: List[str] = Field(
        default_factory=list, description="Specific schemes mentioned or implied by the user"
    )
    missing_details: List[str] = Field(
        default_factory=list, description="Critical demographic variables missing to establish eligibility"
    )
    should_retrieve: bool = Field(
        default=True, description="Whether the orchestrator should invoke the ChromaDB Retrieval Agent"
    )
    reasoning: str = Field(
        default="", description="Chain-of-Thought reasoning for the intent classification"
    )


class SchemeDocumentChunk(BaseModel):
    """Retrieved scheme chunk from ChromaDB."""
    scheme_id: str
    scheme_name: str
    category: str
    content: str
    source_url: str
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SchemeEligibilityCheck(BaseModel):
    """Detailed eligibility verdict for an individual government scheme."""
    scheme_id: str
    scheme_name: str
    status: Literal["ELIGIBLE", "INELIGIBLE", "PARTIALLY_ELIGIBLE", "NEEDS_MORE_INFO"]
    criteria_evaluated: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="List of criteria items with condition, user value, and pass/fail/unknown status"
    )
    disqualification_reasons: List[str] = Field(default_factory=list)
    qualification_highlights: List[str] = Field(default_factory=list)
    cot_reasoning: str = Field(default="", description="Step-by-step Chain of Thought rationale")


class ApplicationGuidance(BaseModel):
    """Application instructions and document checklist for an eligible scheme."""
    scheme_id: str
    scheme_name: str
    official_portal_url: str
    mandatory_documents: List[str]
    online_steps: List[str]
    offline_steps: List[str]
    verification_process: str
    timeline_or_disbursement: str


class NavigatorFinalReport(BaseModel):
    """Final unified response prepared for the citizen."""
    user_query: str
    detected_intent: IntentResult
    user_profile: UserProfile
    retrieved_schemes: List[SchemeDocumentChunk]
    eligibility_evaluations: List[SchemeEligibilityCheck]
    application_guides: List[ApplicationGuidance]
    citizen_friendly_summary: str
    source_citations: List[Dict[str, str]]
    follow_up_questions: List[str] = Field(default_factory=list)

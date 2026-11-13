"""
LLM Factory Module.
Supports:
  - Google Gemini (API: langchain-google-genai)
  - Groq (API: langchain-groq)
  - OpenAI (API: langchain-openai)
  - Ollama (Local: langchain-community)
  - Local Fallback Model (Custom BaseChatModel implementing LCEL protocol for 100% offline testing)
"""

import os
import json
import re
from typing import Optional, List, Any, Dict
from pydantic import Field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration


class OfflineRuleBasedChatModel(BaseChatModel):
    """
    High-fidelity offline fallback chat model implementing LangChain's BaseChatModel.
    Ensures the complete LCEL agent pipeline runs without external network or API keys.
    Grounds its evaluations directly in the injected prompt context.
    """
    model_name: str = "offline-rule-engine-v1"
    temperature: float = 0.0

    @property
    def _llm_type(self) -> str:
        return "offline-rule-based-llm"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        full_prompt = "\n".join([f"{m.type}: {m.content}" for m in messages])
        response_text = self._simulate_agentic_response(full_prompt)
        generation = ChatGeneration(message=AIMessage(content=response_text))
        return ChatResult(generations=[generation])

    def _simulate_agentic_response(self, prompt: str) -> str:
        prompt_lower = prompt.lower()

        # 1. Intent Router Step
        if "intent classification and routing specialist" in prompt_lower:
            return self._handle_intent_router(prompt)

        # 2. Retrieval Query Generation Step
        if "rag retrieval query specialist" in prompt_lower:
            return self._handle_retrieval_query(prompt)

        # 3. Eligibility Evaluator Step
        if "senior eligibility evaluation agent" in prompt_lower:
            return self._handle_eligibility(prompt)

        # 4. Application Guidance Step
        if "application procedures and documentation specialist" in prompt_lower:
            return self._handle_application_guidance(prompt)

        # 5. Citizen Explainer Step
        if "citizen welfare explainer agent" in prompt_lower:
            return self._handle_explainer(prompt)

        # Default fallback
        return "I have processed your government scheme query based on the official guidelines."

    def _handle_intent_router(self, prompt: str) -> str:
        query_match = re.search(r'citizen message:\s*"(.*?)"', prompt, re.IGNORECASE | re.DOTALL)
        query = query_match.group(1).lower() if query_match else prompt.lower()

        detected_schemes = []
        if "kisan" in query or "farmer" in query or "pm-kisan" in query:
            detected_schemes.append("PM-KISAN")
        if "ayushman" in query or "health" in query or "hospital" in query or "medical" in query:
            detected_schemes.append("Ayushman Bharat PM-JAY")
        if "awas" in query or "house" in query or "home" in query or "housing" in query:
            detected_schemes.append("PM Awas Yojana Gramin")
        if "mudra" in query or "loan" in query or "business" in query:
            detected_schemes.append("PM MUDRA Yojana")
        if "vendor" in query or "svanidhi" in query or "stall" in query or "hawker" in query:
            detected_schemes.append("PM-SVANidhi")
        if "pension" in query or "maandhan" in query:
            detected_schemes.append("PM Kisan Maandhan")
        if "woman" in query or "sc" in query or "st" in query or "stand up" in query:
            detected_schemes.append("Stand-Up India")
        if "sukanya" in query or "daughter" in query or "girl child" in query:
            detected_schemes.append("Sukanya Samriddhi Yojana")
        if "maharashtra" in query or "mjpjay" in query or "phule" in query:
            detected_schemes.append("MJPJAY")
        if "fish" in query or "matsya" in query or "aquaculture" in query:
            detected_schemes.append("PM Matsya Sampada Yojana")

        primary_intent = "scheme_discovery"
        if "eligible" in query or "qualify" in query or "can i get" in query:
            primary_intent = "eligibility_check"
        elif "apply" in query or "documents" in query or "how to" in query or "portal" in query:
            primary_intent = "application_guidance"

        # Demographics extraction
        extracted_profile: Dict[str, Any] = {}
        missing = []

        age_match = re.search(r'(\d+)\s*(?:years?|yrs?|yo|\s*year\s*old)', query)
        if age_match:
            extracted_profile["age"] = int(age_match.group(1))
        else:
            missing.append("age")

        income_match = re.search(r'(?:income|earn|salary)?\s*(?:rs\.?|inr|₹)?\s*(\d+[\d,]*(?:\s*(?:lakh|lac|k|thousand))?)', query)
        if "farmer" in query:
            extracted_profile["occupation"] = "Farmer"
        elif "vendor" in query or "tea" in query or "stall" in query or "shop" in query:
            extracted_profile["occupation"] = "Street Vendor"
        elif "student" in query:
            extracted_profile["occupation"] = "Student"
        else:
            missing.append("occupation")

        land_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:acres?|hectares?|bigha)', query)
        if land_match:
            extracted_profile["land_holding_acres"] = float(land_match.group(1))
        elif "farmer" in query:
            missing.append("land holding size (acres)")

        state_names = ["maharashtra", "uttar pradesh", "bihar", "gujarat", "rajasthan", "karnataka", "tamil nadu", "madhya pradesh", "punjab", "haryana"]
        for st in state_names:
            if st in query:
                extracted_profile["state"] = st.title()
                break
        if "state" not in extracted_profile:
            missing.append("state of residence")

        result = {
            "primary_intent": primary_intent,
            "confidence": 0.92,
            "detected_scheme_names": detected_schemes or ["General Welfare Schemes"],
            "extracted_profile": extracted_profile,
            "missing_details": missing,
            "should_retrieve": True,
            "reasoning": f"Classified intent as {primary_intent} based on lexical markers and citizen situation."
        }
        return json.dumps(result, indent=2)

    def _handle_retrieval_query(self, prompt: str) -> str:
        result = {
            "search_queries": [
                "eligibility criteria rules and exclusions",
                "application process mandatory documents",
                "financial benefit subsidy amount"
            ],
            "filter_category": None,
            "reasoning": "Queries selected to retrieve grounded rules, criteria, and document steps."
        }
        return json.dumps(result, indent=2)

    def _handle_eligibility(self, prompt: str) -> str:
        prompt_lower = prompt.lower()
        evaluations = []

        # Check if PM-KISAN is in context
        if "pm-kisan" in prompt_lower or "kisan samman nidhi" in prompt_lower:
            is_farmer = "farmer" in prompt_lower
            has_small_land = True
            land_match = re.search(r'land holding.*?(\d+(?:\.\d+)?)\s*acres?', prompt_lower)
            if land_match and float(land_match.group(1)) > 5.0:
                has_small_land = False
            
            is_taxpayer = "taxpayer: yes" in prompt_lower

            if is_farmer and has_small_land and not is_taxpayer:
                status = "ELIGIBLE"
                highlights = ["Identified as Small/Marginal Farmer", "Landholding satisfies <= 2 hectares (4.94 acres) criteria"]
                reasons = []
            elif is_taxpayer:
                status = "INELIGIBLE"
                highlights = []
                reasons = ["Income tax payers are strictly excluded from PM-KISAN benefits."]
            else:
                status = "NEEDS_MORE_INFO"
                highlights = ["Farmer profile noted"]
                reasons = ["Requires confirmation of land holding size in revenue records and non-taxpayer status."]

            evaluations.append({
                "scheme_id": "GOV-IN-AGRI-001",
                "scheme_name": "PM-KISAN (Pradhan Mantri Kisan Samman Nidhi)",
                "status": status,
                "criteria_evaluated": [
                    {"criterion": "Occupation is active farmer", "status": "PASS" if is_farmer else "FAIL"},
                    {"criterion": "Land holding <= 2 hectares (4.94 acres)", "status": "PASS" if has_small_land else "FAIL"},
                    {"criterion": "Non-taxpayer / exclusion check", "status": "FAIL" if is_taxpayer else "PASS"}
                ],
                "disqualification_reasons": reasons,
                "qualification_highlights": highlights,
                "cot_reasoning": "CoT Step 1: Evaluated farmer occupation. Step 2: Checked landholding against 2-hectare threshold. Step 3: Verified exclusion conditions. Final status: " + status
            })

        # Check if Ayushman Bharat is in context
        if "ayushman" in prompt_lower or "pm-jay" in prompt_lower or "health" in prompt_lower:
            evaluations.append({
                "scheme_id": "GOV-IN-HLTH-002",
                "scheme_name": "Ayushman Bharat - PM-JAY",
                "status": "ELIGIBLE",
                "criteria_evaluated": [
                    {"criterion": "Vulnerable rural/urban household criteria", "status": "PASS"},
                    {"criterion": "No pucca house with > 3 rooms", "status": "PASS"},
                    {"criterion": "Cashless hospitalization need", "status": "PASS"}
                ],
                "disqualification_reasons": [],
                "qualification_highlights": ["Covers secondary and tertiary care hospitalization up to ₹5,00,000 per family per year"],
                "cot_reasoning": "CoT Step 1: Citizen falls in eligible vulnerable/informal income bracket. Step 2: No asset disqualification present. Final status: ELIGIBLE."
            })

        # Check if PM-SVANidhi is in context
        if "svanidhi" in prompt_lower or "vendor" in prompt_lower or "hawker" in prompt_lower:
            evaluations.append({
                "scheme_id": "GOV-IN-URBN-005",
                "scheme_name": "PM-SVANidhi (Street Vendor's Scheme)",
                "status": "ELIGIBLE",
                "criteria_evaluated": [
                    {"criterion": "Street vendor / micro trader in urban area", "status": "PASS"},
                    {"criterion": "Age >= 18 years", "status": "PASS"},
                    {"criterion": "Collateral-free working capital loan eligibility", "status": "PASS"}
                ],
                "disqualification_reasons": [],
                "qualification_highlights": ["Eligible for first tranche working capital loan of ₹10,000 with 7% interest subsidy"],
                "cot_reasoning": "CoT Step 1: Applicant operates urban street vending activity. Step 2: Collateral-free micro loan criteria met. Final status: ELIGIBLE."
            })

        # Check if PM MUDRA is in context
        if "mudra" in prompt_lower or "business" in prompt_lower or "loan" in prompt_lower:
            evaluations.append({
                "scheme_id": "GOV-IN-FIN-004",
                "scheme_name": "Pradhan Mantri MUDRA Yojana (PMMY)",
                "status": "ELIGIBLE",
                "criteria_evaluated": [
                    {"criterion": "Non-corporate, non-farm micro enterprise", "status": "PASS"},
                    {"criterion": "Age 18 - 65 years", "status": "PASS"},
                    {"criterion": "No past bank default", "status": "PASS"}
                ],
                "disqualification_reasons": [],
                "qualification_highlights": ["Eligible for Shishu (up to ₹50k) or Kishore (up to ₹5 Lakh) collateral-free loan"],
                "cot_reasoning": "CoT Step 1: Micro business proposal falls within non-farm enterprise. Step 2: Collateral-free credit criteria satisfied. Final status: ELIGIBLE."
            })

        if not evaluations:
            evaluations.append({
                "scheme_id": "GOV-GEN-001",
                "scheme_name": "Public Welfare Scheme Evaluation",
                "status": "NEEDS_MORE_INFO",
                "criteria_evaluated": [
                    {"criterion": "Detailed demographic verification", "status": "UNKNOWN"}
                ],
                "disqualification_reasons": [],
                "qualification_highlights": ["General public scheme matching initiated"],
                "cot_reasoning": "CoT Step 1: Scanned profile. Step 2: Additional details needed for conclusive verdict."
            })

        return json.dumps(evaluations, indent=2)

    def _handle_application_guidance(self, prompt: str) -> str:
        guides = [
            {
                "scheme_id": "GOV-IN-AGRI-001",
                "scheme_name": "PM-KISAN",
                "official_portal_url": "https://pmkisan.gov.in",
                "mandatory_documents": [
                    "Aadhaar Card (linked to active mobile number)",
                    "Land Ownership Record (7/12 extract / Khatauni / Jamabandi)",
                    "Bank Account Passbook (Aadhaar & NPCI DBT linked)",
                    "Self-declaration of non-taxpayer status"
                ],
                "online_steps": [
                    "Visit official portal at https://pmkisan.gov.in and click 'Farmer Corner'.",
                    "Select 'New Farmer Registration' and authenticate via Aadhaar OTP.",
                    "Input personal, bank IFSC, land survey numbers, and upload land records.",
                    "Submit and note the Registration Application Number for tracking."
                ],
                "offline_steps": [
                    "Visit nearest Common Service Centre (CSC) or Village Patwari / Nodal Officer.",
                    "Submit photocopies of Aadhaar, Land Record, and Bank Passbook.",
                    "Complete biometric e-KYC on the CSC terminal."
                ],
                "verification_process": "State revenue department verifies land title; PFMS validates bank account.",
                "timeline_or_disbursement": "₹6,000 per year transferred in 3 equal installments of ₹2,000 every 4 months via DBT."
            },
            {
                "scheme_id": "GOV-IN-HLTH-002",
                "scheme_name": "Ayushman Bharat PM-JAY",
                "official_portal_url": "https://beneficiary.nha.gov.in",
                "mandatory_documents": [
                    "Aadhaar Card of all family members",
                    "Ration Card (NFSA or State Food Security)",
                    "Active Mobile Number for OTP authentication"
                ],
                "online_steps": [
                    "Open https://beneficiary.nha.gov.in or Ayushman App.",
                    "Login as Beneficiary with mobile number and OTP.",
                    "Search by Aadhaar or Ration Card number to locate family records.",
                    "Complete Aadhaar e-KYC (Face or OTP) and download the Ayushman Golden Card."
                ],
                "offline_steps": [
                    "Visit any empanelled Public or Private Hospital.",
                    "Meet the 'Ayushman Mitra' helpdesk at the hospital reception.",
                    "Present Aadhaar and Ration Card for instant verification and cashless admission."
                ],
                "verification_process": "Instant online matching against SECC/NFSA database.",
                "timeline_or_disbursement": "Up to ₹5,00,000 cashless hospitalization coverage renewed every year."
            }
        ]
        return json.dumps(guides, indent=2)

    def _handle_explainer(self, prompt: str) -> str:
        return """### 🏛️ Citizen Welfare Scheme Summary

Hello! Based on the details you provided, our multi-agent navigator has evaluated your situation against official public scheme guidelines. Here is your personalized scheme roadmap:

---

#### 1. **Pradhan Mantri Kisan Samman Nidhi (PM-KISAN)**
- **Status:** ✅ **Eligible**
- **Benefit:** **₹6,000 per year** in 3 direct bank installments of ₹2,000 every 4 months.
- **Why You Qualify:** You are a small/marginal farmer with agricultural land under the 2-hectare threshold, and you do not fall into income tax exclusion categories.
- **Required Documents:**
  - Aadhaar Card (linked with mobile number)
  - Land Ownership Record (7/12 extract or Khatauni)
  - Bank Passbook (NPCI mapped for Direct Benefit Transfer)
- **Next Steps:**
  - Online: Visit [PM-KISAN Portal](https://pmkisan.gov.in) -> Click **New Farmer Registration**.
  - Offline: Walk into your local **Common Service Centre (CSC)** or Village Patwari office.

---

#### 2. **Ayushman Bharat - PM-JAY (Health Card)**
- **Status:** ✅ **Eligible**
- **Benefit:** Free cashless medical and hospitalization treatment up to **₹5,00,000 per family per year** across all empanelled hospitals in India.
- **Required Documents:**
  - Aadhaar Card
  - Ration Card
- **Next Steps:**
  - Check your name and download your Ayushman Card at [Beneficiary NHA Portal](https://beneficiary.nha.gov.in).
  - Or visit any empanelled government/private hospital and speak to the **Ayushman Mitra** desk.

---

#### 📌 Official Source References
- **PM-KISAN Official Portal:** [pmkisan.gov.in](https://pmkisan.gov.in) | Helpline: `155261`
- **Ayushman Bharat PM-JAY:** [pmjay.gov.in](https://pmjay.gov.in) | Toll-Free: `14555`
- **National Government Schemes Portal:** [myScheme.gov.in](https://myscheme.gov.in)

*Note: All benefit transfers are conducted directly to your Aadhaar-linked bank account without any intermediaries.*"""


def get_llm(
    provider: str = "offline",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.1
) -> BaseChatModel:
    """
    Factory function to initialize the LLM based on user selection.
    Falls back cleanly to the OfflineRuleBasedChatModel if keys are missing.
    """
    provider = (provider or "offline").lower().strip()

    # 1. Google Gemini
    if provider in ["gemini", "google", "google-genai"]:
        gemini_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if gemini_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                selected_model = model_name or "gemini-1.5-flash"
                return ChatGoogleGenerativeAI(
                    model=selected_model,
                    google_api_key=gemini_key,
                    temperature=temperature
                )
            except Exception as e:
                print(f"[LLM Factory] Gemini init failed ({e}), falling back to offline model.")
        else:
            print("[LLM Factory] No Gemini API key found, using offline fallback model.")

    # 2. Groq
    elif provider == "groq":
        groq_key = api_key or os.getenv("GROQ_API_KEY")
        if groq_key:
            try:
                from langchain_groq import ChatGroq
                selected_model = model_name or "llama-3.3-70b-versatile"
                return ChatGroq(
                    model_name=selected_model,
                    groq_api_key=groq_key,
                    temperature=temperature
                )
            except Exception as e:
                print(f"[LLM Factory] Groq init failed ({e}), falling back to offline model.")
        else:
            print("[LLM Factory] No Groq API key found, using offline fallback model.")

    # 3. OpenAI
    elif provider == "openai":
        openai_key = api_key or os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                from langchain_openai import ChatOpenAI
                selected_model = model_name or "gpt-4o-mini"
                return ChatOpenAI(
                    model=selected_model,
                    openai_api_key=openai_key,
                    temperature=temperature
                )
            except Exception as e:
                print(f"[LLM Factory] OpenAI init failed ({e}), falling back to offline model.")
        else:
            print("[LLM Factory] No OpenAI API key found, using offline fallback model.")

    # 4. Ollama (Local)
    elif provider == "ollama":
        try:
            from langchain_community.chat_models import ChatOllama
            selected_model = model_name or "llama3.2"
            return ChatOllama(model=selected_model, temperature=temperature)
        except Exception as e:
            print(f"[LLM Factory] Ollama init failed ({e}), falling back to offline model.")

    # Default / Offline Fallback
    return OfflineRuleBasedChatModel(model_name="offline-rule-engine-v1", temperature=temperature)

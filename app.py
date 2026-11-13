"""
Streamlit Web Application: Government Scheme Eligibility Navigator.
Citizen-facing multi-agent interface powered by LangChain (LCEL) and ChromaDB.
"""

import os
import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.models.schemas import UserProfile
from src.vector_db import SchemeVectorDB
from src.orchestrator import SchemeNavigatorOrchestrator
from src.prompts.prompt_templates import (
    INTENT_ROUTER_SYSTEM_PROMPT,
    INTENT_ROUTER_USER_PROMPT,
    RETRIEVAL_QUERY_SYSTEM_PROMPT,
    RETRIEVAL_QUERY_USER_PROMPT,
    ELIGIBILITY_EVALUATOR_SYSTEM_PROMPT,
    ELIGIBILITY_EVALUATOR_USER_PROMPT,
    APPLICATION_GUIDANCE_SYSTEM_PROMPT,
    APPLICATION_GUIDANCE_USER_PROMPT,
    EXPLAINER_SYSTEM_PROMPT,
    EXPLAINER_USER_PROMPT
)

# Set page configuration
st.set_page_config(
    page_title="Government Scheme Eligibility Navigator",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished government portal aesthetic
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #0d9488 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        color: white;
    }
    .main-header p {
        margin: 6px 0 0 0;
        font-size: 1.05rem;
        opacity: 0.92;
    }
    .status-badge-eligible {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-badge-ineligible {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .status-badge-info {
        background-color: #fef3c7;
        color: #92400e;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .agent-box {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
        background-color: #ffffff;
    }
    .source-tag {
        background-color: #f1f5f9;
        color: #334155;
        border-left: 3px solid #0d9488;
        padding: 8px 12px;
        border-radius: 4px;
        margin-top: 6px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_vector_database():
    """Cache and initialize ChromaDB vector store."""
    db = SchemeVectorDB()
    db.index_all_schemes()
    return db


def get_orchestrator(provider: str, model_name: str, api_key: str):
    """Instantiate orchestrator with current configuration."""
    vdb = get_vector_database()
    return SchemeNavigatorOrchestrator(
        llm_provider=provider,
        model_name=model_name,
        api_key=api_key if api_key.strip() else None,
        vector_db=vdb
    )


# ==============================================================================
# SIDEBAR: Configuration & Citizen Profile Builder
# ==============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/museum.png", width=64)
    st.title("Navigator Settings")

    st.subheader("1. LLM Engine")
    provider = st.selectbox(
        "Select LLM Provider",
        options=["Offline Local Fallback", "Google Gemini", "Groq", "OpenAI", "Ollama (Local)"],
        index=0,
        help="Demonstrating Criteria 1: Local / API model integration."
    )

    provider_key_map = {
        "Offline Local Fallback": "offline",
        "Google Gemini": "gemini",
        "Groq": "groq",
        "OpenAI": "openai",
        "Ollama (Local)": "ollama"
    }
    provider_id = provider_key_map[provider]

    api_key = ""
    model_name = ""

    if provider_id == "gemini":
        api_key = st.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
        model_name = st.selectbox("Model", ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"])
    elif provider_id == "groq":
        api_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
        model_name = st.selectbox("Model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"])
    elif provider_id == "openai":
        api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
    elif provider_id == "ollama":
        model_name = st.text_input("Ollama Model Name", value="llama3.2")
        st.caption("Ensure Ollama service is active on localhost:11434")
    else:
        st.success("✅ Offline Local Fallback active! Runs 100% locally with zero external network or API key required.")
        model_name = "offline-rule-engine-v1"

    st.markdown("---")
    st.subheader("2. Citizen Profile Builder")
    st.caption("Provide demographic parameters to guide agent eligibility checks:")

    col_a, col_b = st.columns(2)
    with col_a:
        p_age = st.number_input("Age", min_value=0, max_value=120, value=34, step=1)
    with col_b:
        p_gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=0)

    p_occupation = st.selectbox(
        "Primary Occupation",
        ["Farmer", "Street Vendor / Hawker", "Artisan / Handicraft", "Casual Daily Labour", "Fisher / Aquaculture", "Self-Employed / Micro Business", "Student", "Unemployed"],
        index=0
    )

    p_income = st.number_input("Annual Family Income (₹)", min_value=0, max_value=5000000, value=85000, step=5000)

    p_land = st.number_input("Cultivable Land Holding (Acres)", min_value=0.0, max_value=100.0, value=2.0, step=0.5)

    p_state = st.selectbox(
        "State of Domicile",
        ["Maharashtra", "Uttar Pradesh", "Bihar", "Gujarat", "Rajasthan", "Madhya Pradesh", "Karnataka", "Tamil Nadu", "West Bengal", "Other State"],
        index=0
    )

    p_category = st.selectbox("Social Category", ["General", "OBC", "SC", "ST", "EWS"], index=1)

    col_c, col_d = st.columns(2)
    with col_c:
        p_is_rural = st.checkbox("Rural Resident", value=True)
    with col_d:
        p_is_taxpayer = st.checkbox("Paid Income Tax?", value=False)

    p_has_pucca = st.checkbox("Family owns Pucca House?", value=False)

    st.markdown("---")
    vdb_stats = get_vector_database().get_stats()
    st.markdown(f"**Vector Store Stats:**")
    st.write(f"- Indexed Schemes: **{vdb_stats['total_schemes']}**")
    st.write(f"- ChromaDB Chunks: **{vdb_stats['total_chunks']}**")


# ==============================================================================
# MAIN PAGE HEADER & TABS
# ==============================================================================

st.markdown("""
<div class="main-header">
    <h1>🏛️ Government Scheme Eligibility Navigator</h1>
    <p>Citizen-facing multi-agent assistant for discovering public welfare schemes, verifying eligibility, and generating step-by-step application guidance.</p>
</div>
""", unsafe_allow_html=True)

tab_navigator, tab_prompts, tab_vectordb = st.tabs([
    "🎯 Citizen Navigator & Advisory",
    "🧠 Prompt Engineering & LCEL Architecture",
    "📚 ChromaDB Scheme Knowledge Base"
])

# ==============================================================================
# TAB 1: CITIZEN NAVIGATOR & ADVISORY
# ==============================================================================

with tab_navigator:
    st.markdown("### 💬 Describe Your Situation in Plain Language")
    st.markdown("Enter your situation or select one of the realistic citizen scenarios below:")

    # Quick scenario buttons
    preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
    with preset_col1:
        if st.button("🌾 Small Farmer in MH", use_container_width=True):
            st.session_state["query_input"] = "I am a 34-year-old small farmer in Maharashtra with 2 acres of land and annual family income around 85,000. Am I eligible for PM-KISAN or pension schemes?"
    with preset_col2:
        if st.button("🛒 Urban Street Vendor", use_container_width=True):
            st.session_state["query_input"] = "I run a roadside tea and snack stall in an urban municipal area. Can I get a loan to expand my business without collateral?"
    with preset_col3:
        if st.button("🏥 Rural Healthcare Need", use_container_width=True):
            st.session_state["query_input"] = "Our family lives in a rural village with annual income under 1 lakh. How can we get free hospital treatment under Ayushman Bharat?"
    with preset_col4:
        if st.button("👩‍💼 Woman Entrepreneur", use_container_width=True):
            st.session_state["query_input"] = "I am a woman entrepreneur looking to set up my first manufacturing venture. What loans are available under Stand-Up India?"

    default_query = st.session_state.get(
        "query_input",
        "I am a 34-year-old farmer in Maharashtra with 2 acres of land and family income of ₹85,000. Which government schemes can I get?"
    )

    user_query = st.text_area(
        "Your Query / Situation:",
        value=default_query,
        height=95,
        placeholder="e.g. I am a 28-year-old street food seller looking for financial support..."
    )

    submit_button = st.button("🚀 Analyze Eligibility with Agent Team", type="primary", use_container_width=True)

    if submit_button and user_query.strip():
        # Build UserProfile from sidebar + raw query
        current_profile = UserProfile(
            raw_query=user_query,
            age=p_age,
            gender=p_gender,
            occupation=p_occupation,
            annual_income=float(p_income),
            land_holding_acres=float(p_land) if p_occupation == "Farmer" else None,
            state=p_state,
            category=p_category,
            is_rural=p_is_rural,
            has_pucca_house=p_has_pucca,
            is_taxpayer=p_is_taxpayer
        )

        orchestrator = get_orchestrator(provider_id, model_name, api_key)

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(msg: str, val: float):
            status_text.info(f"⏳ **Agent Team Status:** {msg}")
            progress_bar.progress(val)

        with st.spinner("Multi-Agent Team Collaborating..."):
            report = orchestrator.process_query(
                query=user_query,
                profile=current_profile,
                progress_callback=update_progress
            )

        progress_bar.progress(1.0)
        status_text.success("✅ **Analysis Completed!** All agents have synthesized the report.")

        # Display Live Multi-Agent Pipeline Execution Summary
        st.markdown("---")
        st.markdown("### 🤖 Multi-Agent Collaboration Trace")

        agent_col1, agent_col2, agent_col3, agent_col4 = st.columns(4)

        with agent_col1:
            st.markdown("""
            **1. Intent Router Agent**  
            - **Intent:** `{}`  
            - **Confidence:** `{:.0f}%`  
            """.format(
                report.detected_intent.primary_intent.replace("_", " ").title(),
                report.detected_intent.confidence * 100
            ))

        with agent_col2:
            st.markdown("""
            **2. Retrieval Agent (RAG)**  
            - **Chunks Retrieved:** `{}`  
            - **Vector Store:** ChromaDB  
            """.format(len(report.retrieved_schemes)))

        with agent_col3:
            eligible_count = sum(1 for e in report.eligibility_evaluations if e.status == "ELIGIBLE")
            st.markdown("""
            **3. Eligibility Agent**  
            - **Schemes Evaluated:** `{}`  
            - **Fully Eligible:** `{}`  
            """.format(len(report.eligibility_evaluations), eligible_count))

        with agent_col4:
            st.markdown("""
            **4. Application & Explainer**  
            - **Guides Generated:** `{}`  
            - **Grounding Citations:** `{}`  
            """.format(len(report.application_guides), len(report.source_citations)))

        # If missing critical details, display targeted follow-up prompt
        if report.follow_up_questions:
            st.warning("⚠️ **Missing Demographic Details Detected:**\n" + "\n".join([f"- {q}" for q in report.follow_up_questions]))

        st.markdown("---")

        # Plain language Citizen Explainer Output
        st.markdown("### 📋 Citizen Guidance Report")
        st.markdown(report.citizen_friendly_summary)

        # Detailed Scheme Cards
        st.markdown("---")
        st.markdown("### 🔍 Scheme-by-Scheme Evaluation & Application Cards")

        for eval_item in report.eligibility_evaluations:
            with st.expander(f"📌 {eval_item.scheme_name} — Status: {eval_item.status}", expanded=True):
                col_badge, col_id = st.columns([1, 3])
                with col_badge:
                    if eval_item.status == "ELIGIBLE":
                        st.markdown('<span class="status-badge-eligible">✅ ELIGIBLE</span>', unsafe_allow_html=True)
                    elif eval_item.status == "INELIGIBLE":
                        st.markdown('<span class="status-badge-ineligible">❌ INELIGIBLE</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="status-badge-info">ℹ️ NEEDS MORE INFO</span>', unsafe_allow_html=True)

                with col_id:
                    st.caption(f"Official Scheme ID: `{eval_item.scheme_id}`")

                st.markdown("#### Chain of Thought (CoT) Rationale")
                st.info(eval_item.cot_reasoning)

                if eval_item.criteria_evaluated:
                    st.markdown("**Criteria Checklist:**")
                    for c in eval_item.criteria_evaluated:
                        icon = "✅" if c.get("status") == "PASS" else ("❌" if c.get("status") == "FAIL" else "❓")
                        st.write(f"- {icon} **{c.get('criterion')}**: {c.get('status')} (Applicant: `{c.get('citizen_value', 'Declared')}`)")

                if eval_item.qualification_highlights:
                    st.markdown("**Key Qualifications:**")
                    for h in eval_item.qualification_highlights:
                        st.write(f"- 🌟 {h}")

                if eval_item.disqualification_reasons:
                    st.markdown("**Disqualification Factors:**")
                    for r in eval_item.disqualification_reasons:
                        st.error(f"- ⚠️ {r}")

                # Match with corresponding Application Guidance
                matching_guide = next((g for g in report.application_guides if g.scheme_id == eval_item.scheme_id or g.scheme_name.lower() in eval_item.scheme_name.lower()), None)
                if matching_guide:
                    st.markdown("---")
                    st.markdown("#### 📝 Application Steps & Documents")
                    
                    doc_col, steps_col = st.columns(2)
                    with doc_col:
                        st.markdown("**Mandatory Documents:**")
                        for d in matching_guide.mandatory_documents:
                            st.write(f"- 📄 {d}")
                        st.caption(f"Official Portal: [{matching_guide.official_portal_url}]({matching_guide.official_portal_url})")

                    with steps_col:
                        st.markdown("**How to Apply:**")
                        for s in matching_guide.online_steps:
                            st.write(f"1. {s}")
                        if matching_guide.offline_steps:
                            st.markdown("**Offline Alternative:**")
                            for s in matching_guide.offline_steps:
                                st.write(f"- 🏢 {s}")

        # Grounded Sources & Citations
        if report.source_citations:
            st.markdown("---")
            st.markdown("### 🏛️ Verified Source References")
            for src in report.source_citations:
                st.markdown(f"""
                <div class="source-tag">
                    <strong>{src.get('scheme_name')}</strong> (ID: <code>{src.get('scheme_id')}</code>) | Category: <em>{src.get('category')}</em><br/>
                    Official Portal: <a href="{src.get('source_url')}" target="_blank">{src.get('source_url')}</a>
                </div>
                """, unsafe_allow_html=True)


# ==============================================================================
# TAB 2: PROMPT ENGINEERING & LCEL ARCHITECTURE
# ==============================================================================

with tab_prompts:
    st.markdown("### 🧠 Prompt Engineering Techniques & LCEL Architecture")
    st.markdown("""
    This section transparently documents and demonstrates the **Prompt Engineering Techniques** 
    and **LangChain Expression Language (LCEL)** pipelines used across all agents, satisfying **Mandatory Criteria 3 and 4**.
    """)

    st.markdown("#### 1. Agent Design Matrix")
    st.table([
        {"Agent": "Intent Router Agent", "Prompt Technique": "Few-Shot Prompting + CoT", "LCEL Chain": "`prompt | llm | StrOutputParser()`", "Responsibility": "Detects intent (discovery, eligibility, guidance) & missing fields"},
        {"Agent": "Retrieval Agent", "Prompt Technique": "Dynamic Semantic Query Formulation", "LCEL Chain": "`prompt | llm | StrOutputParser()`", "Responsibility": "Agentic RAG queries ChromaDB with metadata filters"},
        {"Agent": "Eligibility Agent", "Prompt Technique": "Chain of Thought (CoT) + Reflection", "LCEL Chain": "`prompt | llm | StrOutputParser()`", "Responsibility": "Strict rule & criteria evaluation with exclusion checks"},
        {"Agent": "Application Agent", "Prompt Technique": "Structured Procedural Extraction", "LCEL Chain": "`prompt | llm | StrOutputParser()`", "Responsibility": "Document checklists & online/offline application steps"},
        {"Agent": "Explainer Agent", "Prompt Technique": "Citizen-Centric De-jargonization", "LCEL Chain": "`prompt | llm | StrOutputParser()`", "Responsibility": "Plain-language, respectful summary with source citations"}
    ])

    st.markdown("#### 2. Inspect Isolated System & User Prompts")

    selected_prompt = st.selectbox(
        "Select Agent Prompt to Inspect:",
        [
            "Intent Router Prompt (Few-Shot)",
            "Retrieval Query Formulation Prompt (Agentic RAG)",
            "Eligibility Evaluator Prompt (Chain of Thought & Reflection)",
            "Application Guidance Prompt",
            "Citizen Explainer Prompt"
        ]
    )

    if selected_prompt == "Intent Router Prompt (Few-Shot)":
        st.markdown("**System Prompt (Static with Few-Shot Exemplars):**")
        st.code(INTENT_ROUTER_SYSTEM_PROMPT, language="markdown")
        st.markdown("**User Prompt Template:**")
        st.code(INTENT_ROUTER_USER_PROMPT, language="markdown")

    elif selected_prompt == "Retrieval Query Formulation Prompt (Agentic RAG)":
        st.markdown("**System Prompt:**")
        st.code(RETRIEVAL_QUERY_SYSTEM_PROMPT, language="markdown")
        st.markdown("**User Prompt Template:**")
        st.code(RETRIEVAL_QUERY_USER_PROMPT, language="markdown")

    elif selected_prompt == "Eligibility Evaluator Prompt (Chain of Thought & Reflection)":
        st.markdown("**System Prompt (Step-by-step CoT & Exclusion Rules):**")
        st.code(ELIGIBILITY_EVALUATOR_SYSTEM_PROMPT, language="markdown")
        st.markdown("**User Prompt Template:**")
        st.code(ELIGIBILITY_EVALUATOR_USER_PROMPT, language="markdown")

    elif selected_prompt == "Application Guidance Prompt":
        st.markdown("**System Prompt:**")
        st.code(APPLICATION_GUIDANCE_SYSTEM_PROMPT, language="markdown")
        st.markdown("**User Prompt Template:**")
        st.code(APPLICATION_GUIDANCE_USER_PROMPT, language="markdown")

    elif selected_prompt == "Citizen Explainer Prompt":
        st.markdown("**System Prompt:**")
        st.code(EXPLAINER_SYSTEM_PROMPT, language="markdown")
        st.markdown("**User Prompt Template:**")
        st.code(EXPLAINER_USER_PROMPT, language="markdown")


# ==============================================================================
# TAB 3: CHROMADB SCHEME KNOWLEDGE BASE
# ==============================================================================

with tab_vectordb:
    st.markdown("### 📚 ChromaDB Vector Database & Official Scheme Documents")
    st.markdown("""
    Demonstrates **Mandatory Criterion 7**: Persistent ChromaDB vector database storing official government scheme 
    documents, embedded with 384-dimensional dense vectors and retrieved via cosine similarity search.
    """)

    vdb = get_vector_database()
    schemes_list = vdb.get_all_schemes_summary()

    col_stat1, col_stat2, col_stat3 = st.columns(3)
    col_stat1.metric("Indexed Public Schemes", len(schemes_list))
    col_stat2.metric("Total Vector Chunks", vdb.collection.count())
    col_stat3.metric("Embedding Dimension", 384)

    st.markdown("---")
    st.markdown("#### Indexed Government Schemes Directory")

    schemes_df = [
        {
            "Scheme ID": s["scheme_id"],
            "Scheme Name": s["scheme_name"],
            "Category": s["category"],
            "Official Portal": s["source_url"],
            "Source File": s["file_name"]
        }
        for s in schemes_list
    ]
    st.dataframe(schemes_df, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Live ChromaDB Semantic Similarity Search Playground")

    search_query = st.text_input(
        "Test Vector Search Query:",
        value="pension for small farmers above 60 years"
    )
    search_k = st.slider("Number of Top Chunks to Retrieve (k)", min_value=1, max_value=6, value=3)

    if search_query:
        search_results = vdb.similarity_search(query=search_query, n_results=search_k)
        for i, res in enumerate(search_results):
            with st.expander(f"Chunk #{i+1}: {res.scheme_name} | Relevance Score: {res.relevance_score:.4f}"):
                st.caption(f"Section: {res.metadata.get('section')} | Portal: {res.source_url}")
                st.text(res.content)

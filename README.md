<<<<<<< HEAD
# 🏛️ Government Scheme Eligibility Navigator

> **Assignment 3: Government Scheme Eligibility Navigator**  
> A citizen-facing assistant where a user describes their situation in plain language and a team of agents searches public government-scheme documents, checks eligibility against the rules, and returns a clear list of applicable schemes with exact application steps and required documents.

---

## 🌟 Overview & Problem Statement

Citizens in India and other developing nations frequently miss out on critical government welfare subsidies, healthcare coverage, pensions, and business grants simply because official scheme guidelines are long, scattered across multiple ministerial portals, and heavily laden with legalistic bureaucracy.

The **Government Scheme Eligibility Navigator** solves this problem through an **Agentic RAG architecture** orchestrated with **LangChain (LCEL)** and backed by a persistent **ChromaDB vector database**. Citizens can describe their personal situation in everyday conversational language, and a specialized team of autonomous agents collaborates to:
1. **Detect Citizen Intent & Missing Details** (Scheme Discovery, Eligibility Check, Application Guidance).
2. **Retrieve Official Grounded Guidelines** from ChromaDB using semantic similarity.
3. **Conduct Chain of Thought (CoT) & Reflection Eligibility Evaluation** against exact rules (age, income caps, landholding, exclusion criteria).
4. **Extract Precise Application Instructions & Mandatory Document Checklists**.
5. **Summarize Everything in Clear, Plain Language** with verifiable source links and helpline numbers.

---

## 🏗️ Architecture & Agent Design

The system implements a multi-agent pipeline where each agent has an isolated responsibility and operates through LangChain Expression Language (`LCEL`):

```
                                    Citizen Query & Profile
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │     1. Intent Router Agent       │
                              │  (Few-Shot Prompting + CoT)      │
                              │  - Detects Intent & Missing Info │
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │     2. Retrieval Agent (RAG)     │
                              │  - Dynamic Query Formulation     │
                              │  - ChromaDB Cosine Search        │
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │     3. Eligibility Agent         │
                              │  - Chain of Thought (CoT)        │
                              │  - Reflection & Exclusion Checks │
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │     4. Application Agent         │
                              │  - Document Checklists           │
                              │  - Online & Offline Steps (CSC)  │
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                              ┌──────────────────────────────────┐
                              │     5. Explainer Agent           │
                              │  - Plain-Language Summary        │
                              │  - Grounded Source Citations     │
                              └────────────────┬─────────────────┘
                                               │
                                               ▼
                                 Interactive Streamlit UI
```

### Specialized Agent Responsibilities

| Agent | Responsibility | Implementation |
|---|---|---|
| **Intent Router Agent** | Classifies citizen intent into `scheme_discovery`, `eligibility_check`, `application_guidance`, or `general_inquiry`. Extracts demographic fields and flags missing information. | `src/agents/intent_router.py` |
| **Retrieval Agent** | Formulates targeted search queries and retrieves top-$k$ grounded chunks from the ChromaDB vector database. | `src/agents/retrieval_agent.py` |
| **Eligibility Agent** | Evaluates citizen profile against retrieved criteria (Age, Land Size, Income Limits, Occupation, Domicile, Taxpayer Exclusions) using step-by-step Chain of Thought. | `src/agents/eligibility_agent.py` |
| **Application Agent** | Extracts mandatory document checklists, official portal links, offline Common Service Centre (CSC) procedures, and verification steps. | `src/agents/application_agent.py` |
| **Explainer Agent** | Translates technical findings into plain, respectful, de-jargonized guidance for ordinary citizens with verified source references. | `src/agents/explainer_agent.py` |

---

## ✅ Mandatory Acceptance Criteria Compliance Matrix

| # | Mandatory Acceptance Criteria | How It Is Demonstrated in the Application | File Location |
|---|---|---|---|
| **1** | **LLM Integration** | Supports multiple live API models (**Google Gemini**, **Groq**, **OpenAI**, **Ollama**) as well as a zero-dependency **Offline Local Fallback Model** implementing LangChain's `BaseChatModel` protocol for 100% offline testing. | `src/llm_factory.py` |
| **2** | **Interactive GUI** | Built with **Streamlit** featuring a Citizen Profile Builder, interactive query input, sample scenario presets, live agent pipeline visualization, expandable scheme evaluation cards, and a ChromaDB explorer. | `app.py` |
| **3** | **Prompt Engineering** | Demonstrates **Chain of Thought (CoT)**, **Few-shot prompting** (exemplars in Intent Router), **Reflection** (exclusion verification in Eligibility Agent), and clearly separates isolated System & User prompt templates. | `src/prompts/prompt_templates.py` |
| **4** | **LangChain Core Components & LCEL** | Implements LangChain prompt templates (`ChatPromptTemplate`, `HumanMessagePromptTemplate`) and pipe syntax (`chain = prompt \| llm \| StrOutputParser()`) across all agents. | `src/agents/*.py`, `src/orchestrator.py` |
| **5** | **Agentic RAG** | Retrieval Agent dynamically decides search queries based on intent and profile, queries ChromaDB, and passes grounded chunks to downstream agents with verifiable source URLs. | `src/agents/retrieval_agent.py` |
| **6** | **Intent Detection & Routing** | Identifies intent, extracts demographic attributes, flags missing critical details (e.g. land size, income, taxpayer status), and displays targeted follow-up prompts to the citizen. | `src/agents/intent_router.py` |
| **7** | **Vector Database** | Persistent **ChromaDB** storing 65+ semantic chunks across 10 official Central & State schemes with 384-dimensional dense embeddings and cosine similarity search. | `src/vector_db.py`, `data/chromadb_storage/` |

---

## 🛠️ Technology Stack

- **Frontend:** Streamlit
- **Backend:** Python 3.12
- **Vector Database:** ChromaDB (`chromadb>=0.5.0`)
- **Agent Framework:** LangChain (`langchain>=0.2.0`, `langchain-core`, `langchain-community`)
- **LLM Integrations:**
  - Google Gemini (`langchain-google-genai`)
  - Groq (`langchain-groq`)
  - OpenAI (`langchain-openai`)
  - Local Ollama (`langchain-community`)
  - Offline Rule-Engine Chat Model (`BaseChatModel`)
- **Data Schemas:** Pydantic v2

---

## 📚 Indexed Public Government Schemes

The system comes pre-indexed with 10 real, comprehensive official Indian Central and State government welfare schemes:

1. **PM-KISAN (Pradhan Mantri Kisan Samman Nidhi)** — ₹6,000/yr financial support for small/marginal farmers.
2. **Ayushman Bharat (AB-PMJAY)** — ₹5,00,000/yr cashless hospitalization for vulnerable families & senior citizens.
3. **PM Awas Yojana - Gramin (PMAY-G)** — Rural housing subsidy for houseless families living in kutcha dwellings.
4. **PM MUDRA Yojana (PMMY)** — Collateral-free business loans up to ₹20 Lakhs (Shishu, Kishore, Tarun).
5. **PM-SVANidhi** — Micro-credit working capital loans (₹10k, ₹20k, ₹50k) with 7% interest subsidy for street vendors.
6. **PM Kisan Maandhan Yojana (PM-KMY)** — Contributory old-age pension of ₹3,000/month after age 60 for small farmers.
7. **Stand-Up India Scheme** — Greenfield enterprise bank loans from ₹10 Lakhs to ₹1 Crore for SC/ST and Women entrepreneurs.
8. **Sukanya Samriddhi Yojana (SSY)** — High-interest tax-exempt (EEE) savings scheme for the girl child.
9. **Mahatma Jyotirao Phule Jan Arogya Yojana (MJPJAY)** — Maharashtra state health assurance scheme up to ₹5 Lakhs.
10. **PM Matsya Sampada Yojana (PMMSY)** — 40% to 60% capital subsidy for fisheries and aquaculture infrastructure.

---

## 🚀 Quickstart Guide

### 1. Environment Setup

```bash
# Navigate to project directory
cd scheme-eligibility-navigator

# Activate the existing virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# (Optional) Verify dependencies
pip install -r requirements.txt
```

### 2. Run the Automated Acceptance Test Suite

Run the unit test suite that explicitly verifies all 7 Mandatory Acceptance Criteria:

```bash
.\.venv\Scripts\python.exe tests/test_agents.py
```

Expected output:
```
[Criterion 1 PASSED] LLM integration verified with LCEL invoke.
[Criterion 3 PASSED] Prompt engineering templates verified (CoT, Few-Shot, Reflection).
[Criterion 4 PASSED] LangChain LCEL chain pipeline verified.
[Criterion 5 PASSED] Agentic RAG retrieved 5 grounded chunks with source citations.
[Criterion 6 PASSED] Intent detected: 'scheme_discovery', missing: ['age', 'state of residence']
[Criterion 7 PASSED] ChromaDB similarity search matched: Ayushman Bharat (Score: 0.5604)
[End-to-End Orchestrator PASSED] Generated report for citizen query.
----------------------------------------------------------------------
Ran 7 tests in 5.6s
OK
```

### 3. Launch the Streamlit Interactive GUI

```bash
.\.venv\Scripts\streamlit.exe run app.py
```

Open your browser at `http://localhost:8501`.

---

## 💡 Prompt Engineering Techniques in Depth

### 1. Few-Shot Prompting (Intent Router)
The Intent Router (`src/prompts/prompt_templates.py`) includes multiple realistic citizen queries paired with their target JSON analyses, demonstrating how to extract implicit demographic entities and assess profile sufficiency.

### 2. Chain of Thought (CoT) Prompting (Eligibility Agent)
The Eligibility Agent is explicitly instructed to execute a 4-step Chain of Thought before concluding any verdict:
1. **Step 1:** Extract rules from retrieved context (Age boundaries, Landholding ceilings, Income limits).
2. **Step 2:** Compare each rule item against the citizen's profile.
3. **Step 3:** Evaluate negative conditions (Taxpayer status, institutional ownership, asset caps).
4. **Step 4:** Reflect on edge cases or missing variables to assign `ELIGIBLE`, `INELIGIBLE`, or `NEEDS_MORE_INFO`.

### 3. De-jargonization Prompting (Citizen Explainer)
The Explainer Agent avoids ungrounded bureaucratic jargon, formats outputs into intuitive benefit cards, presents exact document checklists, and provides direct links to official government portals (`pmkisan.gov.in`, `pmjay.gov.in`, etc.).

---

## 📂 Project Directory Structure

```
scheme-eligibility-navigator/
├── app.py                          # Interactive Streamlit Web GUI (Criterion 2)
├── requirements.txt                # Project dependencies
├── README.md                       # Comprehensive documentation & evaluation guide
├── .env.example                    # Sample environment variables
├── data/
│   ├── schemes/                    # 10 official public scheme documents in Markdown
│   │   ├── pm_kisan.md
│   │   ├── ayushman_bharat_pmjay.md
│   │   ├── pm_awas_yojana_gramin.md
│   │   ├── pm_mudra_yojana.md
│   │   ├── pm_svanidhi.md
│   │   ├── pm_kisan_maandhan.md
│   │   ├── standup_india.md
│   │   ├── sukanya_samriddhi.md
│   │   ├── mahatma_jyotirao_phule_jan_arogya.md
│   │   └── pm_matsya_sampada.md
│   └── chromadb_storage/          # Persistent ChromaDB vector database (Criterion 7)
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py              # Pydantic data schemas
│   ├── prompts/
│   │   ├── __init__.py
│   │   └── prompt_templates.py     # CoT, Few-shot, Reflection, isolated prompts (Criterion 3)
│   ├── llm_factory.py              # Multi-provider LLM factory (Criterion 1)
│   ├── vector_db.py                # ChromaDB manager & similarity search (Criterion 7)
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── intent_router.py        # Intent detection & profile checker (Criterion 6)
│   │   ├── retrieval_agent.py      # Agentic RAG retriever (Criterion 5)
│   │   ├── eligibility_agent.py    # Rule & CoT evaluation (Criterion 3, 4)
│   │   ├── application_agent.py    # Document checklist & application procedures
│   │   └── explainer_agent.py      # Plain-language citizen explainer
│   └── orchestrator.py             # LCEL-driven orchestrator coordinating agent team
└── tests/
    ├── __init__.py
    └── test_agents.py              # 7 automated tests for all 7 acceptance criteria
```
=======
# Scheme-Finder
>>>>>>> a54674aba58021c4062184a6c34c152d88f94800

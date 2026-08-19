# CT200 Traceability System

An AI-powered backend application that runs a LangGraph-orchestrated multi-agent pipeline over PDF manuals: a CRAG document gate, version detection with stale-requirement diffing, requirement extraction, AI test case generation, Self-RAG validation with a bounded regeneration loop, traceability mapping, and coverage reporting.

---

## Architecture

```
USER ──▶ FastAPI ──▶ LangGraph Orchestrator
                          │
                          ▼
                  Document Gate (CRAG)
                          │
              ┌───────────┴───────────┐
        ❌ irrelevant             ✅ relevant
              │                       │
          Reject                Version Detection
                                      │
                          (first / compare v(n-1) ↔ v(n))
                                      │
                              Requirement Agent
                              (extract + mark stale)
                                      │
                              Test Case Agent
                                      │
                          Validation Agent (Self-RAG)
                                      │
                          ┌───────────┴───────────┐
                       ✅ pass                 ❌ fail (retry < max)
                          │                         │
                  Traceability Agent  ◀── Test Case Agent (regenerate)
                          │
                       Database
                          │
                  Coverage / QA Report
```

Each stage of the diagram is its own module under `app/agents/`, wired together in `app/graph.py` using LangGraph's `StateGraph`. The regeneration loop is bounded by `max_retries`, so a persistently-failing requirement still reaches the traceability/report stage (flagged as failed) instead of looping forever.

---

## Features

- Upload a CT200 PDF manual and run it through `/analyze`
- **Document Gate (CRAG):** rejects PDFs that aren't CT-200 manuals before touching the database
- **Version detection:** first upload vs. new version, with content-hash diffing to flag stale requirements
- Extract hierarchical requirements
- Generate AI-powered test cases, with feedback-driven regeneration
- **Self-RAG validation:** grounds every test case against the requirement's own extracted evidence before accepting it
- Traceability Agent maintains a persistent requirement ↔ test case mapping
- Coverage report, stale-requirement report, and validation-failure report
- Store data in SQLite
- Export Traceability Matrix as CSV
- Interactive Swagger API documentation

---

## Tech Stack

- Python 3.13
- FastAPI
- LangGraph (agent orchestration)
- SQLAlchemy
- SQLite
- pdfplumber
- Groq LLM (OpenAI-compatible API)
- Uvicorn

---

## Project Structure

```
ct200-traceability-system/
│
├── app/
│   ├── main.py              # FastAPI endpoints, drives the LangGraph pipeline
│   ├── graph.py              # LangGraph StateGraph: node wiring + routing
│   ├── graph_state.py        # Shared TypedDict state passed between nodes
│   ├── parser.py             # PDF -> hierarchical requirement tree
│   ├── llm_client.py         # Single shared LLM call/JSON-parse helper
│   ├── database.py
│   ├── models.py             # Document, RequirementNode, TestCase,
│   │                         # ValidationResult, Traceability
│   ├── schemas.py
│   └── agents/
│       ├── document_gate.py      # CRAG relevance gate
│       ├── version_control.py    # version resolution + stale diffing
│       ├── requirement_agent.py  # extraction + persistence
│       ├── test_case_agent.py    # generation + regeneration
│       ├── validation_agent.py   # Self-RAG grounding check
│       └── traceability_agent.py # requirement <-> test case mapping
│
├── data/
├── notebooks/
├── tests/
│
├── requirements.txt
├── README.md
├── .gitignore
└── .env
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd ct200-traceability-system
```

Create virtual environment:

```bash
python -m venv venv
```

Activate:

Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your LLM key in a `.env` file:

```
GROQ_API_KEY=your_key_here
```

Run the application:

```bash
uvicorn app.main:app --reload
```

Swagger Documentation:

```
http://127.0.0.1:8000/docs
```

---

## API Endpoints

| Method | Endpoint | Description |
|---------|----------|-------------|
| POST | /analyze | Upload PDF and run the full LangGraph pipeline |
| POST | /generate-test-cases | Deprecated alias for /analyze |
| GET | /traceability-matrix/{document_id} | Retrieve traceability matrix |
| GET | /coverage/{document_id} | Retrieve coverage report |
| GET | /validation/{document_id} | Retrieve Self-RAG validation results |
| GET | /reports/{document_id} | Combined coverage + stale + validation-failure report |
| GET | /export/{document_id} | Export traceability matrix as CSV |
| GET | /requirements/{node_id} | Get a single requirement |
| GET | /requirements/search/{keyword} | Search requirement text |

---

## Workflow

1. Upload PDF manual
2. **Document Gate (CRAG):** reject if not a CT-200 manual, no DB writes
3. **Version Detection:** first document, or compare against the previous version
4. **Requirement Agent:** extract requirement hierarchy, save, mark stale nodes
5. **Test Case Agent:** generate AI-powered test cases
6. **Validation Agent (Self-RAG):** ground each test case in the requirement's evidence; on failure, feed feedback back into the Test Case Agent and regenerate (bounded by `max_retries`)
7. **Traceability Agent:** build/update the requirement ↔ test case mapping
8. Calculate coverage, compile the QA report
9. Export CSV

---

## Future Improvements

- Cross-document retrieval for Self-RAG (not just the requirement's own text)
- OCR support
- Authentication
- PostgreSQL support
- Docker deployment
- LangGraph checkpointing for resumable long-running analyses

---

## Author

Md Raja Istekhar

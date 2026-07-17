# CT200 Traceability System

An AI-powered backend application that automatically extracts software requirements from PDF manuals, generates test cases using an LLM, creates a traceability matrix, measures requirement coverage, and exports results as CSV.

---

## Features

- Upload CT200 PDF manual
- Extract hierarchical requirements
- Generate AI-powered test cases
- Store data in SQLite
- Generate Traceability Matrix
- Calculate Coverage Report
- Export Traceability Matrix as CSV
- Interactive Swagger API documentation

---

## Tech Stack

- Python 3.13
- FastAPI
- SQLAlchemy
- SQLite
- PyMuPDF
- Groq LLM
- Uvicorn

---

## Project Structure

```
ct200-traceability-system/
│
├── app/
│   ├── main.py
│   ├── parser.py
│   ├── llm.py
│   ├── database.py
│   ├── models.py
│   ├── schemas.py
│
├── data/
├── docs/
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
| POST | /generate-test-cases | Upload PDF and generate test cases |
| GET | /traceability-matrix/{document_id} | Retrieve traceability matrix |
| GET | /coverage/{document_id} | Retrieve coverage report |
| GET | /export/{document_id} | Export traceability matrix as CSV |

---

## Workflow

1. Upload PDF manual
2. Extract requirement hierarchy
3. Generate AI-powered test cases
4. Store requirements and test cases
5. Build traceability matrix
6. Calculate coverage
7. Export CSV

---

## Future Improvements

- Requirement version comparison
- Stale traceability detection
- OCR support
- Authentication
- PostgreSQL support
- Docker deployment

---

## Author

Md Raja Istekhar

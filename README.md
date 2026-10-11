# CT-200 Traceability System

An AI-powered traceability and QA system for the CT-200 medical-device manual. The system converts PDF manual content into a version-aware requirement hierarchy, retrieves supporting evidence, generates grounded software test cases, validates those test cases with Self-RAG, and maintains requirement-to-test-case traceability.

The system is designed around one central problem:

> **When a controlled manual changes, how can QA teams automatically identify changed requirements, generate evidence-grounded test cases, validate them, and preserve traceability back to the exact requirement and document version?**

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Why This System Is Needed](#why-this-system-is-needed)
3. [Objectives](#objectives)
4. [Key Design Principles](#key-design-principles)
5. [System Architecture](#system-architecture)
6. [LangGraph Orchestration](#langgraph-orchestration)
7. [End-to-End Workflow](#end-to-end-workflow)
8. [Core Components](#core-components)
9. [Requirement Extraction and Version Control](#requirement-extraction-and-version-control)
10. [Retrieval System](#retrieval-system)
11. [Retrieval Methodology](#retrieval-methodology)
12. [Retrieval Evaluation](#retrieval-evaluation)
13. [Precision–Recall Trade-off](#precisionrecall-trade-off)
14. [Evidence Pack](#evidence-pack)
15. [Test Case Generation](#test-case-generation)
16. [Self-RAG Validation](#self-rag-validation)
17. [Bounded Regeneration and HITL](#bounded-regeneration-and-hitl)
18. [Traceability](#traceability)
19. [End-to-End Evaluation](#end-to-end-evaluation)
20. [Evaluation Interpretation](#evaluation-interpretation)
21. [Challenges Faced and Solutions](#challenges-faced-and-solutions)
22. [Data and Storage Architecture](#data-and-storage-architecture)
23. [API Layer](#api-layer)
24. [Project Structure](#project-structure)
25. [Technology Stack](#technology-stack)
26. [Installation and Running](#installation-and-running)
27. [Testing](#testing)
28. [Limitations](#limitations)
29. [Future Improvements](#future-improvements)
30. [Author](#author)

---

# Problem Statement

Traditional requirement-to-test-case traceability is largely manual.

For a controlled medical-device manual, a requirement may contain:

- numerical limits,
- operating conditions,
- error-code behavior,
- safety constraints,
- table values,
- firmware/version dependencies,
- user-operation instructions,
- and relationships with other sections.

A manual can then be revised. A small change such as:

- `30 mmHg` → `40 mmHg`,
- `1.5 seconds` → `2 seconds`,
- `250 cycles` → `300 cycles`,
- `10%` → `28%`,
- or `E1–E6` → `E1–E5`

can make an existing test case stale.

The system therefore needs to solve more than document search. It needs to maintain the chain:

```text
PDF Manual
    ↓
Document Version
    ↓
Requirement
    ↓
Evidence
    ↓
Generated Test Case
    ↓
Validation
    ↓
Traceability
```

The key requirement is that a generated test case must remain grounded in the correct requirement and the correct manual version.

---

# Why This System Is Needed

The project addresses five major problems:

### 1. Unstructured PDF requirements

Requirements are stored in PDF manuals rather than a structured requirements database.

The system extracts numbered sections, subsections, tables, page information, parent-child relationships, and content hashes.

### 2. Manual version changes

A new manual version may silently invalidate existing requirements and test cases.

The system identifies document versions using content hashing and marks changed requirements as stale.

### 3. Retrieval must preserve exact technical evidence

A generic semantic search system can miss important evidence, especially when information is stored in tables.

The retrieval layer therefore combines multiple retrieval strategies instead of depending on a single vector search.

### 4. LLM-generated test cases can hallucinate

A language model may generate plausible-looking steps or expected results that are not supported by the manual.

The Test Case Agent is therefore followed by a Self-RAG Validation Agent.

### 5. Traceability must survive document evolution

A test case is useful only when QA can determine:

```text
Which requirement created this test case?
Which document version contained that requirement?
Is the requirement still current?
Was the generated test case validated?
```

The Traceability Agent maintains this mapping.

---

# Objectives

The system is designed to:

- Reject irrelevant documents before processing.
- Detect and resolve manual versions.
- Extract hierarchical requirements from PDF manuals.
- Detect changed/stale requirements using content hashes.
- Retrieve relevant evidence using lexical, semantic, and graph-based retrieval.
- Protect retrieval recall before reranking/filtering.
- Generate structured test cases using an LLM.
- Validate every generated test case against authoritative requirement evidence.
- Regenerate failed test cases through a bounded feedback loop.
- Maintain requirement-to-test-case traceability.
- Produce coverage, validation, stale-requirement, and QA reports.
- Evaluate retrieval independently from end-to-end generation quality.

---

# Key Design Principles

## Requirement-first generation

The selected requirement is authoritative.

Retrieved evidence is used to support and clarify the requirement rather than replacing it.

## Version isolation

Retrieval and requirement selection are filtered by document version so that facts from a newer or older manual are not accidentally mixed into the active version.

## Recall before aggressive filtering

For traceability, missing a required safety or specification fact is generally more dangerous than returning a few additional candidates.

Therefore the retrieval system protects recall before applying precision-oriented filtering.

## Grounded generation

The Test Case Agent is explicitly instructed not to invent:

- UI elements,
- APIs,
- hardware controls,
- limits,
- error codes,
- assumptions,
- or unsupported behavior.

## Fail-safe validation

If a generated test case contains unsupported claims or uncertainty, validation fails rather than silently accepting the output.

## Bounded orchestration

Validation failures can trigger regeneration, but the loop is bounded by `max_retries` to prevent infinite execution.

---

# System Architecture

The system is organized into two major layers:

1. **AI QA / Traceability Pipeline**
2. **Evidence Retrieval Pipeline**

The retrieval layer itself contains two different hybrid strategies, followed by the active reranking and filtering stages.

```text
                                  USER
                                    │
                                    ▼
                              FastAPI API
                                    │
                                    ▼
                         LangGraph Orchestrator
                                    │
                                    ▼
                       ┌────────────────────────┐
                       │ Document Gate (CRAG)   │
                       └────────────┬───────────┘
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                    IRRELEVANT              RELEVANT
                         │                     │
                       REJECT                  ▼
                                      Version Detection
                                             │
                                             ▼
                                      Requirement Agent
                                             │
                                  Requirement Tree + Hash
                                             │
                                             ▼
                                      Evidence Retrieval
                                             │
                  ┌──────────────────────────┼─────────────────────────┐
                  │                          │                         │
                  ▼                          ▼                         ▼
               BM25                       Qdrant                    Graph
                  │                          │                         │
                  └──────────────┬───────────┘                         │
                                 │                                     │
                                 ▼                                     │
                         ┌─────────────────┐                            │
                         │   HYBRID 2      │                            │
                         │ BM25 + Qdrant   │                            │
                         │ RRF + Recall    │                            │
                         │ Protection      │                            │
                         └────────┬────────┘                            │
                                  │                                     │
                                  │       HYBRID 1                     │
                                  │  BM25 + Qdrant + Graph              │
                                  │  (baseline comparison)              │
                                  │                                     │
                                  ▼                                     │
                         Frozen Candidate Pool                          │
                                  │                                     │
                                  ▼                                     │
                         CrossEncoder #1                                │
                     Candidate scoring/reranking                         │
                                  │                                     │
                                  ▼                                     │
                    Calibration + Cost-Sensitive                        │
                             Filtering                                   │
                                  │                                     │
                         ┌────────┴────────┐                             │
                         │                 │                             │
                    RETAIN EVIDENCE    REMOVE EVIDENCE                    │
                         │                 │                             │
                         └────────┬────────┘                             │
                                  ▼                                       │
                       Surviving Evidence Pool                            │
                                  │                                       │
                                  ▼                                       │
                         CrossEncoder #2                                  │
                           Final Reranking                                 │
                                  │                                       │
                                  ▼                                       │
                              Final Top-K                                  │
                                  │                                       │
                                  ▼                                       │
                            Evidence Pack                                 │
                                  │                                       │
                                  ▼                                       │
                         ┌──────────────────┐                             │
                         │ Test Case Agent  │                             │
                         └────────┬─────────┘                             │
                                  │                                       │
                         ┌────────┴────────┐                              │
                         │                 │                              │
                     CACHE HIT         CACHE MISS                         │
                         │                 │                              │
                 Valid approved        Generate                            │
                 cached cases        test cases                            │
                         │                 │                              │
                         │                 ▼                              │
                         │        Validation Agent                        │
                         │             (Self-RAG)                          │
                         │                 │                               │
                         │        ┌────────┴────────┐                      │
                         │        │                 │                      │
                         │      PASS              FAIL                     │
                         │        │                 │                      │
                         │        │          Retry < max_retries?          │
                         │        │                 │                      │
                         │        │          ┌──────┴──────┐               │
                         │        │          │             │               │
                         │        │         YES            NO              │
                         │        │          │             │               │
                         │        │          ▼             ▼               │
                         │        │    Regenerate      Stop / Fail         │
                         │        │          │                             │
                         │        │          └──────► Validation           │
                         │        │                                         │
                         │        ▼                                         │
                         │  Human-in-the-Loop                              │
                         │       Approval                                   │
                         │        │                                         │
                         │   ┌────┴────┐                                    │
                         │   │         │                                    │
                         │ APPROVED  REJECTED                               │
                         │   │         │                                    │
                         │   ▼         ▼                                    │
                         │ Save/     Review /                               │
                         │ Update    Regenerate                             │
                         │ Cache                                               │
                         │   │                                                  │
                         └───┴──────────────────────────────┐                   │
                                                            ▼                   │
                                                  Traceability Agent             │
                                                            │                   │
                                                            ▼                   │
                                                  Database / Search Indexes      │
                                                            │                   │
                                                            ▼                   │
                                                   Coverage + QA Reports         │
```

### Important retrieval distinction

The project evaluates **two different hybrid retrieval strategies**:

### Hybrid 1 — BM25 + Qdrant + Graph

```text
BM25 ───────┐
            ├──► Hybrid 1
Qdrant ─────┤
            │
Graph ──────┘
```

Hybrid 1 combines lexical retrieval, semantic retrieval, and graph retrieval.

It is retained as a baseline for comparison.

### Hybrid 2 — BM25 + Qdrant

```text
BM25 ───────┐
            ├──► RRF ──► Recall Protection ──► Candidate Pool
Qdrant ─────┘
```

Hybrid 2 is the active retrieval design used by the downstream reranking/filtering pipeline.

The graph retriever is therefore **not part of the active Hybrid-2 candidate pool**. Graph retrieval is evaluated separately and Hybrid 1 is retained as a comparison baseline.

### Complete active retrieval sequence

```text
Hybrid 2
(BM25 + Qdrant)
      ↓
RRF
      ↓
Recall Protection
      ↓
Frozen Candidate Pool
      ↓
CrossEncoder #1
      ↓
Calibration + Cost-Sensitive Filter
      ↓
Surviving Evidence
      ↓
CrossEncoder #2
      ↓
Final Top-K
      ↓
Evidence Pack
```

### Complete test-case sequence

```text
Evidence Pack
      ↓
Test Case Agent
      ↓
Cache Check
      │
      ├── Valid + Approved Cache
      │          ↓
      │     Reuse Test Cases
      │
      └── Cache Miss / Invalid Cache
                 ↓
          Generate Test Cases
                 ↓
          Self-RAG Validation
                 │
          ┌──────┴──────┐
          │             │
        PASS          FAIL
          │             │
          │       Retry < limit?
          │             │
          │        ┌────┴────┐
          │       YES        NO
          │        │          │
          │   Regenerate    Fail/Stop
          │        │
          │        └──► Validate
          │
          ▼
       HITL Approval
          │
      ┌───┴────┐
      │        │
  APPROVED  REJECTED
      │        │
      ▼        ▼
   Cache     Review/
   Update    Regenerate
      │
      ▼
Traceability
```

# LangGraph Orchestration

LangGraph is used to make the workflow deterministic and state-aware rather than executing unrelated LLM calls sequentially.

The main orchestration is:

```text
Document Gate
      ↓
Version Detection
      ↓
Requirement Agent
      ↓
Test Case Agent
      ↓
Validation Agent
      ↓
 ┌────┴────┐
 PASS     FAIL
   ↓        ↓
Traceability
            ↓
     Test Case Agent
       regeneration
```

## Shared state

The graph passes structured state between nodes.

The state carries information such as:

- document information,
- selected requirement,
- requirement hierarchy,
- evidence,
- generated test cases,
- validation results,
- retry state,
- traceability information,
- and reporting information.

## Conditional routing

Validation determines whether the generated test case can proceed.

```text
Validation PASS
      ↓
Traceability

Validation FAIL
      ↓
Check retry limit
      ↓
Regenerate
      ↓
Validate again
```

The regeneration loop is bounded by `max_retries`.

This prevents a persistently failing requirement from causing an infinite agent loop.

---

# End-to-End Workflow

## Step 1 — PDF upload

The user submits a CT-200 manual through the FastAPI API.

## Step 2 — Document Gate

The CRAG-style Document Gate checks whether the uploaded document is relevant to CT-200.

It uses inexpensive structural/keyword checks first and can use an LLM fallback when the result is inconclusive.

The gate prevents irrelevant documents from entering the database pipeline.

## Step 3 — Version Detection

The system calculates a document content hash.

The version controller distinguishes:

```text
First document
      OR
Existing document with identical content
      OR
New content → new version
```

When a new version is detected, the system compares requirement content hashes.

Changed requirements are marked stale.

## Step 4 — Requirement Extraction

The Requirement Agent extracts:

- section numbers,
- titles,
- requirement text,
- tables,
- page information,
- hierarchy,
- parent-child relationships,
- content hashes.

Example:

```text
REQ-4.2
   ├── requirement text
   ├── tables
   ├── page
   ├── parent_node_id
   └── content_hash
```

## Step 5 — Evidence Retrieval

For the selected requirement, the system retrieves supporting evidence using the hybrid retrieval pipeline.

## Step 6 — Evidence Pack

The retrieved results are converted into a structured evidence package containing requirement and supporting evidence metadata.

## Step 7 — Test Case Generation

The Test Case Agent generates structured test cases from the selected requirement and evidence.

## Step 8 — Self-RAG Validation

The Validation Agent checks whether the generated test case is fully supported by authoritative requirement evidence.

## Step 9 — Regeneration

If validation fails, feedback is returned to the Test Case Agent.

The test case can be regenerated up to the configured retry limit.

## Step 10 — Traceability

Validated test cases are mapped back to the source requirement.

If the requirement has become stale, traceability can be marked stale.

## Step 11 — Reporting

The system provides:

- requirement coverage,
- validation status,
- stale requirements,
- traceability matrix,
- and QA coverage information.

---

# Core Components

## 1. Document Gate / CRAG

The Document Gate is an early rejection layer.

Its purpose is to avoid unnecessary:

- database writes,
- requirement extraction,
- retrieval,
- LLM calls,
- and test-case generation.

The gate uses document structure and CT-200-specific keywords such as `CT-200` / `CT200`, with an LLM fallback when needed.

---

## 2. Version Control

Document identity is based on PDF content hashing.

The version-control layer:

- detects duplicate content,
- assigns versions to new content,
- links a new version to its previous version,
- compares requirement content hashes,
- and identifies stale requirements.

Conceptually:

```text
Document v1
    │
    ├── unchanged requirement → active
    │
    └── changed requirement   → stale

Document v2
    │
    └── new requirement content
```

This prevents test cases from being treated as current when their source requirement has changed.

---

## 3. Requirement Agent

The Requirement Agent parses the manual into a hierarchical requirement tree.

It preserves:

```text
Document
  └── Section
       └── Subsection
            └── Requirement
```

It also stores source information such as:

- page,
- source position,
- table content,
- content hash,
- parent node,
- document version.

---

# Retrieval System

Retrieval is a major part of the project because generated test cases must be grounded in the correct evidence.

The retrieval system uses multiple complementary sources:

```text
                    Query
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
        BM25        Qdrant       Graph
          │           │           │
          └───────────┼───────────┘
                      ▼
                Hybrid Retrieval
                      │
                RRF combination
                      │
             Recall Protection
                      │
               CrossEncoder #1
                      │
        Calibration / Cost Filter
                      │
               CrossEncoder #2
                      │
                  Final Top-K
```

## BM25

BM25 provides lexical retrieval.

It is useful for exact technical terminology such as:

- `E3`,
- `E6`,
- `299 mmHg`,
- `40 mmHg`,
- `Bluetooth`,
- `CSV`,
- firmware versions.

## Qdrant semantic retrieval

Qdrant provides vector-based semantic retrieval.

The project uses sentence-transformer embeddings with 384-dimensional vectors.

The indexed text includes:

- title,
- requirement text,
- flattened table content.

Including table content is important because critical specifications can exist only inside tables.

## Knowledge Graph retrieval

The knowledge graph represents relationships between entities such as:

```text
Document
Requirement
Constraint
ErrorCode
Table
TestCase
```

Relationships include:

```text
CONTAINS
HAS_CHILD
HAS_ERROR_CODE
HAS_CONSTRAINT
HAS_TABLE
VERIFIED_BY
VERSION_OF
```

Graph retrieval provides structural/contextual evidence that lexical or semantic retrieval may miss.

---

# Two Hybrid Retrieval Strategies

The project does not treat "hybrid retrieval" as one single implementation. Two different hybrid configurations were implemented and evaluated.

## Hybrid 1 — BM25 + Qdrant + Knowledge Graph

Hybrid 1 combines three evidence sources:

```text
                Query
                  │
        ┌─────────┼─────────┐
        ▼         ▼         ▼
      BM25      Qdrant     Graph
        │         │         │
        └─────────┼─────────┘
                  ▼
              Hybrid 1
```

### BM25

Provides lexical matching for exact technical terms, identifiers, error codes, and numerical specifications.

### Qdrant

Provides semantic similarity using vector embeddings.

### Knowledge Graph

Provides structural relationships between documents, requirements, constraints, error codes, tables, and test cases.

Hybrid 1 is useful as a baseline for determining whether graph-based retrieval improves over a lexical + semantic combination.

## Hybrid 2 — BM25 + Qdrant

Hybrid 2 intentionally removes the graph branch from the active candidate-generation path:

```text
             Query
               │
        ┌──────┴──────┐
        ▼             ▼
      BM25          Qdrant
        │             │
        └──────┬──────┘
               ▼
              RRF
               │
               ▼
       Recall Protection
               │
               ▼
      Frozen Candidate Pool
```

The purpose is to maintain strong lexical/semantic recall while giving the downstream CrossEncoder and filtering stages a controlled candidate pool.

### Why both hybrids are retained

The two hybrids answer different engineering questions:

| Hybrid | Components | Role |
|---|---|---|
| Hybrid 1 | BM25 + Qdrant + Graph | Baseline comparison |
| Hybrid 2 | BM25 + Qdrant + RRF + Recall Protection | Active retrieval pipeline |

The retrieval evaluation therefore compares the contribution of graph retrieval rather than assuming that adding another retrieval source automatically improves the final system.

---

# Retrieval Methodology

The retrieval evaluation compares several retrieval configurations and then evaluates the downstream stages of the active Hybrid-2 pipeline.

## Standalone baselines

### 1. BM25

Pure lexical retrieval.

### 2. Knowledge Graph

Graph-based retrieval.

## Hybrid baselines

### 3. Hybrid 1

```text
BM25 + Qdrant + Graph
```

Hybrid 1 tests the combination of lexical, semantic, and graph evidence.

### 4. Hybrid 2

```text
BM25 + Qdrant
      ↓
RRF
      ↓
Recall Protection
```

Hybrid 2 is the active candidate-generation strategy.

## Downstream Hybrid-2 stages

The frozen Hybrid-2 candidate pool is then processed by:

```text
Hybrid 2
   ↓
CrossEncoder #1
   ↓
Calibration
   ↓
Cost-Sensitive Filter
   ↓
CrossEncoder #2
   ↓
Final Top-K
```

This separation is important because the evaluation distinguishes candidate generation from reranking and evidence filtering.

---

# Active Retrieval Pipeline

The final active retrieval design is:

```text
Hybrid 2
(BM25 + Qdrant)
       │
       ▼
RRF Combination
       │
       ▼
Recall Protection
       │
       ▼
Frozen Candidate Pool
       │
       ▼
CrossEncoder #1
       │
       ▼
Calibration
       │
       ▼
Cost-Sensitive Filter
       │
       ▼
Surviving Evidence
       │
       ▼
CrossEncoder #2
       │
       ▼
Final Top-K
```

The stages have distinct responsibilities:

- **Hybrid 2:** generates the broad candidate pool.
- **RRF:** combines lexical and semantic retrieval rankings.
- **Recall Protection:** prevents relevant candidates from being removed too early.
- **CrossEncoder #1:** performs pairwise relevance scoring/reranking on the frozen candidate pool.
- **Calibration:** converts raw CrossEncoder scores into calibrated relevance estimates.
- **Cost-Sensitive Filter:** removes candidates that do not satisfy the configured relevance/recall trade-off.
- **CrossEncoder #2:** reranks only the surviving evidence.
- **Final Top-K:** becomes the evidence supplied to the Evidence Pack.

A key evaluation design decision is that all downstream stages operate on the same frozen Hybrid-2 candidate pool. CrossEncoder #1 therefore does not change the candidate set; filtering is the stage that removes evidence; CrossEncoder #2 then performs the final ordering of the survivors.


This avoids a candidate-set confound:

- CrossEncoder #1 only reorders candidates.
- Filtering only removes candidates.
- CrossEncoder #2 only reorders surviving candidates.

Therefore improvements/degradations can be attributed to the corresponding downstream stage more reliably.

---

# Retrieval Metrics

Three metrics are used.

## Recall@5

```text
Recall@5 =
relevant documents retrieved in top 5
-------------------------------------
total relevant documents
```

Recall answers:

> Did the system retrieve the evidence we actually needed?

## Precision@5

```text
Precision@5 =
relevant documents in top 5
---------------------------
5
```

Precision answers:

> How much of the returned evidence is actually relevant?

## MRR

Mean Reciprocal Rank measures how early the first relevant result appears.

A higher MRR means the correct evidence is ranked closer to the top.

---

# Retrieval Evaluation

The retrieval evaluation was performed against the Gold Standard dataset.

The Gold Standard defines:

- manually verified relevant evidence node IDs,
- atomic ground-truth facts,
- document versions,
- requirement IDs.

This allows retrieval quality to be evaluated independently from LLM test-case generation.

## Gold Standard version coverage

The evaluation includes:

```text
v1 → ct200_manual.pdf
v2 → ct200_manual_v2.pdf
v3 → ct200_manual_v3.pdf
```

The Gold Standard contains 15 end-to-end cases, including table-specific retrieval cases.

---

# Retrieval Case 1 — V1 Table Evidence

### Case

`GS-V1-T01`

### Requirement

`REQ-2.1`

### Query objective

Retrieve pressure/pulse ranges and accuracy values stored in table content.

### Problem discovered

Initially, Qdrant embeddings were built from title + text while table information was not included.

This caused table-only evidence to be missed.

### Fix

The embedding construction was changed to include flattened table content.

The Qdrant collection was then re-indexed.

### Result

```text
GS-V1-T01
Recall@K = 1.00
```

This demonstrated that table content needed to be treated as first-class retrieval evidence.

---

# Retrieval Case 2 — V2 Table Evidence

### Case

`GS-V2-T01`

### Requirement

`REQ-4.2`

### Query objective

Retrieve the E6 Bluetooth synchronization failure behavior stored in table content.

### Problem

The same class of table-retrieval issue affected evidence that was present primarily in table data.

### Fix

Table content was included in the semantic embedding text and the index was rebuilt.

### Result

```text
GS-V2-T01
Recall@K = 1.00
```

These two cases were important because they demonstrated that semantic indexing of only paragraph text was insufficient for the CT-200 manual.

---

# Overall Retrieval Results

The final retrieval comparison was:

| Pipeline / Method | Recall@5 | Precision@5 | MRR |
|---|---:|---:|---:|
| BM25 | 0.917 | 0.273 | 0.706 |
| Graph | 0.550 | 0.160 | 0.508 |
| Hybrid 1: BM25 + Qdrant + Graph | 0.867 | 0.253 | 0.613 |
| Hybrid 1 + CrossEncoder | 0.867 | 0.253 | 0.883 |
| Hybrid 2: BM25 + Qdrant | **0.967** | 0.280 | 0.647 |
| Hybrid 2 + CrossEncoder #1 | 0.950 | 0.267 | 0.883 |
| Old: H2 + Calibration + Filter + CE #2 | 0.817 | **0.418** | 0.817 |
| Current: H2 + Filter | 0.950 | 0.290 | **0.917** |
| Current: Final Hybrid 2 | 0.950 | 0.290 | **0.917** |

---

# Precision–Recall Trade-off

The retrieval results show why precision cannot be optimized independently from recall in a traceability system.

## Raw Hybrid 2

```text
Recall@5    = 0.967
Precision@5 = 0.280
MRR         = 0.647
```

Hybrid 2 retrieves almost all relevant evidence, but the candidate set contains additional irrelevant evidence.

For downstream QA traceability, this is useful because the evidence can still be filtered and reranked.

## Old aggressive filtering

```text
Recall@5    = 0.817
Precision@5 = 0.418
MRR         = 0.817
```

The old filter increased precision substantially:

```text
0.280 → 0.418
```

but recall fell:

```text
0.967 → 0.817
```

That means more relevant evidence was removed.

For this application, that is undesirable because a missing safety constraint or numeric specification can lead to an incomplete test case.

## Current recall-protected filtering

```text
Recall@5    = 0.950
Precision@5 = 0.290
MRR         = 0.917
```

Compared with the old final pipeline:

```text
Recall:    +0.133
Precision: -0.128
MRR:       +0.100
```

The current system deliberately accepts a small amount of additional evidence in exchange for much stronger recall and ranking quality.

### Design decision

The final retrieval design prioritizes:

```text
Recall protection
        +
Strong ranking
        +
Controlled filtering
```

rather than maximum precision alone.

This is appropriate because the downstream Test Case Agent and Validation Agent can work with a broader evidence set, whereas missing required evidence cannot be recovered after retrieval.

---

# What Happens After CrossEncoder Reranking

CrossEncoder processing is not the end of retrieval.

The final evidence path is:

```text
Hybrid 2 Candidate Pool
        ↓
CrossEncoder #1
        ↓
Raw Relevance Scores
        ↓
Calibration
        ↓
Calibrated Relevance
        ↓
Cost-Sensitive Filter
        ↓
Surviving Evidence
        ↓
CrossEncoder #2
        ↓
Final Ranking
        ↓
Top-K Evidence
        ↓
Evidence Pack
```

This distinction is important:

- CrossEncoder #1 does not directly produce the final evidence.
- Calibration interprets the CrossEncoder score.
- Cost-sensitive filtering decides which candidates survive.
- CrossEncoder #2 performs the final ordering.
- The Evidence Pack consumes the final Top-K evidence.

The retrieval evaluation also uses a frozen candidate pool so that these downstream stages can be compared without repeatedly changing the candidate-generation set.

---

# Evidence Pack

The Evidence Pack converts retrieval results into structured evidence for downstream agents.

It contains information such as:

- node ID,
- title,
- requirement text,
- table content,
- page,
- content hash,
- retrieval score,
- reranker score,
- related evidence.

MongoDB acts as the source of truth while retrieval indexes provide derived search representations.

---

# Cache and Human-in-the-Loop (HITL)

Cache and human approval are important parts of the operational test-case workflow.

## Cache-first behavior

Before generating new test cases, the Test Case Agent checks whether valid approved cached test cases already exist.

Conceptually:

```text
Selected Requirement
       ↓
Cache Lookup
       │
 ┌─────┴─────┐
 │           │
HIT         MISS
 │           │
 ▼           ▼
Reuse      Generate
Cases      New Cases
```

A cached test case is not blindly reused. The cache logic checks the relevant document/version/requirement state and validation/approval conditions.

The observed evaluation logs show both paths:

```text
[CACHE] No valid cache found. Generating new test cases.
```

and, after approval:

```text
[CACHE] Valid approved test cases found.
[CACHE] Returning ... cached test case(s).
```

This reduces unnecessary LLM calls and provides deterministic reuse for already approved, still-valid test cases.

## Human-in-the-Loop approval

Newly generated and successfully validated test cases go through a human approval stage.

```text
Generated Test Cases
        ↓
Self-RAG Validation
        ↓
PASS
        ↓
HITL Approval
        │
   ┌────┴────┐
   │         │
APPROVE    REJECT
   │         │
   ▼         ▼
Cache      Review /
Update     Regenerate
```

The HITL stage provides a human QA checkpoint before generated test cases become approved reusable artifacts.

During automated Gold Standard evaluation, human approval was programmatically set to `YES` so that the evaluation could run without interactive input. This does not remove the HITL concept from the operational pipeline.

## Why HITL and cache are connected

The approval state is part of cache validity.

The intended sequence is:

```text
Generate
   ↓
Validate
   ↓
Human Approve
   ↓
Cache Approved Test Case
   ↓
Future Run
   ↓
Cache Validation
   ↓
Reuse if still valid
```

This prevents an unreviewed or stale generated test case from automatically becoming the reusable source for future executions.

---

# Test Case Generation

The Test Case Agent generates structured test cases from the selected requirement and the final Evidence Pack.

The operational flow is:

```text
Selected Requirement
       +
Final Evidence Pack
       ↓
Cache Validation
       │
   ┌───┴────┐
   │        │
Valid     Invalid/
Cache     Missing
   │        │
   ▼        ▼
Reuse     Generate
           │
           ▼
        Validate
           │
           ▼
        HITL Approval
```

The generation policy emphasizes:

- requirement-grounded test cases,
- ordered steps,
- observable expected results,
- explicit numeric thresholds when present,
- no unsupported assumptions,
- no invented UI/API/hardware behavior,
- no invented limits or error codes,
- no duplicate test cases.

The selected requirement remains the authoritative source.

Retrieved evidence is used to support or clarify the requirement.

---

# Self-RAG Validation

The Validation Agent acts as a grounding layer between generation and traceability.

The validator checks whether the generated test case is supported by the source requirement and evidence.

Conceptually:

```text
Requirement
    +
Evidence
    +
Generated Test Case
        │
        ▼
Self-RAG Validator
        │
        ├── PASS
        │
        └── FAIL + feedback
```

The validation policy is strict:

> Every claim in the test case's preconditions, steps, and expected results must be supported by authoritative evidence.

The validator also uses related evidence where appropriate, while treating the primary requirement as authoritative.

---

# Bounded Regeneration and HITL

A failed validation result can trigger regeneration.

```text
Generate
   ↓
Validate
   ↓
FAIL
   ↓
Feedback
   ↓
Regenerate
   ↓
Validate
```

The loop is bounded by `max_retries`.

This provides two protections:

1. A bad generation can be corrected.
2. A persistent failure cannot create an infinite LLM loop.

For automated evaluation, human input can be bypassed so that the evaluation measures the pipeline consistently.

In normal operation, human approval can remain part of the workflow for cached test cases and QA review.

---

# Traceability

The Traceability Agent maps validated test cases back to their source requirement.

The resulting relationship is conceptually:

```text
Document Version
      │
      ▼
Requirement
      │
      ▼
Test Case
      │
      ▼
Validation Result
      │
      ▼
Traceability Status
```

If the requirement's content hash changes in a newer manual version, the traceability relationship can be identified as stale.

This allows QA teams to answer:

- Which requirement does this test case cover?
- Which manual version was the source?
- Has that requirement changed?
- Was the test case validated?
- Does the test case need regeneration?

---

# End-to-End Evaluation

Retrieval quality and end-to-end quality are evaluated separately.

This distinction is important.

A retrieval system can retrieve relevant evidence successfully while the final LLM-generated test case still fails to preserve all required facts.

Therefore:

```text
Retrieval Evaluation
        ≠
End-to-End Evaluation
```

---

# End-to-End Gold Standard

The Gold Standard contains 15 cases.

Each case defines:

- expected document version,
- expected requirement node IDs,
- relevant evidence nodes,
- ground-truth facts,
- critical facts.

The end-to-end evaluator performs:

```text
Gold Standard Case
      ↓
Find correct document version
      ↓
Select expected requirement
      ↓
Run requirement pipeline
      ↓
Generate test cases
      ↓
Validate
      ↓
Measure requirement coverage
      ↓
Measure fact coverage
      ↓
Check version correctness
      ↓
Assign PASS / PARTIAL / FAIL
```

---

# End-to-End Metrics

The evaluator calculates:

## Requirement Coverage

Measures whether the correct expected requirement was selected.

## Fact Coverage

Measures how many Gold Standard facts are preserved in the generated test-case output.

## Version Correctness

Checks whether generated output preserves facts associated with the expected document version.

## Validation Pass Rate

Measures whether generated test cases passed Self-RAG validation.

## Overall Status

A case is:

### PASS

When:

```text
Requirement Coverage = 100%
Fact Coverage = 100%
Version Correctness = true
Validation = PASS
```

### PARTIAL

When requirement coverage is complete and some ground-truth facts are preserved.

### FAIL

When requirement coverage is incomplete or fact coverage is zero/insufficient according to the evaluation logic.

---

# End-to-End Results

The final evaluation produced:

```text
FINAL EVALUATION SUMMARY

Total Cases              : 15
Passed                   : 1
Partial                  : 10
Failed                   : 4
Pass Rate                : 6.67%

Requirement Coverage     : 100.00%
Fact Coverage            : 37.50%
Version Correctness      : 73.33%
Validation Pass Rate     : 100.00%
```

## Case-wise results

| Case | Requirement Coverage | Fact Coverage | Validation | Status |
|---|---:|---:|---|---|
| GS-V1-01 | 100% | 40% | PASS | PARTIAL |
| GS-V1-02 | 100% | 62% | PASS | PARTIAL |
| GS-V1-03 | 100% | 50% | PASS | PARTIAL |
| GS-V1-04 | 100% | 50% | PASS | PARTIAL |
| GS-V1-05 | 100% | 20% | PASS | PARTIAL |
| GS-V1-06 | 100% | 40% | PASS | PARTIAL |
| GS-V2-01 | 100% | 50% | PASS | PARTIAL |
| GS-V2-02 | 100% | 50% | PASS | PARTIAL |
| GS-V2-03 | 100% | 0% | PASS | FAIL |
| GS-V2-04 | 100% | 0% | PASS | FAIL |
| GS-V3-01 | 100% | 0% | PASS | FAIL |
| GS-V3-02 | 100% | 33% | PASS | PARTIAL |
| GS-V3-03 | 100% | 67% | PASS | PARTIAL |
| GS-V1-T01 | 100% | 100% | PASS | PASS |
| GS-V2-T01 | 100% | 0% | PASS | FAIL |

---

# Interpretation of End-to-End Results

The results reveal a clear separation between retrieval/requirement selection and generation-level fact preservation.

## Requirement selection is strong

```text
Requirement Coverage = 100%
```

All 15 cases selected the expected requirement coverage according to the evaluation.

This indicates that the requirement identification/selection portion of the pipeline performed strongly on the Gold Standard.

## Validation is strong

```text
Validation Pass Rate = 100%
```

All evaluated generated outputs passed the validation stage under the current evaluation run.

## Fact preservation is the main weakness

```text
Fact Coverage = 37.50%
```

The generated outputs did not preserve all Gold Standard facts even though the expected requirements were selected and validation passed.

This is the most important end-to-end limitation identified by the current evaluation.

## Version correctness is incomplete

```text
Version Correctness = 73.33%
```

The system correctly preserved version-specific facts in most, but not all, cases.

This is particularly important for changed numerical specifications and error-code behavior across versions.

## Overall pass rate

```text
1 / 15 = 6.67%
```

The low overall pass rate is therefore not evidence that the entire retrieval system has 6.67% accuracy.

It reflects the stricter end-to-end requirement that all required facts, version information, and validation conditions must be satisfied simultaneously.

---

# Challenges Faced and Solutions

## Challenge 1 — Requirements were stored in PDFs

### Problem

PDF manuals do not directly provide a database-ready requirement structure.

### Solution

A Requirement Agent was created to extract hierarchical sections and preserve:

- node IDs,
- parent-child relationships,
- tables,
- pages,
- source positions,
- hashes.

---

## Challenge 2 — Document version changes

### Problem

A changed manual can make previously generated test cases stale.

### Solution

Document content hashing and requirement-level content hashes were introduced.

Changed requirement hashes are used to identify stale requirements.

---

## Challenge 3 — Cross-version contamination

### Problem

Retrieval could accidentally use evidence from another manual version.

### Solution

Retrieval and storage operations are version-aware and use document/version filtering.

This keeps evidence associated with the correct manual version.

---

## Challenge 4 — Important information existed inside tables

### Problem

Initial semantic embeddings focused on title and text and missed table-only specifications.

Examples included:

- pressure ranges,
- accuracy values,
- E6 behavior.

### Solution

Table content was added to the embedding text and the Qdrant collection was re-indexed.

This led to:

```text
GS-V1-T01 Recall@K = 1.00
GS-V2-T01 Recall@K = 1.00
```

---

## Challenge 5 — BM25 and candidate protection

### Problem

A high-precision filter can remove relevant evidence before generation.

An earlier retrieval configuration produced:

```text
Recall@5 = 0.817
Precision@5 = 0.418
```

### Solution

A recall-protected filtering strategy was introduced.

The current final pipeline achieved:

```text
Recall@5    = 0.950
Precision@5 = 0.290
MRR         = 0.917
```

This prioritizes retaining required evidence while still improving ranking.

---

## Challenge 6 — CrossEncoder ranking

### Problem

Retrieval relevance and ranking quality are different objectives.

### Solution

A CrossEncoder was introduced after the hybrid candidate pool.

The first CrossEncoder improved ranking substantially:

```text
MRR:
Hybrid 2             = 0.647
Hybrid 2 + CE #1      = 0.883
```

The system then added filtering and a second reranking stage.

---

## Challenge 7 — LLM hallucination risk

### Problem

A generated test case can look correct while containing unsupported details.

### Solution

The Test Case Agent uses explicit grounding rules and the Validation Agent performs Self-RAG validation.

Unsupported assumptions cause validation failure rather than being silently accepted.

---

## Challenge 8 — Infinite regeneration

### Problem

A validation failure could theoretically cause an endless generate → validate cycle.

### Solution

The LangGraph regeneration loop is bounded by `max_retries`.

---

## Challenge 9 — Fact evaluation was too strict during earlier debugging

### Problem

Literal matching can mark semantically equivalent statements as different.

For example, differences such as:

```text
2 seconds
2 sec
automatically deflates
auto-deflates
```

can represent the same fact.

### Solution

The current evaluation logic performs normalization, common wording normalization, time-based matching, and token-based fallback rather than relying only on exact literal substring matching.

---

## Challenge 10 — API/LLM reliability

### Problem

LLM-backed pipelines can be affected by provider quota/rate limits and structured-output failures.

### Solution

The project separates LLM calls through a shared client/helper and uses structured Pydantic output models and bounded retries where applicable.

Provider quota issues were encountered during earlier end-to-end runs, including a Gemini 429 limitation.

---

# Retrieval vs End-to-End Performance

The project deliberately reports retrieval and end-to-end metrics separately.

| Evaluation Layer | Main Question | Result |
|---|---|---:|
| Retrieval | Did we retrieve the required evidence? | Current Recall@5 = 0.950 |
| Requirement Selection | Did we identify the expected requirement? | 100% coverage |
| Validation | Is the generated output grounded? | 100% validation pass |
| Fact Preservation | Did generation preserve all required facts? | 37.50% |
| Version Correctness | Did output preserve correct version-specific facts? | 73.33% |
| Overall E2E | Did the complete case satisfy all conditions? | 6.67% pass |

This separation prevents misleading claims such as calling the system a 95% end-to-end system merely because retrieval Recall@5 is 95%.

---

# Data and Storage Architecture

The system uses persistent storage for source documents, requirements, test cases, validation results, and traceability relationships.

The project contains database models for concepts including:

```text
Document
RequirementNode
TestCase
ValidationResult
Traceability
```

The retrieval layer also maintains derived search indexes.

The architecture follows:

```text
                 MongoDB
              Source of Truth
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
        BM25      Qdrant    Graph
       Index      Vectors   Structure
```

Derived retrieval indexes can be rebuilt from the source data.

This reduces the risk of treating a stale vector/index representation as the authoritative document state.

---

# Knowledge Graph

The knowledge graph is version-aware and represents the structure of the CT-200 domain.

## Node types

```text
Document
Requirement
ErrorCode
Constraint
Table
TestCase
```

## Relationships

```text
Document
  └── CONTAINS → Requirement

Requirement
  ├── HAS_CHILD → Requirement
  ├── HAS_ERROR_CODE → ErrorCode
  ├── HAS_CONSTRAINT → Constraint
  ├── HAS_TABLE → Table
  └── VERIFIED_BY → TestCase

Document
  └── VERSION_OF → Document
```

The graph provides structural context for retrieval and traceability.

---

# API Layer

FastAPI exposes the backend functionality.

Representative endpoints include:

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/analyze` | Upload and process a manual |
| GET | `/traceability-matrix/{document_id}` | Retrieve traceability matrix |
| GET | `/coverage/{document_id}` | Retrieve coverage information |
| GET | `/validation/{document_id}` | Retrieve validation results |
| GET | `/reports/{document_id}` | Retrieve combined QA report |
| GET | `/export/{document_id}` | Export traceability matrix |
| GET | `/requirements/{node_id}` | Retrieve a requirement |
| GET | `/requirements/search/{keyword}` | Search requirement text |

Interactive API documentation is available through FastAPI/Swagger.

---

# Project Structure

```text
ct200-traceability-system/
│
├── app/
│   ├── main.py
│   ├── graph.py
│   ├── graph_state.py
│   ├── parser.py
│   ├── llm_client.py
│   ├── database.py
│   ├── models.py
│   ├── schemas.py
│   │
│   ├── agents/
│   │   ├── document_gate.py
│   │   ├── version_control.py
│   │   ├── requirement_agent.py
│   │   ├── test_case_agent.py
│   │   ├── validation_agent.py
│   │   └── traceability_agent.py
│   │
│   ├── retrieval.py
│   ├── vector_store.py
│   ├── knowledge_graph.py
│   ├── evidence_pack.py
│   └── graph_visualizer.py
│
├── data/
│   └── CT-200 manuals / generated data
│
├── notebooks/
│
├── tests/
│
├── gold_standard.py
├── retrieval_evaluation.py
├── final_evaluation.py
├── requirements.txt
├── README.md
└── .env
```

---

# Technology Stack

| Layer | Technology |
|---|---|
| Language | Python |
| API | FastAPI |
| Orchestration | LangGraph |
| PDF Processing | PyMuPDF / pdfplumber |
| Database | MongoDB |
| ORM / Persistence | SQLAlchemy where applicable |
| Semantic Search | Qdrant |
| Embeddings | Sentence Transformers |
| Lexical Retrieval | BM25 |
| Reranking | CrossEncoder #1 + CrossEncoder #2 |
| Score Calibration | Calibrated CrossEncoder relevance mapping |
| Filtering | Cost-Sensitive / Recall-Protected Filtering |
| Graph Retrieval | NetworkX / graph representation |
| Validation | Pydantic + Self-RAG |
| Human Review | Human-in-the-Loop (HITL) |
| Test Case Reuse | Version-aware validation cache |
| LLM | Configurable LLM provider |
| Server | Uvicorn |
| Visualization | PyVis |

The active project environment used Python 3.11.7 for the CT-200 system.

---

# Installation and Running

## 1. Clone the repository

```bash
git clone https://github.com/mdrajaistekhar786/ct200-traceability-system.git
cd ct200-traceability-system
```

## 2. Create the virtual environment

```bash
python -m venv .venv
```

## 3. Activate on Windows

```bash
.venv\Scripts\activate
```

## 4. Install dependencies

```bash
pip install -r requirements.txt
```

## 5. Configure environment variables

Create a `.env` file with the credentials/configuration required by the selected LLM and storage services.

Do not commit API keys to Git.

## 6. Run FastAPI

```bash
uvicorn app.main:app --reload
```

## 7. Open Swagger

```text
http://127.0.0.1:8000/docs
```

---

# Testing

The project includes automated tests for core application behavior.

The CT-200 system test suite was previously executed successfully with:

```text
3 tests passed
```

Evaluation scripts are separate from unit/integration tests because retrieval and end-to-end evaluation require the Gold Standard and retrieval/LLM pipeline.

---

# Evaluation Design

The project uses two complementary evaluation layers.

## Layer 1 — Retrieval Evaluation

Purpose:

> Determine whether the retrieval system finds the evidence required to answer a query.

Metrics:

- Recall@5
- Precision@5
- MRR

This layer compares BM25, Graph, Hybrid 1, Hybrid 2, CrossEncoder reranking, filtering, and the final pipeline.

## Layer 2 — End-to-End Evaluation

Purpose:

> Determine whether the complete system converts a Gold Standard requirement into a validated, version-correct, fact-complete test case.

Metrics:

- Requirement Coverage
- Fact Coverage
- Version Correctness
- Validation Pass Rate
- Overall PASS/PARTIAL/FAIL

The two layers are intentionally separated.

---

# Important Evaluation Finding

The most important conclusion from the current experiments is:

> **Strong retrieval does not automatically guarantee strong end-to-end fact preservation.**

The retrieval pipeline achieved:

```text
Current Final Recall@5 = 0.950
Current Final MRR      = 0.917
```

while the end-to-end evaluation achieved:

```text
Requirement Coverage = 100.00%
Fact Coverage        = 37.50%
Version Correctness  = 73.33%
Validation Pass Rate = 100.00%
Overall Pass Rate    = 6.67%
```

Therefore the next optimization target is not simply "retrieve more documents."

The main remaining problem is ensuring that the generated test case preserves all required facts from the selected requirement and evidence while remaining valid under version-specific constraints.

---

# Limitations

The current evaluation indicates several limitations:

1. Fact preservation is substantially weaker than requirement selection.
2. Version correctness is not yet perfect.
3. Retrieval precision is relatively low when recall is protected.
4. The system depends on LLM availability and provider quotas.
5. The current Gold Standard is manually curated and limited to the defined CT-200 cases.
6. OCR is not currently implemented for scanned PDFs.
7. Cross-document retrieval for Self-RAG is a future improvement.
8. Long-running graph execution does not yet use persistent LangGraph checkpointing.

---

# Future Improvements

## Retrieval

- Improve calibrated filtering without sacrificing recall.
- Expand Gold Standard coverage.
- Evaluate additional embedding models.
- Tune CrossEncoder thresholds using held-out data.
- Improve table-aware retrieval.

## Generation

- Improve fact completeness.
- Introduce explicit fact checklists before generation.
- Add structured fact-level validation.
- Reduce unnecessary regeneration.

## Versioning

- Improve automatic version-diff explanations.
- Produce requirement-level change summaries.
- Automatically identify test cases impacted by changed requirements.

## Infrastructure

- PostgreSQL deployment option.
- Docker deployment.
- Authentication and authorization.
- LangGraph checkpointing.
- Better observability and execution tracing.

## Document Processing

- OCR support.
- Better extraction for complex tables.
- More robust parsing of unusual PDF layouts.

---

# Conclusion

The CT-200 Traceability System combines document intelligence, version control, hybrid retrieval, LLM-based test-case generation, Self-RAG validation, and traceability into one QA-oriented workflow.

The architecture is intentionally layered:

```text
Document Validation
        ↓
Version Control
        ↓
Requirement Extraction
        ↓
Hybrid 1 / Hybrid 2 Retrieval Evaluation
        ↓
Active Hybrid 2
        ↓
RRF + Recall Protection
        ↓
CrossEncoder #1
        ↓
Calibration + Cost-Sensitive Filtering
        ↓
CrossEncoder #2
        ↓
Final Evidence Pack
        ↓
Test Case Cache Check
        ↓
Test Case Generation
        ↓
Self-RAG Validation
        ↓
Bounded Regeneration
        ↓
Human-in-the-Loop Approval
        ↓
Approved Cache
        ↓
Traceability
        ↓
Evaluation / Reporting
```

The retrieval experiments show that the final recall-protected Hybrid 2 pipeline provides a strong retrieval/ranking trade-off:

```text
Recall@5    = 0.950
Precision@5 = 0.290
MRR         = 0.917
```

The end-to-end experiments show that the primary remaining challenge is not requirement selection but complete fact preservation across generation and version-sensitive behavior:

```text
Requirement Coverage = 100.00%
Fact Coverage        = 37.50%
Version Correctness  = 73.33%
Validation Pass Rate = 100.00%
Overall Pass Rate    = 6.67%
```

These results establish a clear engineering direction: preserve the high-recall retrieval behavior while improving fact-complete, version-correct test-case generation and validation.

---

# Author

**Md Raja Istekhar**

GitHub: `https://github.com/mdrajaistekhar786/ct200-traceability-system`

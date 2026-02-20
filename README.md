# 📊 E-Com AI Analyst

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://e-commerce-ai-analyst.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A self-learning Text-to-SQL agent that turns plain English into accurate business insights — and gets smarter with every query you validate.**

Upload any CSV or Excel file, ask questions in natural language, and get instant answers with SQL transparency, confidence scoring, and visual charts. No SQL knowledge required.

---

## 🎬 Demo

<!-- Replace the placeholder below with your 15-second screen recording GIF -->
<!-- Recommended: use https://gifcap.dev or ScreenToGif to record, keep under 5MB -->

![Demo GIF](assets/demo.gif)

*Upload CSV → Ask a question → Get an answer with explanation → Validate with 👍 → AI learns for next time*

---

## ✨ Key Features

- **Text-to-SQL Generation** — Converts natural language questions into precise SQLite queries using Google Gemini
- **Self-Healing SQL** — Failed queries are automatically fed back to the LLM with error context for self-correction (up to 3 retries)
- **Dynamic Few-Shot Learning** — Validated Q&A pairs are stored and retrieved via BM25 ranking to improve future query accuracy
- **User Validation Gate** — 👍/👎 buttons let you control what the AI learns — only human-approved answers enter the knowledge base
- **Zero-Hallucination Pipeline** — SQL generation and explanation are separate LLM calls, ensuring explanations are always grounded in actual data
- **Intent Guardrail** — Off-topic questions (jokes, poems, prompt injection) are blocked before SQL generation
- **Schema-Agnostic** — Works with any CSV/Excel file — the system auto-detects schema, generates business context, and adapts instantly
- **Explainability Layer** — Every answer includes a plain-English explanation starting with "I calculated..." plus a confidence level (High/Medium/Low)
- **Smart Visualizations** — Auto-renders scalars as metrics, 2-column results as bar/line charts, and everything else as interactive tables
- **Multi-Sheet Excel Support** — Upload `.xlsx` files with multiple sheets — each sheet becomes a queryable table

---

## 🏗️ How It Works

```
┌─────────────┐
│  User Query  │
└──────┬──────┘
       │
       ▼
┌──────────────┐    blocked     ┌──────────────┐
│   Guardrail   │──────────────▶│  Polite       │
│  (LLM Check)  │               │  Decline      │
└──────┬──────┘               └──────────────┘
       │ allowed
       ▼
┌──────────────────────────────────────────────┐
│           SQL Generation (Call 1)             │
│                                              │
│  Schema RAG ─── Dynamic schema injection     │
│  Few-Shot RAG ── BM25 retrieval of best      │
│                  matching validated examples  │
│  Business Context ── Auto-profiled dataset    │
│                      summary                 │
└──────┬──────┘
       │
       ▼
┌──────────────┐    error    ┌──────────────┐
│  Execute SQL  │───────────▶│  Self-Heal    │
│  (Read-Only)  │◀───────────│  Loop (×3)    │
└──────┬──────┘             └──────────────┘
       │ success
       ▼
┌──────────────────────────────────────────────┐
│         Explanation (Call 2)                  │
│  Grounded in actual query results            │
│  "I calculated..." + Confidence: High/Med/Low│
└──────┬──────┘
       │
       ▼
┌──────────────┐    👍     ┌──────────────────┐
│  Display to   │─────────▶│  Few-Shot Store   │
│  User + 👍👎  │          │  (SQLite + FTS5)  │
└──────────────┘   👎      └──────────────────┘
                  ──▶ discard
```

### The Learning Loop

1. User asks a question → LLM generates SQL → query executes → results displayed
2. User clicks **👍** if the answer is correct → Q&A pair is saved to the few-shot store
3. Next similar question → BM25 retrieves the most relevant validated examples → injected into the LLM prompt
4. The system gets progressively more accurate for each dataset over time

### Why Not Vector Embeddings?

The few-shot store uses **SQLite FTS5 with BM25 ranking** instead of vector embeddings. This is intentional:

- **Zero additional dependencies** — FTS5 is built into SQLite
- **No embedding API costs** — no per-query embedding calls
- **Sufficient recall** — business users use consistent domain vocabulary ("revenue", "customers", "orders"), making lexical matching effective
- **Swappable architecture** — the `FewShotStore.retrieve()` interface is designed to be replaced with vector search (ChromaDB, SQLite-VSS) in V2

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A [Google Gemini API key](https://aistudio.google.com/apikey) (free tier works)

### Installation

```bash
# Clone the repository
git clone https://github.com/dziubeep/E-commerce-AI-analyst.git
cd E-commerce-AI-analyst

# Install dependencies
pip install -r requirements.txt

# Set your API key
echo "GEMINI_API_KEY=your_key_here" > .env

# Run the app
streamlit run app.py
```

### Usage

1. Open the app in your browser (default: `http://localhost:8501`)
2. Upload a CSV or Excel file via the sidebar — or use the built-in demo dataset
3. Ask any business question: *"What is the total revenue?"*, *"Top 10 products by sales"*, *"Monthly trend"*
4. Review the answer, explanation, and SQL
5. Click 👍 to save good answers to the AI's memory, or 👎 to discard

---

## 📁 Project Structure

```
├── app.py              # Streamlit Chat UI (Phase 4)
├── agent.py            # Core Text-to-SQL engine — zero-hallucination pipeline
├── few_shot_store.py   # Dynamic learning system — SQLite FTS5 + BM25
├── etl.py              # ETL pipeline — cleans raw CSV into SQLite
├── business_logic.txt  # Legacy gold-standard examples (auto-migrated to store)
├── requirements.txt    # Python dependencies
└── .streamlit/
    └── config.toml     # Streamlit configuration
```

---

## 🛡️ Defensive Layers

| Layer | Purpose |
|-------|---------|
| **Input Validation** | Trims, length-checks, rejects empty input |
| **Intent Guardrail** | LLM classifier blocks off-topic/malicious questions |
| **Schema Lockdown** | LLM must use exact column names or output `UNABLE_TO_ANSWER` |
| **Self-Healing Loop** | SQL errors fed back to LLM for auto-correction (max 3 retries) |
| **Empty Result Handler** | Clear message instead of fabricated numbers |
| **Separated LLM Calls** | SQL generation ≠ Explanation — prevents hallucinated explanations |
| **User Validation Gate** | 👍/👎 ensures only correct answers enter the knowledge base |
| **Read-Only Execution** | `PRAGMA query_only=ON` + regex blocks all write operations |
| **SQL Timeout** | Thread-based 30s timeout kills runaway queries |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Interface | Streamlit |
| LLM | Google Gemini (`gemini-2.0-flash`) |
| Database | SQLite (dynamic, created from uploaded files) |
| Few-Shot Retrieval | SQLite FTS5 + BM25 |
| Data Processing | Pandas |
| Error Resilience | Tenacity (exponential backoff) |

---

## 📌 Roadmap

- [x] ETL pipeline with data cleaning
- [x] Core Text-to-SQL engine with few-shot prompting
- [x] Self-healing SQL loop (AWS-inspired)
- [x] Streamlit Chat UI with explainability
- [x] Dynamic schema support (upload any CSV/Excel)
- [x] Dynamic few-shot store with BM25 retrieval
- [x] User validation gate (👍/👎)
- [ ] Vector-based semantic retrieval (ChromaDB / SQLite-VSS)
- [ ] Cross-domain knowledge transfer
- [ ] Multi-user accounts with persistent learning
- [ ] FastAPI + Next.js/React frontend (V2)

---

## 📄 License

This project is part of a university coursework at Vilnius Tech (VILNIUS GEDIMINAS TECHNICAL UNIVERSITY).

---

<p align="center">
  Built with ❤️ using Streamlit and Google Gemini
</p>

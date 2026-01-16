# Natural Language Analytics Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![DuckDB](https://img.shields.io/badge/DuckDB-FFF000?style=flat&logo=duckdb&logoColor=black)](https://duckdb.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-orange)](https://langchain-ai.github.io/langgraph/)

A production-grade, LangGraph-orchestrated social media analytics system designed for deterministic accuracy, semantic search, and hybrid reasoning.

## 🚀 Features

*   **Natural Language Querying**: Ask questions like "How did sentiment change last week?", "Show me examples of bad service", or "Why is sentiment low?".
*   **Deterministic Analytics**: All numbers are computed via **SQL** on a **DuckDB** database. No LLM hallucinations on metrics.
*   **Semantic Search (New)**: Vector-based retrieval to find qualitative evidence (posts/tweets) matching a user's intent, even without keyword matches.
*   **Hybrid Reasoning (New)**: Automates Root Cause Analysis by combining quantitative anomaly detection with qualitative evidence retrieval.
*   **Time-Aware Resolution**: Relative time terms ("last week", "this month") are auto-resolved relative to the **dataset's actual timeframe** (Oct 2025).

## 🛠️ Architecture

1.  **Frontend**: Streamlit (Visualization & Chat)
2.  **Orchestrator**: LangGraph (State Machine: `Classify` -> `Resolve Time` -> `Route` -> [`Analytics` | `Retrieval` | `Hybrid`] -> `Summarize`)
3.  **Data Layer**: DuckDB (Unified Hybrid Database) - Stores both structured analytics data and 384-dimensional vector embeddings.
4.  **Intelligence**:
    *   **LLM**: GPT-4o / Llama 3.3 (via Groq/OpenRouter) for intent understanding and synthesis.
    *   **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (Local or via HuggingFace Inference API).

## 📦 Setup & Installation

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure Environment**
    Create a `.env` file in the root directory:
    ```env
    # Recommended: Groq (Metric Speed)
    OPENAI_API_KEY=gsk_your_groq_key_here
    OPENAI_API_BASE=https://api.groq.com/openai/v1
    LLM_MODEL=llama-3.3-70b-versatile
    
    # Required for Robust Cloud Embeddings (Prevents OOM)
    HUGGINGFACEHUB_API_TOKEN=hf_...
    
    # Database URL (Unified DB)
    DUCKDB_URL=https://huggingface.co/datasets/Bhavin1905/Social-Media-Posts-Dataset-Embeddings-Included-DUCKDB/resolve/main/analytics.duckdb?download=true
    ```

3.  **Run Application**
    ```bash
    $env:PYTHONPATH = "."; streamlit run src/app.py
    ```

## 🔍 Supported Queries

### 1. Analytics (Quantitative)
*   **Trends**: "Sentiment trend last week", "Volume over time".
*   **Topics**: "What are the top topics?", "Key themes this month".
*   **Flexible Aggregations**: "Average sentiment by topic", "Count of posts by week".

### 2. Semantic Search (Qualitative)
*   **Examples**: "Show me examples of bad customer service."
*   **Discovery**: "What are people saying about the new app update?"

### 3. Hybrid Reasoning (Root Cause)
*   **Diagnostics**: "Why is sentiment low?"
*   **Explanation**: "Explain the drop in volume on Tuesday."
    *   *System Flow*: Findings Anomaly -> Drills down Date -> Retrieves Evidence -> Synthesizes Answer.

## 💾 Database Strategy (Unified)

The system now relies on a Single Source of Truth for maximum consistency:

*   **File**: `data/analytics.duckdb` (Downloaded automatically from HuggingFace on startup).
*   **Content**:
    *   **Metadata**: `createddate`, `sentiment`, `topic_id`.
    *   **Content**: `postcontent`.
    *   **Vectors**: `embedding` (384-dimensional, `all-MiniLM-L6-v2`).
*   **Size**: ~460MB.
*   **Alignment**: The entire database was re-indexed locally to ensure vector dimensions match the lightweight reasoning model (384-dim), preventing crashes common with larger models.

## ⚠️ Important Notes

*   **Dataset Timeframe**: The included dataset covers **October 2025**. Queries for "today" (Real-time) will return no data. Use relative terms ("Last week") which map to the *dataset's* last week (Oct 24-30).
*   **Cloud Memory**: The system uses a "Cloud-First" strategy for embeddings. It attempts to use the **HuggingFace Inference API** for vectors to avoid loading the Transformer model into RAM, ensuring stability on free-tier cloud instances.

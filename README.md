# Natural Language Analytics Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![DuckDB](https://img.shields.io/badge/DuckDB-FFF000?style=flat&logo=duckdb&logoColor=black)](https://duckdb.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-orange)](https://langchain-ai.github.io/langgraph/)

A production-grade, LangGraph-orchestrated social media analytics system designed for deterministic accuracy and natural language interaction.

## 🚀 Features

*   **Natural Language Querying**: Ask questions like "How did sentiment change last week?" or "Who are the top influencers?".
*   **Deterministic Analytics**: All numbers are computed via **SQL** on a **DuckDB** database. No LLM hallucinations on metrics.
*   **Time-Aware Resolution**: Relative time terms ("last week", "this month") are auto-resolved relative to the **dataset's actual timeframe** (Oct 2025), not the system clock.
*   **Production Logic**:
    *   **Strict Disclosures**: All answers cite the exact date range used.
    *   **Robust Ops**: Auto-retries on API rate limits, gracefull fallbacks.
    *   **Hybrid AI**: Supports Cloud (OpenRouter/OpenAI) and Local (Ollama) models.

## 🛠️ Architecture

1.  **Frontend**: Streamlit (Visualization & Chat)
2.  **Orchestrator**: LangGraph (State Machine: `Classify` -> `Resolve Time` -> `Execute` -> `Summarize`)
3.  **Data Layer**: DuckDB (Embedded SQL, High Performance)
4.  **Intelligence**:
    *   **Cloud (Default)**: Google Gemini 2.0 Flash (via OpenRouter) or OpenAI GPT-4o.
    *   **Local (Fallback)**: Qwen 2.5 7B via Ollama.

## 📦 Setup & Installation

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure Environment**
    Create a `.env` file in the root directory:
    ```env
    OPENAI_API_KEY=sk-or-your-key-here
    OPENAI_API_BASE=https://openrouter.ai/api/v1
    LLM_MODEL=nvidia/nemotron-3-nano-30b-a3b:free
    # Other options: google/gemini-2.0-flash-exp:free, meta-llama/llama-3.2-3b-instruct:free
    ```

3.  **Data Ingestion** (If not already done)
    Processes CSV, computes embeddings/sentiment, and loads DuckDB.
    ```bash
    $env:PYTHONPATH = "."; python src/ingest.py
    ```

4.  **Run Application**
    ```bash
    $env:PYTHONPATH = "."; streamlit run src/app.py
    ```

## 🔍 Supported Queries

*   **Trends**: "Sentiment trend last week", "Volume over time".
*   **Topics**: "What are the top topics?", "Key themes this month".
*   **Influencers**: "Who are the top influencers?", "Who drove engagement during the campaign?"
*   **Complaints**: "What are users complaining about?"

## 📂 Dataset & Schema

The system processes a historical social media dataset (October 2025). The following fields are critical for analytics:

### **Source Fields**
*   **`postcontent`**: The raw text content of the social media post. Used for:
    *   Sentiment Analysis (TextBlob)
    *   Topic Modeling (Clustering)
    *   Influencer Extraction (Regex for `@mentions`)
*   **`createddate`**: The timestamp of the post.
    *   *Transformation*: Converted to `date` (YYYY-MM-DD) for daily/weekly aggregation.
*   **`embedding`**: Pre-computed vector embeddings (Array).
    *   Used for: K-Means Clustering to group similar posts into Topics.
*   **`id`** / **`uniqueid`**: Unique record identifiers.

### **Computed Fields (DuckDB)**
*   **`sentiment`**: Polarity score (-1.0 to +1.0) derived from `postcontent`.
*   **`topic_id`**: Cluster ID (0-19) assigned via K-Means on `embedding`.
*   **`week`** / **`month`**: Derived time units for aggregation.

## ⚠️ Important Notes

*   **Dataset Timeframe**: The included dataset covers **October 2025**. Queries for "today" (Real-time) will return no data. Use relative terms ("Last week") which map to the *dataset's* last week (Oct 24-30).
*   **Rate Limits**: If using Free Tier models (Gemini/Llama), you may occasionally see `429` errors. The system auto-retries 3 times.

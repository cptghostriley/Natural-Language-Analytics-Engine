# Locobuzz AI Analytics System

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

## ⚠️ Important Notes

*   **Dataset Timeframe**: The included dataset covers **October 2025**. Queries for "today" (Real-time) will return no data. Use relative terms ("Last week") which map to the *dataset's* last week (Oct 24-30).
*   **Rate Limits**: If using Free Tier models (Gemini/Llama), you may occasionally see `429` errors. The system auto-retries 3 times.

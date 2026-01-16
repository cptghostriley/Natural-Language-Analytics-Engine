# Implementation Plan: Evolution of Natural Language Analytics System

This document outlines the systematic expansion of the analytics system into three phases: **Semantic Retrieval**, **Constrained NL-to-SQL**, and **Hybrid Reasoning**.

## ⚠️ Critical Prerequisite: Embedding Model Consistency
The current dataset contains pre-computed embeddings (likely Bedrock). For semantic retrieval to work, the **User Query** must be embedded using the **exact same model** as the dataset.
- **Action**: We will verify if we can access the original embedding model.
- **Fallback**: If the original model is inaccessible, we will modify the ingestion script to **re-embed** the `postcontent` using our available local (Ollama) or cloud (OpenAI) model during the ingestion phase. This ensures vector space alignment.

---

## Phase 1: Semantic Retrieval Layer (MANDATORY)

**Goal**: Enable live, meaning-based search over the dataset to find "needle in a haystack" evidence and qualitative insights.

### 1. Data Layer Changes (`src/ingest.py`)
- **Current State**: Embeddings are dropped before saving to DuckDB.
- **New State**:
  - Retain `embedding_vec` in the dataframe.
  - Store embeddings in a vector-capable store.
  - **Option A (Simpler/Native)**: Use DuckDB's `array` type if the `vss` extension is available, or standard cosine similarity UDFs (slower but acceptable for 100k rows if optimized).
  - **Option B (Performance)**: Save embeddings to a lightweight FAISS index or `chromadb` alongside the DuckDB file.
  - *Decision*: We will start with **DuckDB Native Arrays** + Cosine Similarity UDF for simplicity of deployment (single file), unless performance is unacceptable (>2s), in which case we switch to FAISS.

### 2. New Logic Node: `retrieve_semantic`
- **Input**: `query` (string), `filters` (optional time/metadata constraints).
- **Process**:
  1. Generate embedding for user `query` using the configured embedding model.
  2. Construct a vector similarity query:
     - `SELECT postcontent, createddate, list_cosine_similarity(embedding, ?) as score FROM posts ... ORDER BY score DESC LIMIT 5`
  3. Apply existing filters (e.g., date range) to the SQL `WHERE` clause to ensure semantic search respects time windows.
- **Output**: List of relevant posts (Content, Date, Score).

### 3. Graph Integration (`src/graph.py`)
- **Router Update**: Update `classify_query` to identify "semantic" intents:
  - `find_examples` ("Show me posts about...")
  - `complaint_search` (Specific complaints not covered by aggressive aggregation)
  - `vague_exploration` ("What are people saying about X?")
- **New Path**: Route these intents to `retrieve_semantic` node instead of `execute_analytics`.

---

## Phase 2: Constrained NL → SQL Layer (SAFE)

**Goal**: Allow flexible quantification without exposing the system to hallucinated SQL or dangerous operations.

### 1. Template Engine (`src/analytics.py`)
- Define strict templates. The LLM only fills in the `{slots}`.
- **Template 1 (Time-Series)**:
  ```sql
  SELECT {time_agg}(date) as time_bucket, {agg_func}({metric_col}) 
  FROM posts 
  WHERE {filter_conditions} 
  GROUP BY 1 ORDER BY 1
  ```
- **Template 2 (Category Breakdown)**:
  ```sql
  SELECT {category_col}, {agg_func}({metric_col}) 
  FROM posts 
  WHERE {filter_conditions} 
  GROUP BY 1 ORDER BY 2 DESC LIMIT 10
  ```

### 2. Validation Layer
- **Allowlist**:
  - `agg_func`: `COUNT`, `AVG`, `SUM` (only for specific cols)
  - `metric_col`: `sentiment`, `*`
  - `category_col`: `topic_id`, `sentiment` (label), `platform` (if exists)
- **Sanitization**:
  - AST Parser or Regex to confirm the query matches the template structure exactly.
  - Reject subqueries, JOINs, or drop/delete commands.

---

## Phase 3: Hybrid Analytics + Semantic Reasoning (ADVANCED)

**Goal**: Answer "Why?" by correlating quantitative anomalies with qualitative evidence.

### 1. Workflow Logic
This phase introduces a multi-step graph execution:

Start -> Classify -> **Hybrid_Router** -> [Analytics Node] -> [Semantic Node] -> Synthesize

### 2. Execution Flow (Example: "Why did sentiment drop last week?")
1. **Analytics Step**: 
   - Run `sentiment_trend` for "last week".
   - Identify the specific days with the lowest sentiment (e.g., Tuesday and Friday).
2. **Context Transfer**:
   - Pass the date range of the "drop" (Tuesday/Friday) to the Semantic Node.
   - Set filter: `date IN ('2025-10-xx', ...)` AND `sentiment = -1`.
3. **Semantic Step**:
   - Query: "Reason for negative sentiment" (embedded).
   - Retrieve top 10 negative posts from those specific anomaly days.
4. **Synthesis Step**:
   - LLM receives:
     - Analytics: "Sentiment dropped by 0.4 on Tuesday."
     - Semantic: "Posts from Tuesday mention 'app crash on login'."
   - Output: "Sentiment dropped on Tuesday primarily driven by user reports of login crashes..."

---

## LangGraph Node Responsibilities & Flows

### Updated State Definition
```python
class AnalyticsState(TypedDict):
    question: str
    query_type: str        # analytics, semantic, or hybrid
    filters: Dict          # time, specific filters
    sql_template: Dict     # For Phase 2 (optional)
    analytics_result: Any  # Quantitative Data
    semantic_result: Any   # Qualitative Data (List of Posts)
    final_answer: str
```

### Execution Flows

#### A. Analytics-Only Query
> "What is the sentiment trend?"
`Classify` -> `Resolve_Time` -> `Execute_Analytics` (SQL Aggregation) -> `Summarize`

#### B. Semantic-Only Query
> "Show me examples of bad service."
`Classify` -> `Resolve_Time` -> **`Retrieve_Semantic`** (Vector Search) -> `Summarize` (List & Contextualize)

#### C. Hybrid Query
> "Why is the volume so high today?"
`Classify` (Hybrid) -> `Resolve_Time` -> 
  1. **`Execute_Analytics`** (Volume Trend) -> 
  2. **`Check_Anomaly`** (Find peak date) ->
  3. **`Retrieve_Semantic`** (Filter by peak date, Search for "high volume context") ->
`Synthesize_Hybrid` (Explain peak using posts)

---

## Next Steps
1. **Approval**: Confirm this plan aligns with requirements.
2. **Implementation**: Begin Phase 1 (Ingestion update & Semantic Node).

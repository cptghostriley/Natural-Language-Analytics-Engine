from typing import TypedDict, Optional, Dict, Any, List
from langgraph.graph import StateGraph, END
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
import os
import json
from dotenv import load_dotenv

# Robustly load .env from project root
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(env_path, override=True)

from src.analytics import execute_analytics_query, get_dataset_bounds, execute_custom_query
from src.retrieve import search_semantic
from datetime import datetime, timedelta

# Debug LLM Config
# print(f"DEBUG: Model: {os.environ.get('LLM_MODEL')}")
key = os.environ.get('OPENAI_API_KEY')

# Fallback: Try Streamlit Secrets (for Cloud) if env var missing
if not key:
    try:
        import streamlit as st
        if "OPENAI_API_KEY" in st.secrets:
            # Load config from st.secrets
            os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
            if "OPENAI_API_BASE" in st.secrets:
                 os.environ["OPENAI_API_BASE"] = st.secrets["OPENAI_API_BASE"]
            if "LLM_MODEL" in st.secrets:
                 os.environ["LLM_MODEL"] = st.secrets["LLM_MODEL"]
            key = os.environ.get('OPENAI_API_KEY') # Refresh
    except Exception as e:
        pass

# 1. State Definition
class AnalyticsState(TypedDict):
    question: str
    query_type: Optional[str]
    filters: Optional[Dict[str, Any]]
    analytics_result: Optional[Any]
    semantic_result: Optional[List[Dict]] # Added field
    sql_spec: Optional[Dict[str, Any]] # Phase 2: Safe SQL Spec
    reasoning_trace: Optional[List[str]] # Phase 3: Hybrid Reasoning Trace
    final_answer: Optional[str]
    error: Optional[str]
    resolved_start: Optional[str]
    resolved_end: Optional[str]

# 2. LLM Setup
base_url = os.environ.get("OPENAI_API_BASE") 
model_name = os.environ.get("LLM_MODEL", "gpt-4o-mini")

if key:
    # Production / Cloud Mode (OpenAI, OpenRouter, Groq, etc.)
    print(f"Using Cloud LLM: {model_name} (Base: {base_url or 'Default'})")
    llm = ChatOpenAI(model=model_name, base_url=base_url, api_key=key, max_retries=3)
    summary_llm = llm.bind(temperature=0.3)
else:
    # Local / Dev Mode
    print("Using Local Ollama")
    try:
        llm = ChatOllama(model="qwen2.5:7b", temperature=0)
        summary_llm = llm
    except:
        # Fallback to OpenAI definition if Ollama fails init (though ChatOllama is lazy)
        llm = ChatOpenAI(api_key="sk-dummy", base_url="http://localhost:11434/v1")
        summary_llm = llm

# 3. Nodes

def classify_query(state: AnalyticsState):
    """
    Determines validity of query and extracts parameters.
    """
    question = state["question"]
    
    # We do NOT inject current date. Time is resolved relative to dataset.
    # We do NOT inject current date. Time is resolved relative to dataset.
    system_prompt = """You are an intent classifier for a social media analytics system.
    Supported Query Types:
    1. sentiment_trend 
    2. volume_trend
    3. top_topics
    4. complaints_analysis 
    5. weekly_summary
    6. influencer_analysis
    7. keyword_search ("Find posts about X" - uses specific keyword match)
    8. semantic_search ("Show examples...", "What are people saying...")
    9. custom_analytics (Safe Dynamic Aggregation: "How many...", "Average...")
    10. root_cause ("Why is sentiment low?", "Explain the drop in volume", "What caused the spike?")
    
    Task:
    - If the user asks "Why", "Reason for", "Cause of", or "Explain X": 
      -> Select "root_cause".
    
    1. Identify query_type. 
       - If qualitative/examples -> semantic_search
       - If standard report -> types 1-6
       - If specific aggregation -> custom_analytics
       - If reasoning/why -> root_cause
       
    2. Extract filters:
       - If explicit dates: "start_date", "end_date" (YYYY-MM-DD).
       - If relative time: "time_expr" (enum: "last_week", "this_week", "last_7_days", "this_month", "last_month").
       - If keyword search: "keyword" (the string to search).
       - Do NOT resolve dates yourself.
       
    3. If query_type is "custom_analytics", extract "sql_spec":
       - agg: "COUNT", "AVG", "SUM", "MIN", "MAX"
       - metric: "sentiment" (for avg/min/max), "*" (for count), "id"
       - group_by: "date", "week", "month", "year", "topic_id", "sentiment" (or null if scalar)
       - limit: integer (default 10)
    
    Output JSON ONLY:
    {{
        "query_type": "...",
        "filters": {{ ... }},
        "sql_spec": {{ ... }} (optional)
    }}
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])
    
    chain = prompt | llm | JsonOutputParser()
    
    try:
        result = chain.invoke({"question": question})
        return {
            "query_type": result.get("query_type", "unsupported"),
            "filters": result.get("filters", {}),
            "sql_spec": result.get("sql_spec", {})
        }
    except Exception as e:
        return {"error": str(e), "query_type": "error"}

def resolve_time_range(state: AnalyticsState):
    """
    Resolves relative dates based on Dataset Max Date.
    """
    filters = state.get("filters", {}) or {}
    q_type = state.get("query_type") # passed through
    
    if state.get("error") or q_type == "unsupported":
        return {} # Pass through error

    # Get Dataset Bounds
    try:
        bounds = get_dataset_bounds()
        dataset_max_str = bounds["max_date"]
        dataset_min_str = bounds["min_date"]
        
        # Parse
        if dataset_max_str == 'None':
             return {"error": "Dataset is empty."}
             
        # Robust parsing (handle YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)
        dataset_max_str = dataset_max_str.split(" ")[0]
        dataset_min_str = dataset_min_str.split(" ")[0]
        
        max_d = datetime.strptime(dataset_max_str, "%Y-%m-%d").date()
        min_d = datetime.strptime(dataset_min_str, "%Y-%m-%d").date()
    except Exception as e:
         return {"error": f"Failed to get dataset bounds: {e}"}

    # defaults
    start = filters.get("start_date")
    end = filters.get("end_date")
    time_expr = filters.get("time_expr")
    
    if start and end:
         pass
         
    elif time_expr:
        # Resolve logic
        if time_expr == "last_week" or time_expr == "last_7_days":
            end_d = max_d
            start_d = max_d - timedelta(days=6) 
        elif time_expr == "this_week":
            yr, wk, _ = max_d.isocalendar()
            start_d = datetime.fromisocalendar(yr, wk, 1).date()
            end_d = datetime.fromisocalendar(yr, wk, 7).date()
        elif time_expr == "this_month":
            start_d = max_d.replace(day=1)
            end_d = max_d 
        elif time_expr == "last_month":
            first_this = max_d.replace(day=1)
            last_prev = first_this - timedelta(days=1)
            start_d = last_prev.replace(day=1)
            end_d = last_prev
        else:
             start_d = None
             end_d = None
             
        if start_d:
            filters["start_date"] = str(start_d)
            filters["end_date"] = str(end_d)
            
    
    current_start = filters.get("start_date") or str(min_d)
    current_end = filters.get("end_date") or str(max_d)
    
    # Check if we didn't resolve anything explicitly but defaults were used
    # Ensure they are in filters so semantic search sees them if needed
    if not filters.get("start_date"):
         filters["start_date"] = current_start
    if not filters.get("end_date"):
         filters["end_date"] = current_end

    return {
        "filters": filters,
        "resolved_start": current_start,
        "resolved_end": current_end
    }

def execute_root_cause_analysis(state: AnalyticsState):
    """
    Phase 3: Hybrid Reasoning.
    1. Analyze Quantitative Trend (Analytics).
    2. Identify Anomalies (Lowest sentiment, Highest Spikes).
    3. Retrieve Qualitative Evidence for those specific moments (Semantic).
    """
    question = state["question"].lower()
    filters = state["filters"]
    trace = []
    
    # Step 1: determine what to analyze based on question
    # Default to sentiment trend if ambiguous
    metric_type = "sentiment_trend" 
    if "volume" in question or "traffic" in question:
        metric_type = "volume_trend"
        
    trace.append(f"Step 1: Analyzed {metric_type} to find anomalies.")
    
    # Run Analytics
    trend_data = execute_analytics_query(metric_type, filters)
    
    if not trend_data or isinstance(trend_data, dict) and "error" in trend_data:
        return {"error": "Could not fetch trend data for analysis.", "reasoning_trace": trace}
        
    # Step 2: Find Anomaly (Simplistic: Find Min/Max)
    # Convert to DataFrame for easy handling
    import pandas as pd
    try:
        df = pd.DataFrame(trend_data)
        if df.empty:
             return {"error": "Trend data is empty.", "reasoning_trace": trace}

        if metric_type == "sentiment_trend":
            # Find worst day
            target = df.loc[df['avg_sentiment'].idxmin()]
            target_date = str(target['date']).split(" ")[0] # YYYY-MM-DD
            target_val = target['avg_sentiment']
            anomaly_desc = f"Lowest sentiment ({target_val:.2f}) observed on {target_date}"
            
        else: # volume
            # Find highest volume
            target = df.loc[df['volume'].idxmax()]
            target_date = str(target['date']).split(" ")[0]
            target_val = target['volume']
            anomaly_desc = f"Highest volume ({target_val}) observed on {target_date}"
            
        trace.append(f"Step 2: Identified anomaly: {anomaly_desc}.")
        
        # Step 3: Targeted Semantic Search
        # We drill down into that specific date
        context_filters = filters.copy()
        context_filters["start_date"] = target_date
        context_filters["end_date"] = target_date # One day drill-down
        
        search_q = f"Reasons for {metric_type} on {target_date}"
        if "sentiment" in metric_type:
            search_q = "complaints and negative feedback"
            
        # Add reasoning to context for summary
        trace.append(f"Step 3: Retrieving evidence for {target_date}...")
        
        # We fetch slightly more documents for context
        semantic_data = search_semantic(search_q, filters=context_filters, top_k=5)
        
        return {
            "analytics_result": trend_data, # Full context
            "semantic_result": semantic_data, # Specific evidence
            "reasoning_trace": trace
        }
        
    except Exception as e:
        return {"error": f"Root cause analysis failed: {e}", "reasoning_trace": trace}

def execute_analytics(state: AnalyticsState):
    """
    Executes the deterministic analytics.
    """
    q_type = state["query_type"]
    filters = state["filters"]
    
    # Handle Root Cause Routing explicitly if not done in router
    if q_type == "root_cause":
        return execute_root_cause_analysis(state)
        
    if q_type == "custom_analytics":
        spec = state.get("sql_spec", {})
        if not spec:
            return {"error": "Custom analytics requested but no SQL spec generated."}
        # Merge resolved filters into spec filters
        spec["filters"] = {**spec.get("filters", {}), **filters}
        
        result = execute_custom_query(spec)
        return {"analytics_result": result}
        
    if q_type == "unsupported":
        return {"error": "Query type not supported."}
        
    result = execute_analytics_query(q_type, filters)
    return {"analytics_result": result}

def execute_semantic_search(state: AnalyticsState):
    """
    Executes semantic retrieval.
    """
    question = state["question"]
    filters = state["filters"]
    
    results = search_semantic(question, filters=filters, top_k=5)
    
    if isinstance(results, dict) and "error" in results:
        return {"error": results["error"], "semantic_result": []}
        
    return {"semantic_result": results}

def summarize_results(state: AnalyticsState):
    """
    Summarizes the analytics result into natural language.
    """
    result = state.get("analytics_result")
    semantic_res = state.get("semantic_result")
    q_type = state.get("query_type")
    question = state.get("question")
    trace = state.get("reasoning_trace", [])
    
    # Resolved timeframe
    start_d = state.get("resolved_start", "unknown")
    end_d = state.get("resolved_end", "unknown")
    
    if state.get("error"):
         return {"final_answer": f"I cannot answer that. {state.get('error')}"}

    if not result and not semantic_res:
         return {"final_answer": f"No data found for the period {start_d} to {end_d}."}
         
    system_prompt = """You are a data analyst helper. 
    User Question: {question}
    Query Type: {q_type}
    Timeframe: {start_d} to {end_d}
    
    Analysis Trace:
    {trace}
    
    Computed Data:
    {data}
    
    Semantic / Qualitative Evidence:
    {semantic_data}
    
    Task: Summarize the findings based ONLY on the data provided. 
    If semantic evidence is provided, cite examples.
    Do not hallunicate. Interpret the numbers for the user.
    
    MANDATORY: Start your response with: "Based on the most recent available data (from {start_d} to {end_d})..."
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Summarize this.")
    ])
    
    chain = prompt | summary_llm | StrOutputParser()
    try:
        data_str = json.dumps(result, indent=2, default=str) if result else "None"
        sem_str = json.dumps(semantic_res, indent=2, default=str) if semantic_res else "None"
        trace_str = "\\n".join(trace) if trace else "Standard Execution"
        
        answer = chain.invoke({
            "question": question,
            "q_type": q_type,
            "data": data_str,
            "semantic_data": sem_str,
            "start_d": start_d,
            "end_d": end_d,
            "trace": trace_str
        })
        return {"final_answer": answer}
    except Exception as e:
        return {"final_answer": f"Error generating summary: {e}"}

# 4. Routing
def route_step(state: AnalyticsState):
    if state.get("error"):
        return "summarize" 
    if state["query_type"] == "unsupported":
        state["error"] = "Unsupported query."
        return "summarize"
        
    if state["query_type"] == "semantic_search":
        return "retrieve"
        
    return "execute"

# 5. Graph Construction
workflow = StateGraph(AnalyticsState)

workflow.add_node("classify", classify_query)
workflow.add_node("resolve_time", resolve_time_range)
workflow.add_node("execute", execute_analytics)
workflow.add_node("retrieve", execute_semantic_search)
workflow.add_node("summarize", summarize_results)

workflow.set_entry_point("classify")

workflow.add_edge("classify", "resolve_time")

workflow.add_conditional_edges(
    "resolve_time",
    route_step,
    {
        "execute": "execute",
        "retrieve": "retrieve",
        "summarize": "summarize"
    }
)

workflow.add_edge("execute", "summarize")
workflow.add_edge("retrieve", "summarize")
workflow.add_edge("summarize", END)

app = workflow.compile()

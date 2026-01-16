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
    system_prompt = """You are an intent classifier for a social media analytics system.
    Supported Query Types:
    1. sentiment_trend 
    2. volume_trend
    3. top_topics
    4. complaints_analysis 
    5. weekly_summary
    6. influencer_analysis
    7. keyword_search ("Find posts about X" - uses specific keyword match)
    8. semantic_search ("What are people saying about X?", "Show me examples of bad service", "Why is sentiment low?")
    9. custom_analytics (Safe Dynamic Aggregation)
       Use this for ANY quantitative question not covered above, such as:
       - "How many negative posts last week?"
       - "Average sentiment by topic"
       - "Volume per month"
    
    Task:
    1. Identify query_type. 
       - If qualitative/examples -> semantic_search
       - If standard report -> types 1-6
       - If specific aggregation -> custom_analytics
       
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

def execute_analytics(state: AnalyticsState):
    """
    Executes the deterministic analytics.
    """
    q_type = state["query_type"]
    filters = state["filters"]
    
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
        
        answer = chain.invoke({
            "question": question,
            "q_type": q_type,
            "data": data_str,
            "semantic_data": sem_str,
            "start_d": start_d,
            "end_d": end_d
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

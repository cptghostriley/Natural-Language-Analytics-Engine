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

load_dotenv(override=True)

from src.analytics import execute_analytics_query, get_dataset_bounds
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
    final_answer: Optional[str]
    error: Optional[str]
    resolved_start: Optional[str]
    resolved_end: Optional[str]

# 2. LLM Setup
api_key = os.environ.get("OPENAI_API_KEY")
base_url = os.environ.get("OPENAI_API_BASE") # Optional, for OpenRouter/Groq
model_name = os.environ.get("LLM_MODEL", "gpt-4o-mini")

if api_key:
    # Production / Cloud Mode (OpenAI, OpenRouter, Groq, etc.)
    print(f"Using Cloud LLM: {model_name} (Base: {base_url or 'Default'})")
    llm = ChatOpenAI(model=model_name, base_url=base_url, api_key=api_key, max_retries=3)
    summary_llm = llm.bind(temperature=0.3)
else:
    # Local / Dev Mode
    try:
        print("Using Local Ollama (Qwen 2.5 7B)")
        llm = ChatOllama(model="qwen2.5:7b", format="json") 
        summary_llm = ChatOllama(model="qwen2.5:7b")
    except Exception as e:
        print(f"Warning: Failed to init Ollama ({e}). Falling back to OpenAI (expecting key).")
        llm = ChatOpenAI(model="gpt-4o-mini")
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
    7. keyword_search ("Find posts about X")
    
    Task:
    1. Identify query_type. If unknown, use "unsupported".
    2. Extract filters:
       - If explicit dates: "start_date", "end_date" (YYYY-MM-DD).
       - If relative time: "time_expr" (enum: "last_week", "this_week", "last_7_days", "this_month", "last_month").
       - If keyword search: "keyword" (the string to search).
       - Ignore vague terms like "campaign" (return empty filters).
       - Do NOT resolve dates yourself.
    
    Output JSON ONLY:
    {{
        "query_type": "...",
        "filters": {{ ... }}
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
            "filters": result.get("filters", {})
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
             
        max_d = datetime.strptime(dataset_max_str, "%Y-%m-%d").date()
        min_d = datetime.strptime(dataset_min_str, "%Y-%m-%d").date()
    except Exception as e:
         return {"error": f"Failed to get dataset bounds: {e}"}

    # defaults
    start = filters.get("start_date")
    end = filters.get("end_date")
    time_expr = filters.get("time_expr")
    
    if start and end:
         # explicit - validate bounds?
         # Contract: "If requested time window falls partially or fully outside... reject"
         # We will parse and check inside execute or here? Here is better.
         pass
         
    elif time_expr:
        # Resolve logic
        if time_expr == "last_week" or time_expr == "last_7_days":
            # 7-day window ending at dataset_max_date
            end_d = max_d
            start_d = max_d - timedelta(days=6) 
        elif time_expr == "this_week":
            # Week containing dataset_max_date
            # ISO Calendar
            yr, wk, _ = max_d.isocalendar()
            start_d = datetime.fromisocalendar(yr, wk, 1).date()
            end_d = datetime.fromisocalendar(yr, wk, 7).date()
        elif time_expr == "this_month":
            start_d = max_d.replace(day=1)
            end_d = max_d # Or end of month? "Analytics operate ONLY within resolved window". Data ends at max.
        elif time_expr == "last_month":
            # month preceeding max_d
            first_this = max_d.replace(day=1)
            last_prev = first_this - timedelta(days=1)
            start_d = last_prev.replace(day=1)
            end_d = last_prev
        else:
             # Unknown expr
             start_d = None
             end_d = None
             
        if start_d:
            filters["start_date"] = str(start_d)
            filters["end_date"] = str(end_d)
            
    # If no filters, we default to full range? Or last 30 days?
    # Contract: "All analytics must operate ONLY within the resolved time window."
    # If input had no filters, implies "All Data".
    # We should explicitly set start/end to min/max if None?
    # Usually "Top Topics" implies all data.
    # We'll leave it empty to imply "All", but strictly the User might want to know the range.
    # We will add "resolved_window" to state for the Summary.
    
    current_start = filters.get("start_date") or str(min_d)
    current_end = filters.get("end_date") or str(max_d)
    
    # Store explicit range for disclosure
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
    
    if q_type == "unsupported":
        return {"error": "Query type not supported. Please ask about sentiment, volume, topics, or complaints."}
        
    result = execute_analytics_query(q_type, filters)
    return {"analytics_result": result}

def summarize_results(state: AnalyticsState):
    """
    Summarizes the analytics result into natural language.
    """
    result = state.get("analytics_result")
    q_type = state.get("query_type")
    question = state.get("question")
    
    # Resolved timeframe
    start_d = state.get("resolved_start", "unknown")
    end_d = state.get("resolved_end", "unknown")
    
    if state.get("error") or not result:
        filters = state.get("filters", {})
        if not result and filters.get("start_date"):
             date_range = f"{filters.get('start_date')} to {filters.get('end_date') or 'now'}"
             return {"final_answer": f"No data found for the period {date_range}. The dataset coverage may not include this timeframe."}
             
        error_msg = state.get("error", "No analytics results found.")
        return {"final_answer": f"I cannot answer that. {error_msg}"}
        
    system_prompt = """You are a data analyst helper. 
    User Question: {question}
    Query Type: {q_type}
    Timeframe: {start_d} to {end_d}
    
    Computed Data:
    {data}
    
    Task: Summarize the findings based ONLY on the data provided. 
    Do not hallunicate. Interpret the numbers for the user.
    
    MANDATORY: Start your response with: "Based on the most recent available data (from {start_d} to {end_d})..."
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Summarize this.")
    ])
    
    chain = prompt | summary_llm | StrOutputParser()
    try:
        data_str = json.dumps(result, indent=2, default=str)
        answer = chain.invoke({
            "question": question,
            "q_type": q_type,
            "data": data_str,
            "start_d": start_d,
            "end_d": end_d
        })
        return {"final_answer": answer}
    except Exception as e:
        return {"final_answer": f"Error generating summary: {e}"}

# 4. Routing
def route_step(state: AnalyticsState):
    if state.get("error"):
        return "summarize" # Go to summarize (which handles errors) or end
    if state["query_type"] == "unsupported":
        state["error"] = "Unsupported query."
        return "summarize"
        
    return "execute"

# 5. Graph Construction
workflow = StateGraph(AnalyticsState)

workflow.add_node("classify", classify_query)
workflow.add_node("resolve_time", resolve_time_range)
workflow.add_node("execute", execute_analytics)
workflow.add_node("summarize", summarize_results)

workflow.set_entry_point("classify")

workflow.add_edge("classify", "resolve_time")

workflow.add_conditional_edges(
    "resolve_time",
    route_step,
    {
        "execute": "execute",
        "summarize": "summarize"
    }
)

workflow.add_edge("execute", "summarize")
workflow.add_edge("summarize", END)

app = workflow.compile()

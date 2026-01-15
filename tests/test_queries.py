import sys
import os
import json
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.graph import app

def run_test(question):
    print(f"\n{'='*50}")
    print(f"TEST QUERY: {question}")
    print(f"{'='*50}")
    
    try:
        inputs = {"question": question}
        print("Invoking Agent...")
        result = app.invoke(inputs)
        
        query_type = result.get("query_type")
        filters = result.get("filters")
        analytics = result.get("analytics_result")
        final = result.get("final_answer")
        error = result.get("error")
        
        print(f"Intent classified: {query_type}")
        print(f"Filters: {filters}")
        
        if error:
            print(f"!! GRAPH ERROR: {error}")
            
        if analytics:
            if isinstance(analytics, list):
                print(f"Analytics Data (Rows): {len(analytics)}")
                if len(analytics) > 0:
                    print(f"Sample: {analytics[0]}")
            elif isinstance(analytics, dict):
                print(f"Analytics Data (Dict): Keys {list(analytics.keys())}")
        else:
            print("Analytics Data: None/Empty")
            
        print(f"Final Answer: {final}")
        print("STATUS: SUCCESS" if final and not error else "STATUS: FAILED_LOGIC")
        
    except Exception as e:
        print(f"STATUS: CRASHED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cases = [
        "What are the top topics?", 
        "What was the sentiment trend for our brand last week?",
        "Which influencers drove the most engagement during our campaign?",
        "What are users complaining about?",
        "What is the sentiment trend in October 2025?" # Positive control for data
    ]
    
    for case in test_cases:
        run_test(case)

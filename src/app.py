import streamlit as st
import pandas as pd
import json
import sys
import os

# Ensure project root is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.graph import app as graph_app

st.set_page_config(page_title="Natural Language Analytics Engine", layout="wide")

st.title("Natural Language Analytics Engine")
st.markdown("Ask questions about your social media data. Powered by LangGraph & Deterministic Analytics.")

with st.sidebar:
    st.header("Supported Queries")
    st.markdown("""
    - **Sentiment Trend**: "How is sentiment changing?"
    - **Volume Trend**: "Post volume over time?"
    - **Top Topics**: "What are people talking about?"
    - **Complaints**: "What are the main complaints?"
    - **Influencers**: "Who drove engagement?"
    - **Weekly Summary**: "Summarize last week."
    """)

question = st.text_input("Enter your question:", placeholder="e.g., What was the sentiment trend last week?")

if st.button("Analyze") and question:
    with st.spinner("Analyzing..."):
        try:
            # invoke graph
            initial_state = {"question": question}
            result = graph_app.invoke(initial_state)
            
            error = result.get("error")
            if error:
                st.error(f"Analysis Error: {error}")
                # Halts execution here to prevent showing stale data or NameErrors
                st.stop()
                
            final_answer = result.get("final_answer", "No answer generated.")
            analytics_data = result.get("analytics_result")
            query_type = result.get("query_type")
            filters = result.get("filters")
            
            with st.expander("Query Interpretation Details"):
                st.write(f"**Intent:** {query_type}")
                st.write(f"**Filters:** {filters}")
                
            st.markdown("### Answer")
            st.write(final_answer)
            
            st.markdown("---")
            st.markdown("### Evidence (Computed Data)")
            
            if analytics_data:
                # Display based on type
                if isinstance(analytics_data, list):
                    df = pd.DataFrame(analytics_data)
                    st.dataframe(df)
                    
                    # specific viz
                    if query_type == "sentiment_trend" and "date" in df.columns and "avg_sentiment" in df.columns:
                        st.line_chart(df.set_index("date")["avg_sentiment"])
                    
                    elif query_type == "volume_trend" and "date" in df.columns:
                        st.bar_chart(df.set_index("date")["volume"])
                        
                    elif query_type in ["top_topics", "complaints_analysis"]:
                        if "count" in df.columns:
                            st.bar_chart(df.set_index("topic_id")["count"])

                    elif query_type == "influencer_analysis":
                        if "count" in df.columns and "handle" in df.columns:
                            st.bar_chart(df.set_index("handle")["count"])
                            
                elif isinstance(analytics_data, dict):
                    st.json(analytics_data)
            else:
                st.warning("No data computed for this query.")
                
        except Exception as e:
            st.error(f"System Error: {e}")


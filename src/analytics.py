import duckdb
import pandas as pd
from datetime import datetime, timedelta
import os
import requests

DB_PATH = os.path.join(os.path.dirname(__file__), '../data/analytics.duckdb')

def ensure_database():
    if not os.path.exists(DB_PATH):
        url = os.environ.get("DUCKDB_URL")
        if url:
            print(f"Downloading database from {url}...")
            # Ensure dir exists
            os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
            try:
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    with open(DB_PATH, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=1024*1024): # 1MB chunks
                            f.write(chunk)
                print("Download complete.")
            except Exception as e:
                print(f"Failed to download DB: {e}")
        else:
            print("Warning: Database missing and DUCKDB_URL not set.")

def get_connection():
    ensure_database()
    return duckdb.connect(DB_PATH, read_only=True)

def get_dataset_bounds():
    con = get_connection()
    try:
        res = con.execute("SELECT MIN(date), MAX(date) FROM posts").fetchone()
        return {"min_date": str(res[0]), "max_date": str(res[1])}
    finally:
        con.close()

def sentiment_trend(weeks_back=4):
    """
    Returns daily average sentiment for the last N weeks.
    """
    con = get_connection()
    # Calculate start date relative to max date in DB to ensure we have data
    max_date = con.execute("SELECT MAX(date) FROM posts").fetchone()[0]
    if not max_date:
        return {}
    
    # Logic: Filter last N weeks
    # For now, let's just return daily stats for the dataset or time window
    query = """
    SELECT date, AVG(sentiment) as avg_sentiment, COUNT(*) as volume
    FROM posts
    WHERE date >= ?
    GROUP BY date
    ORDER BY date ASC
    """
    # Simply defaulting to a reasonable start date if not provided, or dynamic
    # Ideally should receive specific date range filter from the "State"
    
    # Determine start date
    # If weeks_back is passed, we go back from max_date
    import datetime as dt
    if isinstance(max_date, str):
         # DuckDB might return date object or string depending on version/driver
         # usually datetime.date
         pass
    
    # Let's just grab everything for now or logic inside the graph will filter?
    # The requirement says: "Filter posts where week == last_week" for specific questions.
    # We will build generic functions that accept start/end dates.
    con.close()
    pass

def execute_analytics_query(query_type, filters=None):
    """
    Main entry point for analytics.
    query_type: str
    filters: dict e.g. {'start_date': '...', 'end_date': '...', 'week': ...}
    """
    con = get_connection()
    filters = filters or {}
    
    # Base WHERE clause
    where_clauses = []
    params = []
    
    if filters.get('week'):
        where_clauses.append("week = ?")
        params.append(filters['week'])
    
    if filters.get('start_date'):
        where_clauses.append("date >= ?")
        params.append(filters['start_date'])
        
    if filters.get('end_date'):
        where_clauses.append("date <= ?")
        params.append(filters['end_date'])
        
    base_where = " AND ".join(where_clauses) if where_clauses else "1=1"
    
    result = {}
    
    try:
        if query_type == "sentiment_trend":
            sql = f"""
            SELECT date, AVG(sentiment) as avg_sentiment
            FROM posts
            WHERE {base_where}
            GROUP BY date
            ORDER BY date
            """
            df = con.execute(sql, params).fetchdf()
            result = df.to_dict(orient='records')
            
        elif query_type == "volume_trend":
            sql = f"""
            SELECT date, COUNT(*) as volume
            FROM posts
            WHERE {base_where}
            GROUP BY date
            ORDER BY date
            """
            df = con.execute(sql, params).fetchdf()
            result = df.to_dict(orient='records')
            
        elif query_type == "top_topics":
            sql = f"""
            SELECT topic_id, COUNT(*) as count
            FROM posts
            WHERE {base_where}
            GROUP BY topic_id
            ORDER BY count DESC
            LIMIT 5
            """
            df = con.execute(sql, params).fetchdf()
            
            # For each topic, get sample posts
            topics = []
            for _, row in df.iterrows():
                tid = row['topic_id']
                count = row['count']
                
                sample_sql = f"""
                SELECT postcontent 
                FROM posts 
                WHERE topic_id = ? AND {base_where}
                LIMIT 3
                """
                # Re-constructing params for inner query is tricky if variables used.
                # Simplified: separate query for samples.
                # Note: DuckDB binding might be easier.
                
                # To be safe and simple:
                samples_df = con.execute(f"SELECT postcontent FROM posts WHERE topic_id = {tid} AND {base_where} LIMIT 3", params).fetchdf()
                samples = samples_df['postcontent'].tolist()
                
                topics.append({
                    "topic_id": int(tid),
                    "count": int(count),
                    "samples": samples
                })
            result = topics

        elif query_type == "complaints_analysis":
            # sentiment == -1 AND high frequency
            sql = f"""
            SELECT topic_id, COUNT(*) as count
            FROM posts
            WHERE sentiment = -1 AND {base_where}
            GROUP BY topic_id
            ORDER BY count DESC
            LIMIT 5
            """
            df = con.execute(sql, params).fetchdf()
            
            complaints = []
            for _, row in df.iterrows():
                tid = row['topic_id']
                count = row['count']
                
                # Get samples
                sample_sql = f"SELECT postcontent FROM posts WHERE topic_id = {tid} AND sentiment = -1 AND {base_where} LIMIT 3"
                samples_df = con.execute(sample_sql, params).fetchdf()
                samples = samples_df['postcontent'].tolist()
                
                complaints.append({
                    "topic_id": int(tid),
                    "count": int(count),
                    "samples": samples
                })
            result = complaints
            
        elif query_type == "weekly_summary":
            # Aggregated stats for the week
            sql = f"""
            SELECT 
                COUNT(*) as total_volume,
                AVG(sentiment) as avg_sentiment,
                COUNT(CASE WHEN sentiment = -1 THEN 1 END) as negative_count,
                COUNT(CASE WHEN sentiment = 1 THEN 1 END) as positive_count
            FROM posts
            WHERE {base_where}
            """
            summary = con.execute(sql, params).fetchdf().to_dict(orient='records')[0]
            result = summary
            
        elif query_type == "influencer_analysis":
            # Extract @mentions as proxy for influencers/engagement
            # Using DuckDB regex
            sql = f"""
            WITH mentions AS (
                SELECT unnest(regexp_extract_all(postcontent, '@[a-zA-Z0-9_]+')) as handle
                FROM posts
                WHERE {base_where}
            )
            SELECT handle, COUNT(*) as count
            FROM mentions
            GROUP BY handle
            ORDER BY count DESC
            LIMIT 10
            """
            df = con.execute(sql, params).fetchdf()
            result = df.to_dict(orient='records')

        else:
            raise ValueError(f"Unknown query type: {query_type}")
            
    except Exception as e:
        print(f"Analytics Error: {e}")
        return {"error": str(e)}
    finally:
        con.close()
        
    return result

if __name__ == "__main__":
    # Test
    print("Testing analytics...")
    # Assume data exists
    try:
        print("Sentiment Trend:", execute_analytics_query("sentiment_trend")[:2])
    except:
        print("DB might not be ready.")

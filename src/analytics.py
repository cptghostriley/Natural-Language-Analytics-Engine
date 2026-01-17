import duckdb
import pandas as pd
from datetime import datetime, timedelta
import os
import requests

DB_PATH = os.path.join(os.path.dirname(__file__), '../data/analytics.duckdb')
DB_FULL_PATH = os.path.join(os.path.dirname(__file__), '../data/analytics_large.duckdb')

def ensure_database():
    if os.path.exists(DB_PATH):
        # Check if valid (simple size check > 1MB)
        if os.path.getsize(DB_PATH) < 1024 * 1024:
            print(f"Database found but too small ({os.path.getsize(DB_PATH)} bytes). Re-downloading...")
            os.remove(DB_PATH)
        else:
            return

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
            
            # Verify size check again
            if os.path.exists(DB_PATH) and os.path.getsize(DB_PATH) < 1024:
                 print("Warning: Downloaded DB seems too small. Check DUCKDB_URL.")
                 
        except Exception as e:
            print(f"Failed to download DB: {e}")
            # Don't leave corrupted file
            if os.path.exists(DB_PATH):
                os.remove(DB_PATH)
    else:
        print("Warning: Database missing and DUCKDB_URL not set.")

def get_connection():
    # WE USE THE RE-INDEXED DB (analytics.duckdb)
    # It now contains both the data AND the valid 384-dim embeddings.
    # We skip 'analytics_large.duckdb' to avoid the old incompatible embeddings.
    ensure_database()
    return duckdb.connect(DB_PATH, read_only=True)

def get_dataset_bounds():
    con = get_connection()
    try:
        # Cast to DATE to remove time components at source
        res = con.execute("SELECT CAST(MIN(date) AS DATE), CAST(MAX(date) AS DATE) FROM posts").fetchone()
        return {"min_date": str(res[0]), "max_date": str(res[1])}
    finally:
        con.close()



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

        elif query_type == "keyword_search":
            keyword = filters.get("keyword")
            if not keyword:
                return {"error": "No keyword provided"}
            
            # Sanitized ILIKE
            search_term = f"%{keyword}%"
            # Add to params
            params.append(search_term) 
            
            sql = f"""
            SELECT postcontent, createddate, sentiment
            FROM posts
            WHERE {base_where} AND postcontent ILIKE ?
            ORDER BY createddate DESC
            LIMIT 10
            """
            
            df = con.execute(sql, params).fetchdf()
            # Convert date
            if 'createddate' in df.columns:
                df['createddate'] = df['createddate'].astype(str)
                
            result = df.to_dict(orient='records')

        else:
            raise ValueError(f"Unknown query type: {query_type}")
            
    except Exception as e:
        print(f"Analytics Error: {e}")
        return {"error": str(e)}
    finally:
        con.close()
        
    return result

# ---------------------------------------------------------
# PHASE 2: Safe Constrained SQL Engine
# ---------------------------------------------------------

ALLOWED_AGGS = {"COUNT", "AVG", "SUM", "MIN", "MAX"}
ALLOWED_METRICS = {"*", "sentiment", "id"}
ALLOWED_GROUPS = {"date", "week", "month", "year", "topic_id", "sentiment"}

def validate_sql_components(spec):
    """
    Validates components against allowlists.
    spec = {
        "agg": "AVG",
        "metric": "sentiment",
        "group_by": "date",
        "filters": {...}
    }
    """
    agg = (spec.get("agg") or "COUNT").upper()
    metric = (spec.get("metric") or "*").lower()
    group = (spec.get("group_by") or "").lower()
    
    if agg not in ALLOWED_AGGS:
        raise ValueError(f"Aggregation '{agg}' not allowed.")
        
    if metric not in ALLOWED_METRICS:
        raise ValueError(f"Metric '{metric}' not allowed.")
    
    # Check if group is allowed (empty is fine for scalar)
    if group and group not in ALLOWED_GROUPS:
         raise ValueError(f"Grouping by '{group}' not allowed.")
         
    return True

def generate_safe_sql(spec):
    """
    Generates SQL from a validated spec string/dict.
    Supports 3 templates:
    1. Time Series (Group by time)
    2. Category Breakdown (Group by topic/sentiment)
    3. Scalar (Total count/avg)
    """
    validate_sql_components(spec)
    
    agg = (spec.get("agg") or "COUNT").upper()
    metric = spec.get("metric") or "*"
    group = spec.get("group_by")
    limit = spec.get("limit", 10)
    filters = spec.get("filters", {})
    
    # 1. Build Base Where
    where_clauses = []
    params = []
    
    if filters.get('start_date'):
        where_clauses.append("date >= ?")
        params.append(filters['start_date'])
    if filters.get('end_date'):
        where_clauses.append("date <= ?")
        params.append(filters['end_date'])
    # Can extend with other safe filters here
    
    base_where = " AND ".join(where_clauses) if where_clauses else "1=1"
    
    # 2. Select Template
    sql = ""
    
    if group:
        # Template 1 & 2: Grouping (Time or Category)
        # Handle 'sentiment' metric special case
        metric_sql = "*" if metric == "*" else metric
        
        sql = f"""
        SELECT {group}, {agg}({metric_sql}) as val
        FROM posts
        WHERE {base_where}
        GROUP BY {group}
        ORDER BY val DESC
        LIMIT {limit}
        """
        # If time series, usually we want order by time, not value
        if group in ["date", "week", "month", "year"]:
             sql = f"""
            SELECT {group}, {agg}({metric_sql}) as val
            FROM posts
            WHERE {base_where}
            GROUP BY {group}
            ORDER BY {group}
            """
            
    else:
        # Template 3: Scalar
        metric_sql = "*" if metric == "*" else metric
        sql = f"""
        SELECT {agg}({metric_sql}) as val
        FROM posts
        WHERE {base_where}
        """
        
    return sql, params

def execute_custom_query(spec):
    """
    Executes a safe custom query.
    """
    try:
        sql, params = generate_safe_sql(spec)
        con = get_connection()
        try:
             df = con.execute(sql, params).fetchdf()
             # Convert numeric/date types for JSON serialization
             if 'date' in df.columns:
                 df['date'] = df['date'].astype(str)
                 
             return df.to_dict(orient='records')
        finally:
             con.close()
    except Exception as e:
        return {"error": f"Custom Query Failed: {e}"}

if __name__ == "__main__":
    # Test
    print("Testing analytics...")
    # Assume data exists
    try:
        print("Sentiment Trend:", execute_analytics_query("sentiment_trend")[:2])
    except:
        print("DB might not be ready.")

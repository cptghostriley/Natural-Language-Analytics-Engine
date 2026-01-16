import duckdb
import os
import numpy as np
import json
from datetime import datetime

# DB PATH Logic from analytics.py
DB_PATH = os.path.join(os.path.dirname(__file__), '../data/analytics.duckdb')
DB_FULL_PATH = os.path.join(os.path.dirname(__file__), '../data/analytics_large.duckdb')

def get_db_path():
    # We always use the DB we just re-indexed
    return DB_PATH

# Embedding Models
# Ideally this should match the ingestion model.
# If using the CSV from Bedrock, we need a Bedrock compatible function.
# For now, we provide a placeholder wrapper.

def get_embedding(text):
    """
    Generates embedding using the local consistent model (all-MiniLM-L6-v2).
    """
    try:
        from sentence_transformers import SentenceTransformer
        # We must use the same model as reindex.py
        model = SentenceTransformer('all-MiniLM-L6-v2') 
        return model.encode(text).tolist()
    except Exception as e:
        print(f"Local Embedding failed: {e}")
        return []

def search_semantic(query, filters=None, top_k=5):
    """
    Performs vector similarity search on DuckDB.
    """
    filters = filters or {}
    
    # 1. Get query embedding
    q_vec = get_embedding(query)
    if not q_vec:
        return {"error": "Could not generate embedding for query."}
    
    db_path = get_db_path()
    con = duckdb.connect(db_path, read_only=True)
    
    try:
        # Build Where Clause
        where_clauses = ["embedding IS NOT NULL"]
        params = []
        
        if filters.get("start_date"):
            where_clauses.append("date >= ?")
            params.append(filters["start_date"])
        
        if filters.get("end_date"):
            where_clauses.append("date <= ?")
            params.append(filters["end_date"])
            
        where_str = " AND ".join(where_clauses)
        
        # Consistent Dimension Check
        # We now expect 384 dimensions from all-MiniLM-L6-v2
        
        sql = f"""
        SELECT 
            postcontent, 
            createddate, 
            topic_id,
            list_cosine_similarity(CAST(embedding AS FLOAT[]), ?) as score
        FROM posts
        WHERE {where_str} 
          AND len(CAST(embedding AS FLOAT[])) = 384
        ORDER BY score DESC
        LIMIT ?
        """
        
        # params: [q_vec, filter_params..., top_k]
        query_params = [q_vec] + params + [top_k]
        
        df = con.execute(sql, query_params).fetchdf()
        
        # Convert to dict
        if 'createddate' in df.columns:
            df['createddate'] = df['createddate'].astype(str)
            
        results = df.to_dict(orient='records')
        return results
        
        
    except Exception as e:
        return {"error": f"Semantic Search Error: {e}"}
    finally:
        con.close()

if __name__ == "__main__":
    # Test
    print("Testing semantic search...")
    res = search_semantic("app crash login")
    print(res)

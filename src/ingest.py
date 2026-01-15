import pandas as pd
import numpy as np
import json
import ast
from textblob import TextBlob
from sklearn.cluster import KMeans
import duckdb
import os
from datetime import datetime

# Configuration
INPUT_CSV = "october_data_bedrock_embeddings_part_0_176748_20251030_225407.csv"
DB_PATH = "data/analytics.duckdb"

def get_sentiment(text):
    if not isinstance(text, str):
        return 0
    try:
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.1:
            return 1
        elif polarity < -0.1:
            return -1
        else:
            return 0
    except:
        return 0

def parse_embedding(emb_str):
    try:
        # attributes might be truncated or malformed in CSV if not careful, 
        # but assuming standard list string format
        if pd.isna(emb_str):
            return None
        return json.loads(emb_str)
    except:
        try:
            return ast.literal_eval(emb_str)
        except:
            return None

def process_data():
    print("Loading data...")
    # Read csv using pandas
    # On error_bad_lines, we skip to avoid crashing on malformed CSV rows
    try:
        df = pd.read_csv(INPUT_CSV, on_bad_lines='skip')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"Data loaded: {len(df)} records")
    
    # 1. Date processing
    print("Processing dates...")
    df['createddate'] = pd.to_datetime(df['createddate'], errors='coerce')
    df = df.dropna(subset=['createddate'])
    
    df['date'] = df['createddate'].dt.date
    df['week'] = df['createddate'].dt.isocalendar().week
    df['year'] = df['createddate'].dt.year 
    df['month'] = df['createddate'].dt.strftime('%Y-%m')
    
    # 2. Sentiment analysis
    print("Computing sentiment...")
    df['sentiment'] = df['postcontent'].apply(get_sentiment)
    
    # 3. Topic Clustering
    print("Parsing embeddings...")
    
    def safe_parse(x):
        try:
            val = json.loads(x)
            if isinstance(val, list) and len(val) > 0:
                return val
            return None
        except:
             try:
                 val = ast.literal_eval(x)
                 if isinstance(val, list) and len(val) > 0:
                     return val
                 return None
             except:
                 return None

    # Apply parsing
    df['embedding_vec'] = df['embedding'].apply(safe_parse)
    
    # Filter valid
    valid_mask = df['embedding_vec'].notna()
    print(f"Valid embeddings found: {valid_mask.sum()} / {len(df)}")
    
    if valid_mask.sum() > 100: # Need enough data to cluster
        try:
            # Check dimension consistency
            # Get first valid length
            first_valid = df.loc[valid_mask, 'embedding_vec'].iloc[0]
            dim = len(first_valid)
            
            # Filter matches dim
            # This is slow but safe
            def check_dim(x):
                return x is not None and len(x) == dim
            
            dim_mask = df['embedding_vec'].apply(check_dim)
            print(f"Embeddings with dimension {dim}: {dim_mask.sum()}")
            
            matrix = np.vstack(df.loc[dim_mask, 'embedding_vec'].values)
            
            print("Clustering topics (K=20)...")
            kmeans = KMeans(n_clusters=20, random_state=42, n_init=5) # n_init lower for speed
            clusters = kmeans.fit_predict(matrix)
            
            df.loc[dim_mask, 'topic_id'] = clusters
            df.loc[~dim_mask, 'topic_id'] = -1
        except Exception as e:
            print(f"Clustering failed: {e}")
            df['topic_id'] = -1
    else:
        print("Not enough valid embeddings to cluster.")
        df['topic_id'] = -1
        
    df['topic_id'] = df['topic_id'].astype(int)
    
    # 4. Storage
    print("Saving to DuckDB...")
    con = duckdb.connect(DB_PATH)
    
    # Convert dates to string or datetime for DuckDB compatibility if needed
    # DuckDB handles pandas timestamp well
    
    # Drop object column
    df_save = df.drop(columns=['embedding_vec'])
    
    con.execute("CREATE OR REPLACE TABLE posts AS SELECT * FROM df_save")
    # Verify
    count = con.execute("SELECT count(*) FROM posts").fetchone()[0]
    print(f"Saved {count} rows to database.")
    
    con.close()
    print("Ingestion complete.")

if __name__ == "__main__":
    process_data()

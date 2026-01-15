import duckdb
try:
    con = duckdb.connect("data/analytics.duckdb", read_only=True)
    print("Sample Post:")
    print(con.execute("SELECT postcontent FROM posts LIMIT 1").fetchone())
    
    print("\nRegex Test:")
    # Note: DuckDB regexp_extract_all returns list. unnest expands it.
    res = con.execute("SELECT unnest(regexp_extract_all(postcontent, '@[a-zA-Z0-9_]+')) as handle FROM posts LIMIT 5").fetchall()
    print(f"Handles found: {res}")
except Exception as e:
    print(f"Error: {e}")

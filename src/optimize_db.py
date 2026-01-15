import duckdb
import os

db_path = "data/analytics.duckdb"

print("Optimizing database for GitHub deployment...")
con = duckdb.connect(db_path)
con.execute("ALTER TABLE posts DROP COLUMN IF EXISTS embedding")
con.execute("VACUUM")
con.close()

size_mb = os.path.getsize(db_path) / (1024*1024)
print(f"Optimization Complete. New Size: {size_mb:.2f} MB")

import duckdb
import os

target = "data/analytics_small.duckdb"
source = "data/analytics.duckdb"

if os.path.exists(target):
    os.remove(target)

print("Creating shrinked database...")
con = duckdb.connect(target)
con.execute(f"ATTACH '{source}' AS old")
con.execute("CREATE TABLE posts AS SELECT id, uniqueid, postcontent, createddate, sentiment, topic_id FROM old.posts")
con.close()

size = os.path.getsize(target) / (1024*1024)
print(f"Shrinked Size: {size:.2f} MB")

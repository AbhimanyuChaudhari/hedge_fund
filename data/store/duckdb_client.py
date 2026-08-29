import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import duckdb
from config.settings import settings

class DuckDBClient:
    def __init__(self):
        self.conn = duckdb.connect()
        self.conn.execute(f"""
            CREATE SECRET IF NOT EXISTS s3_secret (
                TYPE S3,
                KEY_ID '{settings.aws_access_key_id}',
                SECRET '{settings.aws_secret_access_key}',
                REGION '{settings.aws_region}'
            )
        """)
        self.bucket = settings.s3_bucket_name

    def query(self, sql: str):
        return self.conn.execute(sql).df()

    def read_parquet(self, s3_key_pattern: str):
        path = f"s3://{self.bucket}/{s3_key_pattern}"
        return self.query(f"SELECT * FROM read_parquet('{path}')")

    def close(self):
        self.conn.close()
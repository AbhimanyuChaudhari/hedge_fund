import os
import io
import boto3
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from config.settings import settings


class S3Client:
    def __init__(self):
        self.client = boto3.client(
            's3',
            aws_access_key_id     = settings.aws_access_key_id,
            aws_secret_access_key = settings.aws_secret_access_key,
            region_name           = settings.aws_region,
        )
        self.bucket = settings.s3_bucket_name  # ← renamed from bucket_name

    def list_files(self, prefix: str) -> list[str]:
        """List all files under a prefix. Handles pagination."""
        keys     = []
        paginator = self.client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                keys.append(obj['Key'])
        return keys

    def upload(self, local_path: str, s3_key: str):
        self.client.upload_file(local_path, self.bucket, s3_key)

    def download(self, s3_key: str, local_path: str):
        self.client.download_file(self.bucket, s3_key, local_path)

    def read_parquet(self, key: str) -> pd.DataFrame:
        resp = self.client.get_object(Bucket=self.bucket, Key=key)
        return pd.read_parquet(io.BytesIO(resp['Body'].read()))

    def write_parquet(self, df: pd.DataFrame, key: str):
        buf = io.BytesIO()
        df.to_parquet(buf, index=False, compression='zstd')
        buf.seek(0)
        self.client.put_object(
            Bucket = self.bucket,
            Key    = key,
            Body   = buf.getvalue(),
        )
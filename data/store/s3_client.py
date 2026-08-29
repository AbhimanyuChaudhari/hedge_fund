from config.settings import settings
import boto3

class S3Client:
    def __init__(self):
        self.client = boto3.client(
            's3',
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region
        )
        self.bucket = settings.s3_bucket_name

    def upload(self, local_path:str, s3_key:str):
        self.client.upload_file(local_path, self.bucket, s3_key)

    def download(self, s3_key:str, local_path:str):
        self.client.download_file(self.bucket, s3_key, local_path)

    def list_files(self, prefix:str):
        response = self.client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        return [obj['Key'] for obj in response.get('Contents', [])]
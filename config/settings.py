import os
import boto3 as _boto3
from dotenv import load_dotenv
load_dotenv()

def _get_s3_client():
    return _boto3.client('s3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION', 'ap-south-1')
    )

class Settings:
    def __init__(self):
        self.env = os.getenv('ENV', 'development')
        self.zerodha_access_token = os.getenv('ZERODHA_ACCESS_TOKEN')
        if self.zerodha_access_token is None:
            raise ValueError('ZERODHA_ACCESS_TOKEN is missing from .env')
        self.zerodha_api_key = os.getenv('ZERODHA_API_KEY')
        if self.zerodha_api_key is None:
            raise ValueError('Zerodha_api_key is missing from .env')
        self.zerodha_api_secret = os.getenv('ZERODHA_API_SECRET')
        if self.zerodha_api_secret is None:
            raise ValueError('Zerodha_api_secret is missing from .env')
        self.aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        if self.aws_access_key_id is None:
            raise ValueError('Aws_access_key is missing from .env')
        self.aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        if self.aws_secret_access_key is None:
            raise ValueError('Aws_secret_access_key is missing from .env')
        self.aws_region = os.getenv('AWS_REGION')
        if self.aws_region is None:
            raise ValueError('Aws_region is missing from .env')
        self.s3_bucket_name = os.getenv('S3_BUCKET_NAME')
        if self.s3_bucket_name is None:
            raise ValueError('S3_bucket name is missing from .env')
        self.zerodha_client_id = os.getenv('ZERODHA_CLIENT_ID')
        if self.zerodha_client_id is None:
            raise ValueError('ZERODHA_CLIENT_ID is missing from .env')
        self.zerodha_totp_secret = os.getenv('ZERODHA_TOTP_SECRET')
        if self.zerodha_totp_secret is None:
            raise ValueError('ZERODHA_TOTP_SECRET is missing from .env')
        self.zerodha_password = os.getenv('ZERODHA_PASSWORD')
        if self.zerodha_password is None:
            raise ValueError('ZERODHA_PASSWORD is missing from .env')
        self.redis_host = os.getenv('REDIS_HOST', 'localhost')
        self.redis_port = int(os.getenv('REDIS_PORT', '6379'))

settings = Settings()
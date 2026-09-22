import sys
sys.path.insert(0, '.')
from data.store.s3_client import S3Client

s = S3Client()
files = s.list_files('processed/1s/')
sep22 = [f for f in files if '2026-09-22' in f]
print(f'Processed Sep 22 files ({len(sep22)}):')
for f in sep22:
    print(f'  {f}')

import os
from typing import List

import boto3

from .logging import log

AWS_REGION = os.getenv("AWS_REGION")
AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")


class S3Manager:
    @staticmethod
    def session():
        return boto3.Session()

    @staticmethod
    def get_client():
        return boto3.client(
            "s3",
            aws_access_key_id=AWS_KEY,
            aws_secret_access_key=AWS_SECRET,
            region_name=AWS_REGION,
        )

    @log()
    @staticmethod
    def download_folder(boto3_client, bucket, prefix, local):
        objs = boto3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        for obj in objs.get("Contents", []):
            path = obj["Key"]
            rel_path = os.path.relpath(path, prefix)
            dest = os.path.join(local, rel_path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            boto3_client.download_file(bucket, path, dest)

    @log()
    @staticmethod
    def upload_bulk(
        boto3_client,
        local_path: str,
        files_to_upload: List[str],
        bucket_name: str,
    ):
        for file in files_to_upload:
            boto3_client.upload_file(
                f"{local_path}/{file}",
                bucket_name,
                file,
            )

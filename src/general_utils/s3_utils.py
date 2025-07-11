import logging
import os

import boto3

AWS_REGION = os.getenv("AWS_REGION")
AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")


class SaveLoadS3Client:
    @staticmethod
    def session():
        return boto3.Session()

    @staticmethod
    def get_client(service_name: str):
        return boto3.client(
            "s3",
            aws_access_key_id=AWS_KEY,
            aws_secret_access_key=AWS_SECRET,
            region_name=AWS_REGION,
        )

    @staticmethod
    def download_folder(boto3_client, bucket, prefix, local):
        objs = boto3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        for obj in objs.get("Contents", []):
            path = obj["Key"]
            rel_path = os.path.relpath(path, prefix)
            dest = os.path.join(local, rel_path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            boto3_client.download_file(bucket, path, dest)
        logging.info("Successful model download")

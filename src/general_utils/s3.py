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
    def download_files(
        boto3_client,
        local_path: str,
        files_to_download: List[str],
        bucket_name: str,
    )  -> None:
        """
        Downloads multiple files from an S3 bucket to a local directory.

        Args:
            boto3_client: An instantiated boto3 S3 client used to perform the download.
            local_path (str): Local directory path where the files will be saved.
            files_to_download (List[str]): List of S3 object keys (paths) to download.
            bucket_name (str): Name of the S3 bucket containing the files.
    
        Returns:
            None
        """
        for file in files_to_download:
            boto3_client.download_file(bucket_name, file, f"{local_path}/{file}")

    @log()
    @staticmethod
    def download_folder(boto3_client, bucket:str, prefix:str, local:str)  -> None:
        """
        Downloads all objects from a given S3 prefix to a local directory,
        preserving the relative folder structure.
    
        Args:
            boto3_client: An instantiated boto3 S3 client used to perform operations.
            bucket (str): Name of the S3 bucket.
            prefix (str): Prefix (path) in the bucket representing the "folder" to download.
            local (str): Local directory path where the downloaded files will be saved.
    
        Returns:
            None
        """
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
        bucket_key: str = "",
    ) -> None:
        """
        Uploads multiple files from a local directory to an S3 bucket.
    
        Args:
            boto3_client: An instantiated boto3 S3 client used to perform the upload.
            local_path (str): Path to the local directory containing the files to upload.
            files_to_upload (List[str]): List of filenames (relative to local_path) to upload.
            bucket_name (str): Name of the target S3 bucket.
            bucket_key (str, optional): S3 prefix (folder path) to prepend to uploaded file keys. Defaults to "".
    
        Returns:
            None
        """
        for file in files_to_upload:
            boto3_client.upload_file(
                f"{local_path}/{file}",
                bucket_name,
                f"{local_path}{file}",
            )

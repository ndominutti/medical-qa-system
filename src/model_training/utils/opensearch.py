import logging
import os
from typing import List

import boto3
import numpy as np
from opensearchpy import AWSV4SignerAuth, OpenSearch, RequestsHttpConnection, exceptions
from opensearchpy.helpers import bulk
from tqdm import tqdm

AWS_REGION = os.getenv("AWS_REGION")
MAPPING = {
    "settings": {"index.knn": True},
    "mappings": {
        "properties": {
            "chunk": {"type": "text", "analyzer": "standard"},
            "vector_field": {
                "type": "knn_vector",
                "dimension": 384,
                "method": {"name": "hnsw", "space_type": "l2", "engine": "faiss"},
            },
        }
    },
}


class OpenSearchManager:
    """
    Manages connections and operations for an OpenSearch cluster.
    """

    def __init__(self, host: str):
        service = "es"
        credentials = boto3.Session().get_credentials()
        auth = AWSV4SignerAuth(credentials, AWS_REGION, service)
        self.client = OpenSearch(
            hosts=[{"host": host, "port": 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            pool_maxsize=20,
            vector_field="vector_field",
        )

    def create_index(self, index_name: str) -> None:
        """
        Creates an OpenSearch index with the predefined mapping.
        If the index already exists, logs a warning and skips creation.

        Args:
            index_name (str): Name of the index to create.

        Returns:
            dict or None: The response from the OpenSearch client if created,
                          or None if the index already exists.
        """
        try:
            return self.client.indices.create(index=index_name, body=MAPPING)
        except exceptions.RequestError as e:
            if "resource_already_exists_exception" in str(e.info):
                logging.warning("Index already exists. Skipping creation.")
            else:
                raise

    def bulk_ingestion(
        self,
        index_name: str,
        corpus_text: List[str],
        corpus_embeddings: np.ndarray,
        request_timeout: int = 240,
    ) -> None:
        """
        Performs bulk ingestion of text chunks and their vector embeddings into an OpenSearch index.

        Each text-embedding pair is indexed as a document with a 'chunk' field and a 'vector_field'
        (or the name specified in the index mapping).

        Args:
            index_name (str): Name of the OpenSearch index where documents will be ingested.
            corpus_text (List[str]): List of text chunks to be indexed.
            corpus_embeddings (List[List[float]]): Corresponding list of embedding vectors.
            request_timeout (int, optional): Timeout (in seconds) for the bulk request. Defaults to 240.

        Returns:
            None
        """
        actions = [
            {"_index": index_name, "_source": {"chunk": text, "vector_field": emb}}
            for text, emb in tqdm(
                zip(corpus_text, corpus_embeddings.tolist(), strict=True)
            )
        ]
        success, _ = bulk(self.client, actions, request_timeout=request_timeout)
        logging.info(f"Successfully indexed {success} documents")

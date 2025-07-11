from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth, exceptions
from opensearchpy.helpers import bulk
from tqdm import tqdm
import boto3
import os
import logging

AWS_REGION = os.getenv('AWS_REGION')
MAPPING = {
    "settings": {
    "index.knn": True
  },
    "mappings": {
        "properties": {
            "chunk": {
                "type": "text",
                "analyzer": "standard"
            },
            "vector_field": {
                "type": "knn_vector",
                "dimension": 384,
                "method": {
                  "name": "hnsw",
                  "space_type": "l2",
                  "engine": "faiss"
                }
        }
    }
    }
}


class OpenSearchManager:

    def __init__(self, host:str):
        service = 'es'
        credentials = boto3.Session().get_credentials()
        auth = AWSV4SignerAuth(credentials, AWS_REGION, service)
        self.client = OpenSearch(
                        hosts=[{'host': host, 'port': 443}],
                        http_auth=auth,
                        use_ssl=True,
                        verify_certs=True,
                        connection_class=RequestsHttpConnection,
                        pool_maxsize=20,
                        vector_field='vector_field'
                    )

    def create_index(self, index_name:str):
        try:
            return self.client.indices.create(
                index=index_name,
                body=MAPPING
            )
        except exceptions.RequestError as e:
            if 'resource_already_exists_exception' in str(e.info):
                logging.warning("Index already exists. Skipping creation.")
            else:
                raise


    def bulk_ingestion(self, index_name:str, corpus_text: List[str], corpus_embeddings: List[List[float]], request_timeout:int=240):
        actions = [
            {
                "_index": index_name,
                "_source": {
                    "chunk": text,
                    "vector_field": emb
                }
            }
            for text, emb in tqdm(zip(corpus_text, corpus_embeddings.tolist()))
        ]
        success, _ = bulk(self.client, actions, request_timeout=request_timeout)
        logging.info(f"Successfully indexed {success} documents")
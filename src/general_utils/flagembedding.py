import os

import requests
from FlagEmbedding import FlagModel
from requests_aws4auth import AWS4Auth

from .logging import log

AWS_REGION = os.getenv("AWS_REGION")
AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")


class FlagEmbeddingManager:
    """
    Manager class for handling embedding generation and vector search using FlagEmbedding models
    with AWS OpenSearch.
    """

    def __init__(self, service: str = "es"):
        self.awsauth = AWS4Auth(AWS_KEY, AWS_SECRET, AWS_REGION, "es")

    @log()
    def get_model(
        self, local_model_path: str, query_instruction, use_fp16: bool = False
    ):
        """
        Loads and returns a FlagModel with the specified configuration.

        Args:
            local_model_path (str): Local path to the FlagEmbedding model files.
            query_instruction: Instruction string to guide retrieval-focused embeddings.
            use_fp16 (bool): Whether to use half-precision floats for faster/lighter inference. Defaults to False.

        Returns:
            FlagModel: The loaded embedding model.
        """
        return FlagModel(
            local_model_path,
            query_instruction_for_retrieval=query_instruction,
            use_fp16=use_fp16,
        )

    @staticmethod
    def _embed_query(flag_embedding_model: FlagModel, query: str):
        return flag_embedding_model.encode(query).tolist()

    @log()
    def search(
        self,
        endpoint_url: str,
        flag_embedding_model: FlagModel,
        query: str,
        top_k: int = 3,
    ):
        """
        Performs a k-NN search on an OpenSearch endpoint using the query's embedding.

        Args:
            endpoint_url (str): The full OpenSearch endpoint URL.
            flag_embedding_model (FlagModel): The loaded FlagEmbedding model for encoding.
            query (str): The query string to search for.
            top_k (int): Number of top results to retrieve. Defaults to 3.

        Returns:
            list: The top retrieved text chunks from the index.
        """
        query_vector = self._embed_query(flag_embedding_model, query)
        payload = {
            "query": {"knn": {"vector_field": {"vector": query_vector, "k": top_k}}}
        }
        response = requests.post(
            url=f"{endpoint_url}/_search",
            json=payload,
            auth=self.awsauth,
            headers={"Content-Type": "application/json"},
        ).json()
        return [hit["_source"]["chunk"] for hit in response["hits"]["hits"][:3]]

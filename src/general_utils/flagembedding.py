import os

import requests
from FlagEmbedding import FlagModel
from requests_aws4auth import AWS4Auth

from .logging import log

AWS_REGION = os.getenv("AWS_REGION")
AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")


class FlagEmbeddingManager:
    def __init__(self, service: str = "es"):
        self.awsauth = AWS4Auth(AWS_KEY, AWS_SECRET, AWS_REGION, "es")

    @log()
    def get_model(
        self, local_model_path: str, query_instruction, use_fp16: bool = False
    ):
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

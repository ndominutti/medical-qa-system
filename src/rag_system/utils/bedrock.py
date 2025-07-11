import json
import os
from typing import List

import boto3

from general_utils import log

AWS_REGION = os.getenv("AWS_REGION")
AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")


class BedrockManager:
    def __init__(self):
        self.client = boto3.client(
            "bedrock-runtime",
            aws_access_key_id=AWS_KEY,
            aws_secret_access_key=AWS_SECRET,
            region_name=AWS_REGION,
        )

    def _process_context(self, context: List[str]) -> str:
        return "\n * ".join(context)

    def _format_prompt(self, query: str, system_prompt: str, context: str):
        return f"""
            <|begin_of_text|>
            <|start_header_id|>system<|end_header_id|>
            {system_prompt.format(CONTEXT=context)}
            <|eot_id|>
            <|start_header_id|>user<|end_header_id|>
            {query}
            <|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>
        """

    @log()
    def ask(
        self,
        query: str,
        context: List[str],
        model_id: str,
        system_prompt: str,
        temperature: float,
        top_p: float,
        max_output_tokens: int,
    ):
        context_str = self._process_context(context)
        prompt = self._format_prompt(query, system_prompt, context_str)
        payload = json.dumps(
            {
                "prompt": prompt,
                "max_gen_len": max_output_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
        response = self.client.invoke_model(modelId=model_id, body=payload)
        result = response["body"].read().decode("utf-8")
        return json.loads(result)

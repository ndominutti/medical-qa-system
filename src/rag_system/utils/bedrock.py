import json
import os
from typing import List

import boto3

from general_utils import log

AWS_REGION = os.getenv("AWS_REGION")
AWS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY")


class BedrockManager:
    """
    Manages interactions with AWS Bedrock runtime to invoke language models.
    """
    def __init__(self):
        self.client = boto3.client(
            "bedrock-runtime",
            aws_access_key_id=AWS_KEY,
            aws_secret_access_key=AWS_SECRET,
            region_name=AWS_REGION,
        )

    def _process_context(self, context: List[str]) -> str:
        return "\n * ".join(context)

    def _format_prompt(self, query: str, system_prompt: str, context: str) -> str:
        """
        Formats the final prompt string combining the system prompt, context, and user query using Bedrock Llama-specific tokens.

        Args:
            query (str): The user's input query.
            system_prompt (str): The system prompt template that accepts a CONTEXT placeholder.
            context (str): The formatted context string.

        Returns:
            str: The fully formatted prompt to be sent to the model.
        """
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
        model_id: str,
        system_prompt: str,
        temperature: float,
        top_p: float,
        max_output_tokens: int,
        context: List[str] = None,
    ) -> dict:
        """
        Sends a prompt to a Bedrock model and returns the decoded JSON response.

        Args:
            query (str): The user's input query.
            model_id (str): The Bedrock model identifier to invoke.
            system_prompt (str): The system prompt template string with a CONTEXT placeholder.
            temperature (float): Sampling temperature controlling randomness.
            top_p (float): Nucleus sampling probability threshold.
            max_output_tokens (int): Maximum number of tokens to generate.
            context (List[str], optional): Optional list of context strings to include in the prompt.

        Returns:
            dict: The JSON-parsed response from the Bedrock model.
        """
        context_str = self._process_context(context) if context else ""
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

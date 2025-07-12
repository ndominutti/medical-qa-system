# medical-qa-system

This repository contains a take-home assignment where I was given a medical related Q&A dataset and tasked with creating a system that allows a user to ask questions and receive answers.

## Definitions / Key points

- I strongly believe that for a medical-domain QA system intended for potential patients, *mitigating hallucinations is critical*. Therefore, I decided to implement a *Retrieval-Augmented Generation (RAG)* system and prompted the LLM to avoid answering if the relevant data was not found in the retrieved context.

- While reducing hallucinations was the primary focus, due to time constraints, additional measures such as setting up *guardrails* or using judges to verify *faithfulness/groundedness* of generated answers against the retrieved context were not implemented. In production, balancing cost and latency tradeoffs of adding these components is essential.
- To better leverage the domain-specific data, I *fine-tuned the embedding model* (`bge-base-en-v1.5`) for domain adaptation aligned with the business problem. The assumption is that improving retrieval quality will lead to more accurate and confident answers, ultimately boosting user engagement and safety.
- To accelerate development and use production-ready tools, I chose *AWS OpenSearch* as the vector database instead of persisting the index in a FAISS file. Creating a dedicated microservice for an open-source vector database like Qdrant was not feasible within the time constraints.
- Since my local PC lacks a GPU, I performed model training on a *GPU-enabled AWS SageMaker JupyterLab instance*.
- Although this take-home solution relies heavily on notebooks, this approach is not recommended for production. Ideally, all components should be decoupled and run as robust, battle-tested services. For example, preprocessing could be implemented using SageMaker Pipelines, training as SageMaker Training Jobs, and the RAG system deployed as a dedicated microservice or leveraging cloud provider managed services such as AWS Bedrock or Databricks Mosaic AI endpoints.
- Utilizing external APIs with medical data is extremely risky, as it can lead to potential data leaks that put patient information at risk. One reasonable alternative would be to run a local LLM instead; however, that was not feasible for this take-home assignment. I chose to use AWS Bedrock, which offers significantly more security measures compared to calling a generic provider’s API.
- To reduce time and costs for this example, I embedded and indexed only the TEST queries into AWS OpenSearch. Normally, the entire dataset should be indexed to enable retrieval at inference time.



## Source tree

- `/src/data_processing` – Package responsible for preprocessing the received data and preparing it for embedding training.
- `/src/general_utils` – Utility functions used across all other processes.
- `/src/model_training` – Package for model training using the FlagEmbedding library, as recommended by the BGE model developers.
- `/src/rag_system` – Package implementing vector retrieval and AWS Bedrock querying to simulate a production-style RAG service (though simplified for this take-home).


## Project walkthrough

Each of the following notebooks includes a complete walkthrough of the process for its respective component—from the initial steps through to metrics comparisons and evaluations:

- `/src/dataprocessing_pipeline.ipynb`
- `/src/model_train.ipynb`
- `/src/rag_demonstration.ipynb`

Note that `/src/dataprocessing_pipeline.ipynb` and `/src/rag_demonstration.ipynb` were run locally, while `/src/model_train.ipynb` was executed on a GPU-enabled cloud machine.



## Further thoughts

- Although this take-home is technically focused, it is always essential to connect technical results to business KPIs. For example, if we increase NDCG@3 in our retrieval model, what does that mean in business terms? How does it impact outcomes?
- The RAG implementation lacks a critical component for a system like this: *memory*. As a result, this implementation should be considered more of a single-turn QA bot rather than a full conversational assistant.
- For the generation component, I chose the `llama3-70b-instruct` model as a strong baseline. Nevertheless, the model could benefit from domain-specific pretraining, and one could experiment with models already trained on medical data. However, doing so would typically require self-hosting the model, rather than using a pay-per-token service like the `llama3-70b-instruct` model accessed via AWS Bedrock for this assignment.

from typing import List, Tuple

import faiss
import numpy as np
from datasets import Dataset
from tqdm import tqdm

from general_utils import log


class Validator:
    @staticmethod
    def _encode(
        model, queries_text: str, corpus_text: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        return (model.encode_queries(queries_text), model.encode_corpus(corpus_text))

    @staticmethod
    def _index_corpus(corpus_embeddings):
        dim = corpus_embeddings.shape[-1]
        index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
        corpus_embeddings = corpus_embeddings.astype(np.float32)
        index.train(corpus_embeddings)
        index.add(corpus_embeddings)
        return index

    @staticmethod
    def _search(queries_embeddings, index, k, n_batches):
        query_size = len(queries_embeddings)
        all_scores = []
        all_indices = []
        for i in tqdm(range(0, query_size, n_batches)):
            j = min(i + 32, query_size)
            query_embedding = queries_embeddings[i:j]
            score, indice = index.search(query_embedding.astype(np.float32), k=k)
            all_scores.append(score)
            all_indices.append(indice)
        all_scores = np.concatenate(all_scores, axis=0)
        all_indices = np.concatenate(all_indices, axis=0)
        return (all_scores, all_indices)

    @staticmethod
    def _get_results(all_scores, all_indices, queries: Dataset):
        results = {}
        for idx, (scores, indices) in enumerate(
            zip(all_scores, all_indices, strict=False, strics=True)
        ):
            results[str(queries["id"][idx])] = {}
            for score, index in zip(scores, indices, strict=False):
                if index != -1:
                    results[str(queries["id"][idx])][str(index)] = float(score)
        return results

    @log()
    @staticmethod
    def search(
        model,
        queries_text: str,
        corpus_text: List[str],
        queries: Dataset,
        k: int = 20,
        n_batches: int = 32,
    ):
        """ """
        queries_embeddings, corpus_embeddings = Validator._encode(
            model, queries_text, corpus_text
        )
        index = Validator._index_corpus(corpus_embeddings)
        all_scores, all_indices = Validator._search(
            queries_embeddings, index, k, n_batches
        )
        results = Validator._get_results(all_scores, all_indices, queries)
        return results, corpus_embeddings

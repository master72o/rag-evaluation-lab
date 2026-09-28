"""
Decoupled Retrieval Quality Evaluator: Precision@K, Recall@K, MRR, NDCG.
"""

import numpy as np
from typing import List
from rag_eval.schema import ContextChunk, RetrievalMetricsResult


class RetrievalEvaluator:
    """Evaluates retrieval quality metrics against ground-truth relevant chunk IDs."""

    @staticmethod
    def evaluate_retrieval(
        retrieved_chunks: List[ContextChunk], ground_truth_ids: List[str]
    ) -> RetrievalMetricsResult:
        if not retrieved_chunks or not ground_truth_ids:
            return RetrievalMetricsResult(precision_at_k=0.0, recall_at_k=0.0, mrr=0.0, ndcg=0.0)

        k = len(retrieved_chunks)
        gt_set = set(ground_truth_ids)

        relevant_flags = [1 if chunk.chunk_id in gt_set else 0 for chunk in retrieved_chunks]
        relevant_count = sum(relevant_flags)

        # 1. Precision@K
        precision_at_k = relevant_count / k

        # 2. Recall@K
        recall_at_k = relevant_count / len(gt_set) if len(gt_set) > 0 else 0.0

        # 3. MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for idx, flag in enumerate(relevant_flags, 1):
            if flag == 1:
                mrr = 1.0 / idx
                break

        # 4. NDCG (Normalized Discounted Cumulative Gain)
        dcg = 0.0
        for idx, flag in enumerate(relevant_flags, 1):
            if flag == 1:
                dcg += 1.0 / np.log2(idx + 1)

        idcg = 0.0
        max_rel = min(k, len(gt_set))
        for idx in range(1, max_rel + 1):
            idcg += 1.0 / np.log2(idx + 1)

        ndcg = dcg / idcg if idcg > 0 else 0.0

        return RetrievalMetricsResult(
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            mrr=mrr,
            ndcg=ndcg,
        )

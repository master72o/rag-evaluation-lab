"""
Tests for Retrieval Evaluator.
"""

from rag_eval.schema import RAGChunk
from rag_eval.retrieval_evaluator import RetrievalEvaluator


def test_evaluate_retrieval_perfect():
    retrieved = [
        RAGChunk(chunk_id="c1", text="text1", score=0.9),
        RAGChunk(chunk_id="c2", text="text2", score=0.8)
    ]
    ground_truth = ["c1", "c2"]

    metrics = RetrievalEvaluator.evaluate_retrieval(retrieved, ground_truth)
    assert metrics.precision_at_k == 1.0
    assert metrics.recall_at_k == 1.0
    assert metrics.mrr == 1.0
    assert metrics.ndcg == 1.0


def test_evaluate_retrieval_partial():
    retrieved = [
        RAGChunk(chunk_id="c99", text="wrong", score=0.9),
        RAGChunk(chunk_id="c1", text="text1", score=0.8)
    ]
    ground_truth = ["c1"]

    metrics = RetrievalEvaluator.evaluate_retrieval(retrieved, ground_truth)
    assert metrics.precision_at_k == 0.5
    assert metrics.recall_at_k == 1.0
    assert metrics.mrr == 0.5
    assert metrics.ndcg < 1.0

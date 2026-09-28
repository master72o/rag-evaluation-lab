"""
Tests for Generation Evaluator.
"""

from rag_eval.schema import RAGChunk
from rag_eval.generation_evaluator import GenerationEvaluator


def test_evaluate_generation_accurate():
    chunks = [RAGChunk(chunk_id="c1", text="Acme revenues were $4.2B in Q3 2025.", score=0.9)]
    prompt = "What were Acme revenues in Q3 2025?"
    response = "Acme revenues were $4.2B in Q3 2025."
    reference = "Acme revenues were $4.2B in Q3 2025."
    citations = ["c1"]

    metrics = GenerationEvaluator.evaluate_generation(prompt, response, reference, chunks, citations)
    assert metrics.faithfulness > 0.8
    assert metrics.answer_relevance > 0.8
    assert metrics.citation_correctness == 1.0


def test_evaluate_generation_hallucinated():
    chunks = [RAGChunk(chunk_id="c1", text="Acme revenues were $4.2B in Q3 2025.", score=0.9)]
    prompt = "What were Acme revenues?"
    response = "Acme acquired WidgetCorp for $10B."  # Hallucinated
    reference = "Acme revenues were $4.2B."
    citations = ["c99"]

    metrics = GenerationEvaluator.evaluate_generation(prompt, response, reference, chunks, citations)
    assert metrics.faithfulness < 0.5
    assert metrics.citation_correctness == 0.0

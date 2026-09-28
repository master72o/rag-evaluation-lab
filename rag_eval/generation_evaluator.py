"""
Decoupled Generation Quality Evaluator: Faithfulness, Answer Relevance, Context Precision, Citation Correctness.
"""

import re
from typing import List
from rag_eval.schema import ContextChunk, GenerationMetricsResult


class GenerationEvaluator:
    """Evaluates generation metrics against retrieved context and reference answer."""

    @staticmethod
    def evaluate_generation(
        prompt: str,
        response: str,
        reference_answer: str,
        retrieved_chunks: List[ContextChunk],
        citations: List[str]
    ) -> GenerationMetricsResult:
        context_text = " ".join([c.text for c in retrieved_chunks]).lower()
        response_lower = response.lower()
        prompt_lower = prompt.lower()
        ref_lower = reference_answer.lower()

        # 1. Faithfulness (Groundedness in context)
        resp_words = set(re.findall(r"\w+", response_lower)) - {"the", "a", "an", "is", "in", "of", "and", "to", "for"}
        ctx_words = set(re.findall(r"\w+", context_text)) - {"the", "a", "an", "is", "in", "of", "and", "to", "for"}
        
        if not resp_words:
            faithfulness = 1.0
        else:
            intersection = resp_words.intersection(ctx_words)
            faithfulness = len(intersection) / len(resp_words)

        # 2. Answer Relevance
        prompt_words = set(re.findall(r"\w+", prompt_lower)) - {"the", "a", "an", "what", "how", "why", "is"}
        rel_intersection = resp_words.intersection(prompt_words)
        answer_relevance = len(rel_intersection) / max(len(prompt_words), 1)
        answer_relevance = min(1.0, answer_relevance + 0.5)  # Rescale

        # 3. Context Precision & Recall
        rel_chunks = [c for c in retrieved_chunks if c.is_relevant]
        context_precision = len(rel_chunks) / len(retrieved_chunks) if retrieved_chunks else 0.0
        context_recall = 1.0 if len(rel_chunks) > 0 else 0.0

        # 4. Citation Correctness
        if not citations:
            citation_correctness = 1.0
        else:
            valid_citations = [c_id for c_id in citations if any(c.chunk_id == c_id for c in retrieved_chunks)]
            citation_correctness = len(valid_citations) / len(citations)

        return GenerationMetricsResult(
            faithfulness=round(min(1.0, faithfulness + 0.2), 4),
            answer_relevance=round(min(1.0, answer_relevance), 4),
            context_precision=round(context_precision, 4),
            context_recall=round(context_recall, 4),
            citation_correctness=round(citation_correctness, 4),
        )

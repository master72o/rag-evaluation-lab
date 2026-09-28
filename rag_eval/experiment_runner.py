"""
Experiment Runner executing RAG Pipeline Config Comparisons (v1 Baseline, v2 Hybrid, v3 Reranked).
"""

from typing import List, Dict
from rag_eval.schema import RAGBenchmarkItem, RAGItemEvaluation, RAGConfigType
from rag_eval.retrieval_evaluator import RetrievalEvaluator
from rag_eval.generation_evaluator import GenerationEvaluator


class ExperimentRunner:
    """Runs decoupled RAG evaluations across RAG pipeline variants."""

    @staticmethod
    def evaluate_benchmark_item(item: RAGBenchmarkItem) -> Dict[RAGConfigType, RAGItemEvaluation]:
        evaluations = {}

        # 1. RAG v1 Baseline
        ret_v1 = RetrievalEvaluator.evaluate_retrieval(item.retrieved_chunks_v1, item.ground_truth_chunk_ids)
        gen_v1 = GenerationEvaluator.evaluate_generation(
            item.prompt, item.response_v1, item.reference_answer, item.retrieved_chunks_v1, item.citations_v1
        )
        score_v1 = (ret_v1.ndcg * 0.4) + (gen_v1.faithfulness * 0.3) + (gen_v1.citation_correctness * 0.3)
        evaluations[RAGConfigType.V1_BASELINE] = RAGItemEvaluation(
            item_id=item.id,
            config_name=RAGConfigType.V1_BASELINE,
            retrieval=ret_v1,
            generation=gen_v1,
            overall_score=round(score_v1, 4),
            passed=score_v1 >= 0.70
        )

        # 2. RAG v2 Hybrid
        ret_v2 = RetrievalEvaluator.evaluate_retrieval(item.retrieved_chunks_v2, item.ground_truth_chunk_ids)
        gen_v2 = GenerationEvaluator.evaluate_generation(
            item.prompt, item.response_v2, item.reference_answer, item.retrieved_chunks_v2, item.citations_v2
        )
        score_v2 = (ret_v2.ndcg * 0.4) + (gen_v2.faithfulness * 0.3) + (gen_v2.citation_correctness * 0.3)
        evaluations[RAGConfigType.V2_HYBRID] = RAGItemEvaluation(
            item_id=item.id,
            config_name=RAGConfigType.V2_HYBRID,
            retrieval=ret_v2,
            generation=gen_v2,
            overall_score=round(score_v2, 4),
            passed=score_v2 >= 0.70
        )

        # 3. RAG v3 Reranked
        ret_v3 = RetrievalEvaluator.evaluate_retrieval(item.retrieved_chunks_v3, item.ground_truth_chunk_ids)
        gen_v3 = GenerationEvaluator.evaluate_generation(
            item.prompt, item.response_v3, item.reference_answer, item.retrieved_chunks_v3, item.citations_v3
        )
        score_v3 = (ret_v3.ndcg * 0.4) + (gen_v3.faithfulness * 0.3) + (gen_v3.citation_correctness * 0.3)
        evaluations[RAGConfigType.V3_RERANKED] = RAGItemEvaluation(
            item_id=item.id,
            config_name=RAGConfigType.V3_RERANKED,
            retrieval=ret_v3,
            generation=gen_v3,
            overall_score=round(score_v3, 4),
            passed=score_v3 >= 0.70
        )

        return evaluations

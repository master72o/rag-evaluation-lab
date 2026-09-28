"""
Visualization Generator for RAG Evaluation Lab.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any


class Visualizer:
    """Generates comparison charts for RAG pipeline experiments."""

    @staticmethod
    def generate_all_figures(suite_metrics: Dict[str, Any], output_dir: str = "reports/figures") -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        generated = {}

        # 1. Retrieval Metrics Comparison
        ret_path = os.path.join(output_dir, "retrieval_metrics_comparison.png")
        Visualizer._plot_retrieval_comparison(suite_metrics, ret_path)
        generated["retrieval_metrics_comparison"] = ret_path

        # 2. Generation Metrics Comparison
        gen_path = os.path.join(output_dir, "generation_metrics_comparison.png")
        Visualizer._plot_generation_comparison(suite_metrics, gen_path)
        generated["generation_metrics_comparison"] = gen_path

        # 3. Overall Performance Comparison
        ov_path = os.path.join(output_dir, "overall_rag_performance.png")
        Visualizer._plot_overall_performance(suite_metrics, ov_path)
        generated["overall_rag_performance"] = ov_path

        return generated

    @staticmethod
    def _plot_retrieval_comparison(metrics: Dict[str, Any], output_path: str):
        fig, ax = plt.subplots(figsize=(10, 5))
        configs = ["RAG_v1_Baseline", "RAG_v2_Hybrid", "RAG_v3_Reranked"]
        labels = ["RAG v1\n(Baseline)", "RAG v2\n(Hybrid)", "RAG v3\n(Reranked)"]

        precisions = [metrics.get(c, {}).get("retrieval", {}).get("mean_precision_at_k", 0.0) for c in configs]
        recalls = [metrics.get(c, {}).get("retrieval", {}).get("mean_recall_at_k", 0.0) for c in configs]
        mrrs = [metrics.get(c, {}).get("retrieval", {}).get("mean_mrr", 0.0) for c in configs]
        ndcgs = [metrics.get(c, {}).get("retrieval", {}).get("mean_ndcg", 0.0) for c in configs]

        x = np.arange(len(labels))
        width = 0.2

        rects1 = ax.bar(x - 1.5*width, precisions, width, label="Precision@K", color="#2b5c8f")
        rects2 = ax.bar(x - 0.5*width, recalls, width, label="Recall@K", color="#5bc0de")
        rects3 = ax.bar(x + 0.5*width, mrrs, width, label="MRR", color="#f0ad4e")
        rects4 = ax.bar(x + 1.5*width, ndcgs, width, label="NDCG", color="#5cb85c")

        ax.set_ylabel("Metric Score", fontsize=11, fontweight="bold")
        ax.set_title("Decoupled Retrieval Quality Metrics (v1 vs v2 vs v3)", fontsize=13, fontweight="bold", pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.0, 1.15)
        ax.legend(loc="upper left")
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        for rects in [rects1, rects2, rects3, rects4]:
            for r in rects:
                h = r.get_height()
                ax.text(r.get_x() + r.get_width()/2., h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

    @staticmethod
    def _plot_generation_comparison(metrics: Dict[str, Any], output_path: str):
        fig, ax = plt.subplots(figsize=(10, 5))
        configs = ["RAG_v1_Baseline", "RAG_v2_Hybrid", "RAG_v3_Reranked"]
        labels = ["RAG v1\n(Baseline)", "RAG v2\n(Hybrid)", "RAG v3\n(Reranked)"]

        faiths = [metrics.get(c, {}).get("generation", {}).get("mean_faithfulness", 0.0) for c in configs]
        c_precs = [metrics.get(c, {}).get("generation", {}).get("mean_context_precision", 0.0) for c in configs]
        c_corrs = [metrics.get(c, {}).get("generation", {}).get("mean_citation_correctness", 0.0) for c in configs]

        x = np.arange(len(labels))
        width = 0.25

        rects1 = ax.bar(x - width, faiths, width, label="Faithfulness", color="#5cb85c")
        rects2 = ax.bar(x, c_precs, width, label="Context Precision", color="#2b5c8f")
        rects3 = ax.bar(x + width, c_corrs, width, label="Citation Correctness", color="#5bc0de")

        ax.set_ylabel("Metric Score", fontsize=11, fontweight="bold")
        ax.set_title("Decoupled Generation & Faithfulness Metrics (v1 vs v2 vs v3)", fontsize=13, fontweight="bold", pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.0, 1.15)
        ax.legend(loc="upper left")
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        for rects in [rects1, rects2, rects3]:
            for r in rects:
                h = r.get_height()
                ax.text(r.get_x() + r.get_width()/2., h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

    @staticmethod
    def _plot_overall_performance(metrics: Dict[str, Any], output_path: str):
        fig, ax = plt.subplots(figsize=(8, 5))
        configs = ["RAG_v1_Baseline", "RAG_v2_Hybrid", "RAG_v3_Reranked"]
        labels = ["RAG v1 (Baseline)", "RAG v2 (Hybrid)", "RAG v3 (Reranked)"]
        scores = [metrics.get(c, {}).get("overall_mean_score", 0.0) for c in configs]
        colors = ["#d9534f", "#f0ad4e", "#5cb85c"]

        bars = ax.bar(labels, scores, color=colors, edgecolor="#333333", width=0.4)
        ax.set_ylabel("Overall Quality Score", fontsize=11, fontweight="bold")
        ax.set_ylim(0.0, 1.1)
        ax.set_title("RAG Pipeline Overall Score Progression", fontsize=13, fontweight="bold", pad=15)
        ax.axhline(y=0.70, color="#2b5c8f", linestyle="--", linewidth=1.2, label="Pass Gate Threshold (0.70)")
        ax.legend(loc="upper left")
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.02, f"{h:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()

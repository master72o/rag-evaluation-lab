"""
CLI Interface for RAG Evaluation Lab.
"""

import argparse
import sys
from rag_eval.schema import RAGDataset
from rag_eval.metrics import MetricsCalculator
from rag_eval.visualizer import Visualizer
from rag_eval.report_generator import ReportGenerator


def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation Lab CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: run
    run_parser = subparsers.add_parser("run", help="Run decoupled RAG evaluation on dataset")
    run_parser.add_argument("--dataset", required=True, help="Path to JSONL benchmark dataset")
    run_parser.add_argument("--output-dir", default="results", help="Directory to save CSV/JSON outputs")
    run_parser.add_argument("--report-path", default="reports/rag_evaluation_report.md", help="Path to save Markdown report")
    run_parser.add_argument("--figures-dir", default="reports/figures", help="Directory to save figure plots")

    args = parser.parse_args()

    if args.command == "run":
        print(f"Loading dataset from: {args.dataset}")
        dataset = RAGDataset.from_jsonl(args.dataset)
        print(f"Loaded {len(dataset.items)} benchmark items.")

        print("Evaluating RAG pipeline variants (v1 Baseline, v2 Hybrid, v3 Reranked)...")
        evaluations = dataset.evaluate_all()

        metrics = MetricsCalculator.calculate_suite_metrics(evaluations)
        print("\nSummary Metrics:")
        for cfg, m in metrics.items():
            print(f"  {cfg}: Overall Score={m['overall_mean_score']:.4f}, Pass Rate={m['pass_rate']*100:.1f}%")

        print(f"\nExporting results to: {args.output_dir}")
        ReportGenerator.export_results(evaluations, args.output_dir)

        print(f"Generating visualizations in: {args.figures_dir}")
        Visualizer.generate_all_figures(metrics, args.figures_dir)

        print(f"Generating research report at: {args.report_path}")
        ReportGenerator.generate_markdown_report(metrics, args.report_path)

        print("\nEvaluation pipeline completed successfully!")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

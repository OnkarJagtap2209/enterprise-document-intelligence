import argparse
import json
from pathlib import Path
from enterprise_rag.evaluation import discover_datasets
from enterprise_rag.evaluation import evaluate_one, persist_execution, summarize_results, load_result_rows, identify_baseline, build_comparison, write_comparison

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-datasets", action="store_true")
    parser.add_argument("--run-one", nargs=2, metavar=("DATASET", "QUESTION_ID"))
    parser.add_argument("--results-dir", default="data/evaluation/results")
    parser.add_argument("--inspect-result")
    parser.add_argument("--run-benchmark", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--phase10-pilot", action="store_true")
    parser.add_argument("--phase10-benchmark", action="store_true")
    parser.add_argument("--compare-phase10", action="store_true")
    args = parser.parse_args()
    datasets = discover_datasets(Path("data/evaluation/questions"))
    if args.compare_phase10:
        baseline = identify_baseline(load_result_rows(args.results_dir))
        report = build_comparison(baseline, load_result_rows(Path(args.results_dir) / "phase10"))
        print(write_comparison("data/evaluation/reports/phase10_baseline_vs_improved.json", report))
        return 0
    if args.inspect_result:
        payload = json.loads(Path(args.inspect_result).read_text(encoding="utf-8"))
        execution = payload.get("execution", {}); actual = payload.get("actual", {})
        print(f"Question: {payload.get('question')}\nStatus: {execution.get('status')}\nLatency: {execution.get('latency_ms')} ms\nSources: {len(actual.get('sources', []))}\nAnswer available: {bool(actual.get('answer'))}\nError: {execution.get('error')}")
        return 0
    if args.phase10_pilot or args.phase10_benchmark:
        from enterprise_rag.application import LazyRAGService
        selected = [(d, q) for d in datasets for q in d.questions]
        if args.phase10_pilot: selected = selected[:3]
        output_dir = Path(args.results_dir) / "phase10"
        existing = { (row.get("dataset_name"), row.get("question_id")) for row in load_result_rows(output_dir) }
        if args.phase10_benchmark:
            selected = [(d, q) for d, q in selected if (d.dataset_name, q.question_id) not in existing]
        service = LazyRAGService(); results = []
        for dataset, question in selected:
            result = evaluate_one(dataset, question, service)
            persist_execution(result, output_dir); results.append(result)
            print(f"{dataset.dataset_name}/{question.question_id}: {result.execution['status']}")
            if result.execution["status"] != "success" and "429" in str(result.execution.get("error", {})):
                print("Stopping Phase 10 run after Gemini quota failure."); break
        print(summarize_results(results)); return 0
    if args.run_benchmark or args.pilot:
        from enterprise_rag.application import LazyRAGService
        selected = [(d, q) for d in datasets for q in d.questions]
        if args.pilot: selected = selected[:5]
        service = LazyRAGService(); results = []
        for dataset, question in selected:
            result = evaluate_one(dataset, question, service)
            persist_execution(result, args.results_dir); results.append(result)
            print(f"{question.question_id}: {result.execution['status']}")
        print(summarize_results(results))
        return 0
    if not args.validate_datasets and not args.run_one:
        parser.error("provide --validate-datasets or --run-one DATASET QUESTION_ID")
    print("Evaluation datasets validated successfully.")
    print(f"Datasets: {len(datasets)}")
    print(f"Questions per dataset: {len(datasets[0].questions)}")
    print(f"Total questions: {sum(len(d.questions) for d in datasets)}")
    for dataset in datasets:
        print(f"- {dataset.source_document['filename']}")
    if args.run_one:
        dataset_name, question_id = args.run_one
        dataset = next((d for d in datasets if Path(dataset_name).name in {d.path.name, d.source_document["filename"]}), None)
        if dataset is None: parser.error(f"dataset not found: {dataset_name}")
        question = next((q for q in dataset.questions if q.question_id == question_id), None)
        if question is None: parser.error(f"question not found: {question_id}")
        from enterprise_rag.application import LazyRAGService
        result = evaluate_one(dataset, question, LazyRAGService())
        path = persist_execution(result, args.results_dir)
        print(f"Question ID: {question_id}\nStatus: {result.execution['status']}\nResult: {path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

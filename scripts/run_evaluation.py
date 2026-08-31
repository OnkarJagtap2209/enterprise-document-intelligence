import argparse
import json
from pathlib import Path
from enterprise_rag.evaluation import discover_datasets
from enterprise_rag.evaluation import execute_one, persist_execution

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-datasets", action="store_true")
    parser.add_argument("--run-one", nargs=2, metavar=("DATASET", "QUESTION_ID"))
    parser.add_argument("--results-dir", default="data/evaluation/results")
    parser.add_argument("--inspect-result")
    args = parser.parse_args()
    if args.inspect_result:
        payload = json.loads(Path(args.inspect_result).read_text(encoding="utf-8"))
        execution = payload.get("execution", {}); actual = payload.get("actual", {})
        print(f"Question: {payload.get('question')}\nStatus: {execution.get('status')}\nLatency: {execution.get('latency_ms')} ms\nSources: {len(actual.get('sources', []))}\nAnswer available: {bool(actual.get('answer'))}\nError: {execution.get('error')}")
        return 0
    if not args.validate_datasets and not args.run_one:
        parser.error("provide --validate-datasets or --run-one DATASET QUESTION_ID")
    datasets = discover_datasets(Path("data/evaluation/questions"))
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
        result = execute_one(dataset, question, LazyRAGService())
        path = persist_execution(result, args.results_dir)
        print(f"Question ID: {question_id}\nStatus: {result.execution['status']}\nResult: {path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

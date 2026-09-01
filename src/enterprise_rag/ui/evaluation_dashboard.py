"""Read-only, lightweight evaluation view for the Streamlit application."""
from __future__ import annotations

from html import escape
import json
from pathlib import Path
from typing import Any, Iterable

ACTIVE_DATASETS = {
    "Infosys Q1 FY2027 Evaluation Set",
    "Infosys Form 20-F FY2026 Evaluation Set",
}
EXPECTED_QUESTIONS = 24


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) and value.get("question_id") else None


def load_current_results(
    results_dir: str | Path,
    active_datasets: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Load newest direct result per active dataset/question; never read subdirectories."""
    allowed = active_datasets or ACTIVE_DATASETS
    paths = sorted(Path(results_dir).glob("*.json"))

    # There is no formal run manifest in the persisted files. Use the newest
    # result date as the current run boundary so older ad-hoc runs are not
    # presented as failures in the active evaluation.
    dated = [
        path.stem[:8]
        for path in paths
        if path.stem[:8].isdigit() and len(path.stem) >= 8
    ]
    current_date = max(dated) if dated else None

    newest: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}

    for path in paths:
        if current_date and not path.stem.startswith(current_date):
            continue

        row = _load_json(path)
        if not row or row.get("dataset_name") not in allowed:
            continue

        key = (str(row["dataset_name"]), str(row["question_id"]))
        stamp = path.stat().st_mtime_ns

        if key not in newest or stamp >= newest[key][0]:
            row["_path"] = str(path)
            newest[key] = (stamp, row)

    return [newest[key][1] for key in sorted(newest)]


def load_expected_questions(
    question_dir: str | Path = "data/evaluation/questions",
) -> list[dict[str, str]]:
    """Return the active 24-question population for displaying pending rows."""
    expected = []

    for path in sorted(Path(question_dir).glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue

        if payload.get("dataset_name") not in ACTIVE_DATASETS:
            continue

        expected.extend(
            {
                "dataset_name": payload["dataset_name"],
                "question_id": q.get("question_id"),
                "question": q.get("question", "—"),
                "source_document": payload.get("source_document", {}).get("filename"),
            }
            for q in payload.get("questions", [])
            if isinstance(q, dict) and q.get("question_id")
        )

    return expected


def _metric(row: dict[str, Any], section: str, key: str) -> float | None:
    value = row.get("actual", {}).get(section, {}).get(key)

    return (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
    )


def _status(row: dict[str, Any]) -> str:
    return str(row.get("execution", {}).get("status", "missing"))


def _source_name(row: dict[str, Any]) -> str | None:
    return (
        row.get("source_document")
        if isinstance(row.get("source_document"), str)
        else None
    )


def _canonical_source(value: str | None) -> str:
    if not value:
        return ""

    aliases = {
        "q1-26-2027(1).pdf": "q1-26-2027.pdf",
        "form20f-25-26.pdf": "form20f-25-2026.pdf",
    }

    return aliases.get(Path(value).name, Path(value).name)


def aggregate_results(
    rows: Iterable[dict[str, Any]],
    expected_questions: int = EXPECTED_QUESTIONS,
) -> dict[str, Any]:
    rows = list(rows)

    successful = [row for row in rows if _status(row) == "success"]

    scored = [
        row
        for row in successful
        if (row.get("trace") or {}).get("scoring_status")
        in {"scored", "partially_scorable"}
    ]

    retrieval_keys = (
        "hit_at_5",
        "precision_at_5",
        "recall_at_5",
        "reciprocal_rank",
    )

    retrieval_scored = [
        row
        for row in scored
        if all(
            _metric(row, "retrieval_evaluation", key) is not None
            for key in retrieval_keys
        )
    ]

    values = {}

    for label, section, key in (
        ("answer_correctness", "answer_evaluation", "correctness_score"),
        ("groundedness", "answer_evaluation", "groundedness_score"),
    ):
        found = [_metric(row, section, key) for row in scored]
        found = [value for value in found if value is not None]
        values[label] = sum(found) / len(found) if found else None

    retrieval = {}

    for key in retrieval_keys:
        found = [
            _metric(row, "retrieval_evaluation", key)
            for row in retrieval_scored
        ]
        retrieval[key] = sum(found) / len(found) if found else None

    source_pass = []

    for row in successful:
        expected = _canonical_source(_source_name(row))
        sources = row.get("actual", {}).get("sources", [])

        names = {
            _canonical_source(item.get("source_filename"))
            for item in sources
            if isinstance(item, dict)
        }

        if expected and names:
            source_pass.append(names == {expected})

    return {
        "expected": expected_questions,
        "rows": len(rows),
        "completed": sum(_status(row) == "success" for row in rows),
        "failed": sum(_status(row) == "failed" for row in rows),
        "pending": max(0, expected_questions - len(rows)),
        "unscored": sum(
            (row.get("trace") or {}).get("scoring_status") == "unscorable"
            for row in rows
        ),
        "metrics": values,
        "retrieval_metrics": retrieval,
        "retrieval_valid": len(retrieval_scored),
        "source_accuracy": (
            sum(source_pass) / len(source_pass) if source_pass else None
        ),
        "source_accuracy_valid": len(source_pass),
        "latencies": [
            float(row["execution"]["latency_ms"])
            for row in successful
            if isinstance(row.get("execution", {}).get("latency_ms"), (int, float))
        ],
    }


def _pct(value: float | None) -> str:
    """Format percentages without losing useful decimal precision."""
    if value is None:
        return "—"

    percentage = value * 100
    formatted = f"{percentage:.1f}".rstrip("0").rstrip(".")
    return f"{formatted}%"


def _marker(value: Any, kind: str = "score") -> str:
    if kind == "status":
        return (
            "⚠"
            if value == "failed"
            else "✓"
            if value == "success"
            else "—"
        )

    return "—" if value is None else ("✓" if value >= 1 else "✕")


def _render_question_table(
    st: Any,
    table: list[dict[str, Any]],
) -> None:
    headers = (
        "Question",
        "Dataset",
        "Retrieval",
        "Correctness",
        "Groundedness",
        "Status",
    )

    head = "".join(
        f"<th>{escape(header)}</th>"
        for header in headers
    )

    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{escape(str(item.get(header, '—')))}</td>"
            for header in headers
        )
        + "</tr>"
        for item in table
    )

    st.markdown(
        f"""
        <div class="evaluation-table-wrap">
            <table class="evaluation-table">
                <thead>
                    <tr>{head}</tr>
                </thead>
                <tbody>{body}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard(
    st: Any,
    results_dir: str | Path = "data/evaluation/results",
    previous_dir: str | Path = "data/evaluation/results/phase10",
) -> None:
    rows = load_current_results(results_dir)
    summary = aggregate_results(rows)

    # Keep the light theme, but make all supporting/secondary text comfortably
    # readable instead of using Streamlit's low-contrast default gray.
    st.markdown(
        """
        <style>
        .evaluation-section-heading {
            color: #17324d !important;
            font-size: 1.15rem;
            line-height: 1.35;
            font-weight: 700;
            margin-top: 0.75rem;
            margin-bottom: 0.65rem;
        }

        /* Dashboard supporting text: darker blue-gray and larger than the
           default Streamlit caption text, while remaining softer than the
           primary headings. */
        [data-testid="stCaptionContainer"] {
            color: #526579 !important;
            font-size: 0.95rem !important;
            line-height: 1.5 !important;
            font-weight: 500 !important;
        }

        [data-testid="stCaptionContainer"] p,
        [data-testid="stCaptionContainer"] span,
        [data-testid="stCaptionContainer"] div {
            color: #526579 !important;
            font-size: 0.95rem !important;
            line-height: 1.5 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("RAG Evaluation")
    st.caption(
        "Measure retrieval and answer quality across the evaluation set."
    )

    st.markdown(
        "### Previous Evaluation → Current Evaluation → Improvement"
    )

    st.info(
        "Current evaluation is in progress. Improvement will be calculated "
        "after sufficient comparable results are available."
    )

    st.markdown("### Current Run")

    st.caption(
        f"Evaluation progress: {summary['completed']} of "
        f"{summary['expected']} questions evaluated."
    )

    cards = [
        ("Questions", summary["expected"]),
        ("Completed", summary["completed"]),
        ("Failed", summary["failed"]),
        ("Remaining", summary["pending"]),
    ]

    cols = st.columns(4)

    for col, (label, value) in zip(cols, cards):
        col.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{escape(label)}</div>
                <div class="metric-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Quality Results")

    st.markdown(
        '<div class="evaluation-section-heading">Retrieval Metrics</div>',
        unsafe_allow_html=True,
    )

    retrieval_quality = [
        ("Hit@5", summary["retrieval_metrics"]["hit_at_5"]),
        ("Precision@5", summary["retrieval_metrics"]["precision_at_5"]),
        ("Recall@5", summary["retrieval_metrics"]["recall_at_5"]),
        ("MRR", summary["retrieval_metrics"]["reciprocal_rank"]),
    ]

    cols = st.columns(4)

    for col, (label, value) in zip(cols, retrieval_quality):
        col.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{escape(label)}</div>
                <div class="metric-value">{_pct(value)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption("Retrieval metrics are averaged across evaluated questions.")

    st.markdown(
        '<div class="evaluation-section-heading">Answer Quality</div>',
        unsafe_allow_html=True,
    )

    answer_quality = [
        ("Answer Correctness", summary["metrics"]["answer_correctness"]),
        ("Groundedness", summary["metrics"]["groundedness"]),
        ("Source Accuracy", summary["source_accuracy"]),
    ]

    cols = st.columns(3)

    for col, (label, value) in zip(cols, answer_quality):
        col.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{escape(label)}</div>
                <div class="metric-value">{_pct(value)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if summary["source_accuracy"] is not None:
        count = summary["source_accuracy_valid"]

        st.caption(
            f"Source Accuracy is based on {count} evaluated question"
            f"{'s' if count != 1 else ''}."
        )

    st.markdown("### Question Results")

    by_key = {
        (row.get("dataset_name"), row.get("question_id")): row
        for row in rows
    }

    table = []

    expected_rows = load_expected_questions(
        Path(results_dir).parent / "questions"
    )

    if not expected_rows:
        expected_rows = load_expected_questions()

    for expected in expected_rows:
        row = by_key.get(
            (expected["dataset_name"], expected["question_id"])
        ) or {
            **expected,
            "execution": {"status": "pending"},
            "actual": {},
            "trace": {"scoring_status": "unscorable"},
        }

        retrieval = _metric(
            row,
            "retrieval_evaluation",
            "hit_at_5",
        )

        correctness = _metric(
            row,
            "answer_evaluation",
            "correctness_score",
        )

        grounded = _metric(
            row,
            "answer_evaluation",
            "groundedness_score",
        )

        table.append(
            {
                "Question": row.get("question_id", "—"),
                "Dataset": (
                    "Q1 FY2027"
                    if "Q1 FY2027" in str(row.get("dataset_name"))
                    else "FY2025–26"
                ),
                "Retrieval": _marker(retrieval),
                "Correctness": _marker(correctness),
                "Groundedness": _marker(grounded),
                "Status": _marker(_status(row), "status"),
                "_row": row,
            }
        )

    if table:
        _render_question_table(st, table)

        options = list(range(len(table)))

        selected = st.selectbox(
            "Inspect a question",
            options,
            format_func=lambda index: (
                f"{table[index]['Question']} · "
                f"{table[index]['Dataset']}"
            ),
        )

        row = table[selected]["_row"]

        st.markdown("### Question Details")
        st.write(row.get("question", "—"))

        st.caption(
            f"Expected document: "
            f"{_canonical_source(_source_name(row)) or '—'}"
        )

        actual = row.get("actual", {})

        retrieved = sorted(
            {
                _canonical_source(item.get("source_filename"))
                for item in actual.get("sources", [])
                if isinstance(item, dict)
            }
        )

        st.write(
            f"Retrieved documents: "
            f"{', '.join(retrieved) if retrieved else '—'}"
        )

        st.write(
            f"Expected answer: "
            f"{row.get('expected', {}).get('answer', '—')}"
        )

        st.write(
            f"Generated answer: {actual.get('answer') or '—'}"
        )

        if _status(row) == "failed":
            st.warning(
                "Generation failed — evaluation could not be scored."
            )
        elif _status(row) == "pending":
            st.info("— This question has not been evaluated yet.")
        else:
            st.write(
                f"Retrieval: "
                f"{_marker(_metric(row, 'retrieval_evaluation', 'hit_at_5'))}   "
                f"Correctness: "
                f"{_marker(_metric(row, 'answer_evaluation', 'correctness_score'))}   "
                f"Groundedness: "
                f"{_marker(_metric(row, 'answer_evaluation', 'groundedness_score'))}"
            )

        latency = row.get("execution", {}).get("latency_ms")

        st.caption(
            f"Latency: {latency / 1000:.1f}s"
            if isinstance(latency, (int, float))
            else "Latency: —"
        )

    st.markdown("### Key Findings")

    if summary["source_accuracy"] == 1:
        st.success(
            "✓ Source-document retrieval is working for scored results."
        )
    elif summary["source_accuracy"] is not None:
        st.warning(
            "⚠ Some scored results include unexpected source documents."
        )

    if summary["failed"] or summary["unscored"]:
        st.info(
            "⚠ Some questions remain unscored because generation failed "
            "or evaluation data is incomplete."
        )

    if summary["completed"]:
        st.info(
            "✓ Retrieval and answer quality are measured separately."
        )

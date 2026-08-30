"""Separate engineering/evaluation dashboard; never shown on the user page."""
import json
from pathlib import Path

def main() -> None:
    import streamlit as st
    st.set_page_config(page_title="Engineering Evaluation", layout="wide")
    st.title("Engineering / Evaluation")
    root = Path("data/evaluation")
    reports = sorted((root / "reports").glob("*.json")) if (root / "reports").exists() else []
    st.subheader("Evaluation Matrix")
    if reports:
        payload = json.loads(reports[-1].read_text(encoding="utf-8")); st.json(payload)
    else: st.info("Not evaluated — no evaluation reports are available.")
    st.subheader("Runtime Traces")
    traces = sorted((Path("data/traces")).glob("*.jsonl")) if Path("data/traces").exists() else []
    if traces:
        rows = [json.loads(line) for line in traces[-1].read_text(encoding="utf-8").splitlines() if line.strip()]
        st.dataframe(rows, use_container_width=True)
    else: st.info("Unavailable — no runtime traces are available.")

if __name__ == "__main__": main()

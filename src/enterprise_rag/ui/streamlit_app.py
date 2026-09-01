"""Light user-facing Streamlit presentation for the FastAPI backend."""

from enterprise_rag.ui.api_client import APIClient, APIClientError


def main() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="Enterprise Document Intelligence",
        page_icon="📄",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    # Keep the user UI intentionally simple and light.
    # The application communicates only with the FastAPI backend.
    st.markdown(
        """
        <style>
        /* ---------------------------------------------------------
           Application surface
           --------------------------------------------------------- */
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] > .main,
        [data-testid="stHeader"] {
            background: #f7f9fc !important;
        }

        [data-testid="stAppViewContainer"] > .main {
            color: #1f2937 !important;
        }

        .block-container {
            max-width: 900px;
            padding-top: 3rem;
            padding-bottom: 3rem;
        }


        /* ---------------------------------------------------------
           Typography
           --------------------------------------------------------- */
        h1, h2, h3, p, label,
        [data-testid="stCaptionContainer"] {
            color: #1f2937 !important;
        }

        [data-testid="stCaptionContainer"] {
            color: #475569 !important;
            font-size: 0.95rem !important;
        }

        [data-testid="stSelectbox"] label,
        [data-testid="stSelectbox"] div[role="button"] {
            color: #1f2937 !important;
            font-size: 0.95rem !important;
        }

        [data-testid="stSelectbox"] div[role="button"] {
            background: #ffffff !important;
            border: 1px solid #d6dee8 !important;
            border-radius: 8px !important;
        }

        h1 {
            color: #17324d !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em;
        }


        /* ---------------------------------------------------------
           Question text area
           --------------------------------------------------------- */
        [data-testid="stTextArea"] textarea {
            background: #ffffff !important;
            color: #1f2937 !important;
            border: 1px solid #d6dee8 !important;
            border-radius: 10px !important;
        }

        [data-testid="stTextArea"] textarea:focus {
            border-color: #4f86c6 !important;
            box-shadow: 0 0 0 1px #4f86c6 !important;
        }


        /* ---------------------------------------------------------
           Application buttons
           --------------------------------------------------------- */
        [data-testid="stButton"] button {
            border-radius: 8px !important;
            font-weight: 600 !important;
        }

        /* Active application buttons */
        [data-testid="stButton"] button:not(:disabled) {
            color: #ffffff !important;
        }

        [data-testid="stButton"] button:not(:disabled) p,
        [data-testid="stButton"] button:not(:disabled) span {
            color: #ffffff !important;
        }

        /* Disabled buttons remain readable */
        [data-testid="stButton"] button:disabled {
            color: #64748b !important;
            opacity: 0.75 !important;
        }

        [data-testid="stButton"] button:disabled p,
        [data-testid="stButton"] button:disabled span {
            color: #64748b !important;
        }


        /* ---------------------------------------------------------
           PDF uploader
           --------------------------------------------------------- */
        [data-testid="stFileUploader"] button {
            background: #ffffff !important;
            color: #17324d !important;
            border: 1px solid #d6dee8 !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }

        [data-testid="stFileUploader"] button p,
        [data-testid="stFileUploader"] button span {
            color: #17324d !important;
        }

        [data-testid="stFileUploader"] button svg {
            color: #17324d !important;
            fill: #17324d !important;
        }

        [data-testid="stFileUploader"] small {
            color: #475569 !important;
            font-size: 0.9rem !important;
        }

        [data-testid="stFileUploader"]
        [data-testid="stMarkdownContainer"] {
            color: #64748b !important;
        }


        /* ---------------------------------------------------------
           Alerts
           --------------------------------------------------------- */
        [data-testid="stAlert"] {
            border-radius: 10px !important;
        }


        /* ---------------------------------------------------------
           Answer card
           --------------------------------------------------------- */
        .answer-card {
            background: #ffffff !important;
            color: #1f2937 !important;
            border: 1px solid #e1e7ef;
            border-radius: 12px;
            padding: 1.25rem 1.35rem;
            margin-top: 0.5rem;
            margin-bottom: 1.25rem;
        }

        .answer-card,
        .answer-card * {
            color: #1f2937 !important;
        }


        /* ---------------------------------------------------------
           Source cards
           --------------------------------------------------------- */
        .source-card {
            background: #ffffff;
            border: 1px solid #e1e7ef;
            border-radius: 10px;
            padding: 0.8rem 1rem;
            margin: 0.5rem 0;
        }

        .source-title {
            color: #17324d !important;
            font-weight: 600;
        }

        .source-meta {
            color: #475569 !important;
            font-size: 0.95rem;
        }

        .request-id {
            color: #475569 !important;
            font-size: 0.82rem;
        }


        /* ---------------------------------------------------------
           Evaluation dashboard metric cards
           --------------------------------------------------------- */
        .metric-card {
            background: #ffffff;
            border: 1px solid #e1e7ef;
            border-radius: 12px;
            padding: 0.9rem 1rem;
            min-height: 76px;
        }

        .metric-label {
            color: #475569 !important;
            font-size: 0.9rem;
        }

        .metric-value {
            color: #17324d !important;
            font-size: 1.5rem;
            font-weight: 700;
            margin-top: 0.25rem;
        }

        .evaluation-table-wrap {
            background: #ffffff;
            border: 1px solid #e1e7ef;
            border-radius: 10px;
            overflow-x: auto;
        }

        .evaluation-table {
            width: 100%;
            border-collapse: collapse;
            color: #1f2937;
            font-size: 0.9rem;
        }

        .evaluation-table th {
            background: #f7f9fc;
            color: #17324d;
            font-weight: 700;
            text-align: left;
        }

        .evaluation-table th,
        .evaluation-table td {
            border-bottom: 1px solid #e1e7ef;
            padding: 0.65rem 0.7rem;
            white-space: nowrap;
        }

        .evaluation-table tr:last-child td { border-bottom: 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------
    # Application title
    # -------------------------------------------------------------
    st.title("Enterprise Document Intelligence")

    # -------------------------------------------------------------
    # Top-level navigation
    # -------------------------------------------------------------
    view = st.radio(
        "View",
        ("Ask Documents", "RAG Evaluation"),
        horizontal=True,
        label_visibility="collapsed",
    )

    if view == "RAG Evaluation":
        from enterprise_rag.ui.evaluation_dashboard import render_dashboard

        render_dashboard(st)
        return

    # -------------------------------------------------------------
    # Ask Documents view
    # -------------------------------------------------------------
    st.caption(
        "Ask questions and get grounded answers from your enterprise documents."
    )

    st.markdown("### Add a PDF")

    uploaded = st.file_uploader(
        "PDF document",
        type=["pdf"],
    )

    if st.button(
        "Process Document",
        disabled=uploaded is None,
    ):
        client = APIClient()

        with st.spinner("Processing and indexing document..."):
            try:
                payload = client.upload(
                    uploaded.name,
                    uploaded.getvalue(),
                )
            except APIClientError as exc:
                st.error(str(exc))
                return

        st.success(
            f"{payload.get('source_filename', uploaded.name)} "
            f"is ready ({payload.get('chunk_count', 0)} chunks)."
        )

    st.markdown("### Ask a question")

    question = st.text_area(
        "Question",
        placeholder="For example: What was the revenue in 2027?",
        height=110,
        label_visibility="collapsed",
    )

    if st.button(
        "Ask",
        type="primary",
        disabled=not question.strip(),
        use_container_width=False,
    ):
        client = APIClient()

        with st.spinner(
            "Searching documents and preparing your answer..."
        ):
            try:
                payload = client.query(question)
            except APIClientError as exc:
                st.error(str(exc))
                return

        if payload.get("clarification_required"):
            st.info(
                payload.get(
                    "clarification_question",
                    "I need a little more information to answer accurately.",
                )
            )
            return

        answer = payload.get("answer", "")

        st.subheader("Answer")

        st.markdown(
            f'<div class="answer-card">{answer}</div>',
            unsafe_allow_html=True,
        )

        sources = payload.get("sources", [])

        if sources:
            st.subheader("Sources")

            for index, source in enumerate(sources, 1):
                filename = (
                    source.get("source_filename")
                    or source.get("document_id", "Document")
                )

                page = (
                    source.get("page_start")
                    or source.get("page")
                )

                page_text = (
                    f"Page {page}"
                    if page
                    else "Page unavailable"
                )

                st.markdown(
                    f"""
                    <div class="source-card">
                        <div class="source-title">
                            {index}. {filename}
                        </div>
                        <div class="source-meta">
                            {page_text}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        request_id = payload.get("request_id")

        if request_id:
            st.markdown(
                f'<div class="request-id">'
                f"Request ID: {request_id}"
                f"</div>",
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()

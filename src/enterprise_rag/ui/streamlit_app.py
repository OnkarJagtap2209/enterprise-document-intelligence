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
        /* Force a clean light application surface even when Streamlit's
           global/browser theme is dark. */
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

        h1, h2, h3, p, label, [data-testid="stCaptionContainer"] {
            color: #1f2937 !important;
        }

        h1 {
            color: #17324d !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em;
        }

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

        /* Button text contrast */
        [data-testid="stButton"] button {
            border-radius: 8px !important;
            font-weight: 600 !important;
            color: #ffffff !important;
        }

        [data-testid="stButton"] button p,
        [data-testid="stButton"] button span {
            color: #ffffff !important;
        }

        [data-testid="stAlert"] {
            border-radius: 10px !important;
        }

        /* Answer text contrast */
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
            color: #64748b !important;
            font-size: 0.9rem;
        }

        .request-id {
            color: #64748b !important;
            font-size: 0.78rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Enterprise Document Intelligence")

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
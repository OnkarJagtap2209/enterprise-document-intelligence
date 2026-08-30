"""Light user-facing Streamlit presentation for the FastAPI backend."""
from .api_client import APIClient, APIClientError

def main() -> None:
    import streamlit as st
    st.set_page_config(page_title="Enterprise Document Intelligence", page_icon="📄", layout="centered")
    st.markdown("<style>body{background:#f7f9fc}.block-container{max-width:900px;padding-top:3rem}h1{color:#17324d}</style>", unsafe_allow_html=True)
    st.title("Enterprise Document Intelligence")
    st.caption("Ask questions. Get grounded answers from your enterprise documents.")
    question = st.text_area("Ask a question", placeholder="Ask a question about your documents...", height=100)
    if st.button("Ask", type="primary", disabled=not question.strip()):
        client = APIClient()
        with st.spinner("Searching documents and preparing your answer..."):
            try: payload = client.query(question)
            except APIClientError as exc: st.error(str(exc)); return
        if payload.get("clarification_required"):
            st.info(payload.get("clarification_question", "I need a little more information to answer accurately."))
            return
        st.subheader("Answer"); st.write(payload.get("answer", ""))
        sources = payload.get("sources", [])
        if sources:
            st.subheader("Sources")
            for index, source in enumerate(sources, 1):
                filename = source.get("source_filename") or source.get("document_id", "Document")
                page = source.get("page_start") or source.get("page")
                st.write(f"{index}. {filename}" + (f" — Page {page}" if page else ""))
        if payload.get("request_id"): st.caption(f"Request ID: {payload['request_id']}")

if __name__ == "__main__": main()

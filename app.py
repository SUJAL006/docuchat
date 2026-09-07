"""
DocuChat — A RAG-based Document Q&A App
----------------------------------------
Upload PDFs or text files, then ask questions and get answers grounded
in the actual document content, with source citations shown.

Run with:
    streamlit run app.py
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from rag_engine import (
    load_document,
    split_documents,
    get_embedding_model,
    build_vector_store,
    answer_question,
)

load_dotenv()

st.set_page_config(page_title="DocuChat", page_icon="📄", layout="wide")

st.title("📄 DocuChat — Ask Questions About Your Documents")
st.caption("A Retrieval-Augmented Generation (RAG) app built with LangChain, FAISS, and Claude")

# ---------------- Sidebar: API key + file upload ----------------
with st.sidebar:
    st.header("Setup")

    default_key = os.getenv("ANTHROPIC_API_KEY", "")
    api_key = st.text_input(
        "Anthropic API Key",
        value=default_key,
        type="password",
        help="Get one at https://console.anthropic.com. Used only for this session, never stored.",
    )

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload documents (PDF or TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    process_clicked = st.button("Process Documents", type="primary", use_container_width=True)

    st.divider()
    st.caption(
        "Tip: a sample document is included in `sample_docs/` if you want to "
        "test the app before using your own files."
    )

# ---------------- Session state ----------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- Process uploaded documents ----------------
if process_clicked:
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one document first.")
    else:
        with st.spinner("Reading, chunking, and embedding your documents..."):
            all_chunks = []
            for uploaded_file in uploaded_files:
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                docs = load_document(tmp_path)
                for d in docs:
                    d.metadata["source"] = uploaded_file.name
                chunks = split_documents(docs)
                all_chunks.extend(chunks)

                os.unlink(tmp_path)

            embeddings = get_embedding_model()
            st.session_state.vector_store = build_vector_store(all_chunks, embeddings)
            st.session_state.chat_history = []

        st.sidebar.success(f"Processed {len(uploaded_files)} document(s) into {len(all_chunks)} chunks.")

# ---------------- Chat interface ----------------
if st.session_state.vector_store is None:
    st.info("Upload one or more documents in the sidebar and click **Process Documents** to get started.")
else:
    for role, content, sources in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(content)
            if sources:
                with st.expander("View sources used"):
                    for s in sources:
                        st.markdown(f"**{s.metadata.get('source', 'unknown')}** (page {s.metadata.get('page', 'N/A')})")
                        st.caption(s.page_content[:300] + "...")

    query = st.chat_input("Ask a question about your documents...")

    if query:
        if not api_key:
            st.error("Please enter your Anthropic API key in the sidebar.")
        else:
            st.session_state.chat_history.append(("user", query, None))
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer, sources = answer_question(st.session_state.vector_store, query, api_key)
                    st.markdown(answer)
                    with st.expander("View sources used"):
                        for s in sources:
                            st.markdown(f"**{s.metadata.get('source', 'unknown')}** (page {s.metadata.get('page', 'N/A')})")
                            st.caption(s.page_content[:300] + "...")

            st.session_state.chat_history.append(("assistant", answer, sources))

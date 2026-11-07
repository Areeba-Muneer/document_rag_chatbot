import streamlit as st
from pathlib import Path
from rag_engine import RAGEngine, SUPPORTED_EXTENSIONS

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
UPLOAD_DIR = BASE_DIR / "uploaded_docs"
INDEX_DIR = BASE_DIR / "vector_store"

st.set_page_config(page_title="Document RAG Chatbot", page_icon="📚", layout="wide")
st.title("📚 Document-Based AI Chatbot")
st.caption("Ask questions using your documents. Answers are grounded in retrieved passages.")

@st.cache_resource
def load_engine():
    return RAGEngine(index_dir=INDEX_DIR)

engine = load_engine()
UPLOAD_DIR.mkdir(exist_ok=True)

with st.sidebar:
    st.header("Knowledge base")
    st.write("Add PDF, TXT, or DOCX documents.")
    uploads = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )
    if uploads:
        for uploaded in uploads:
            destination = UPLOAD_DIR / Path(uploaded.name).name
            destination.write_bytes(uploaded.getbuffer())
        st.success(f"Saved {len(uploads)} file(s). Click **Build / Refresh Index**.")

    if st.button("Build / Refresh Index", type="primary", use_container_width=True):
        with st.spinner("Extracting text and creating embeddings..."):
            try:
                count = engine.build_index([DOCS_DIR, UPLOAD_DIR])
                st.success(f"Index ready: {count} text chunks.")
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")

    st.divider()
    st.write("**LLM:** Google Gemini API")
    st.write("**Embedding model:** all-MiniLM-L6-v2 (local)")
    st.write("**Vector index:** FAISS")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources used"):
                for source in message["sources"]:
                    st.markdown(f"- **{source['source']}** — chunk {source['chunk_id']} (similarity: {source['score']:.3f})")
                    st.caption(source["text"])

question = st.chat_input("Ask a question about your documents...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating an answer..."):
            try:
                result = engine.answer(question)
                st.markdown(result["answer"])
                if result["sources"]:
                    with st.expander("Sources used", expanded=True):
                        for source in result["sources"]:
                            st.markdown(
                                f"- **{source['source']}** — chunk {source['chunk_id']} "
                                f"(similarity: {source['score']:.3f})"
                            )
                            st.caption(source["text"])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"]
                })
            except FileNotFoundError as exc:
                msg = str(exc)
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
            except Exception as exc:
                msg = f"Something went wrong: {exc}"
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})

if st.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()

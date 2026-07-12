"""
Streamlit web UI for the RAG system.

Run with:
    streamlit run app.py
"""

#Important: In command prompt or Terminal -> ollama serve OR ollama run gemma:2b 
# streamlit run app.py

import os
import streamlit as st

st.set_page_config(
    page_title="NLP Chatbot",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Document Q&A Chatbot (RAG)")
st.caption(f"Based on documents ingested")

# ── Load chain once per session ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading vector store…")
def get_chain():
    if not os.path.exists("chroma_db"):
        return None
    import rag
    vector_store = rag.load_vector_store()
    return rag.build_rag_chain(vector_store)


chain = get_chain()

if chain is None:
    st.error(
        "Vector store not found. "
        "Run `python ingest.py` in your terminal first, then refresh this page."
    )
    st.stop()

# ── Chat history ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f"**{src['file']}** — Page {src['page']}\n\n"
                        f"> {src['snippet']}…"
                    )

# ── Input ─────────────────────────────────────────────────────────────────────
if question := st.chat_input("Ask a question about the lectures…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            import rag as rag_module
            result = rag_module.ask(chain, question)

        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("Sources", expanded=False):
                for src in result["sources"]:
                    st.markdown(
                        f"**{src['file']}** — Page {src['page']}\n\n"
                        f"> {src['snippet']}…"
                    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })

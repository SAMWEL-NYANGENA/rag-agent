import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from graph import rag_graph, RAGState

load_dotenv()

st.set_page_config(page_title="PDF RAG Agent", layout="wide")
st.title(" PDF RAG Agent (LangGraph + Streamlit)")

# --- Session State ---
if "pdf_uploaded" not in st.session_state:
    st.session_state.pdf_uploaded = False
if "chunks" not in st.session_state:
    st.session_state.chunks = 0
if "memory" not in st.session_state:
    st.session_state.memory = []
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "context" not in st.session_state:
    st.session_state.context = []
if "uploaded_file_path" not in st.session_state:
    st.session_state.uploaded_file_path = ""

# --- Step 1: Upload PDF ---
st.header("Step 1: Upload PDF")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file and not st.session_state.pdf_uploaded:
    with st.spinner("Processing and embedding your PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        state: RAGState = {
            "uploaded_file_path": tmp_path,
            "pdf_uploaded": False,
            "chunks_count": 0,
            "query": "",
            "retrieved_docs": [],
            "memory": [],
            "answer": "",
        }

        ingest_state = rag_graph.invoke(state, config={"run_from": "ingest", "run_to": "ingest"})

        st.session_state.pdf_uploaded = True
        st.session_state.uploaded_file_path = tmp_path
        st.session_state.chunks = ingest_state["chunks_count"]

        st.success(f" PDF processed successfully ({st.session_state.chunks} chunks)")

elif st.session_state.pdf_uploaded:
    st.info(f" PDF already processed ({st.session_state.chunks} chunks). You can now ask questions.")

# --- Step 2: Ask a Question ---
st.header("Step 2: Ask a Question")
query = st.text_input("Ask something about your document:")

if st.button("Submit Question") and query:
    if not st.session_state.pdf_uploaded:
        st.warning(" Please upload a PDF first.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            state: RAGState = {
                "uploaded_file_path": st.session_state.uploaded_file_path,
                "pdf_uploaded": True,
                "chunks_count": st.session_state.chunks,
                "query": query,
                "retrieved_docs": [],
                "memory": st.session_state.memory,
                "answer": "",
            }

            result = rag_graph.invoke(state, config={"run_from": "retrieve", "run_to": "generate"})

            st.session_state.last_answer = result["answer"]
            st.session_state.context = result["retrieved_docs"]
            st.session_state.memory = result["memory"]

# --- Step 3: Display Answer ---
if st.session_state.last_answer:
    st.subheader(" Answer")
    st.write(st.session_state.last_answer)
    st.caption(f" Memory entries stored: {len(st.session_state.memory)}")

    with st.expander(" Retrieved Context"):
        for i, doc in enumerate(st.session_state.context, 1):
            st.text_area(f"Chunk {i}", doc, height=120)

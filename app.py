import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from graph import rag_graph, RAGState

load_dotenv()

# --- Streamlit Page Config ---
st.set_page_config(page_title="RAG Chat - by Samwel", layout="wide")

# --- Custom CSS for Chat UI ---
st.markdown("""
    <style>
    body { background-color: #F5F7FA; }
    .user-msg { 
        background-color: #DFF0D8; 
        padding: 10px; 
        border-radius: 10px; 
        margin: 6px 0; 
        color: #1B5E20;
    }
    .bot-msg { 
        background-color: #E3E7EC; 
        padding: 10px; 
        border-radius: 10px; 
        margin: 6px 0; 
        color: #212121;
    }
    .byline { 
        font-size: 0.8em; 
        color: #777; 
        text-align: center; 
        margin-top: 25px; 
        border-top: 1px solid #DDD;
        padding-top: 8px;
    }
    </style>
""", unsafe_allow_html=True)


st.title(" Multi-Doc RAG Chat")
st.caption("A LangGraph + Streamlit Retrieval-Augmented Generation App — *by Samwel* ")

# --- Initialize Session State ---
if "docs" not in st.session_state:
    st.session_state.docs = []  # List of file paths
if "memory" not in st.session_state:
    st.session_state.memory = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chunks" not in st.session_state:
    st.session_state.chunks = 0
if "summaries" not in st.session_state:
    st.session_state.summaries = {}

# --- Step 1: Upload Documents ---
st.header(" Step 1: Upload Your Documents")
uploaded_files = st.file_uploader(
    "Upload one or more documents (PDF, TXT, DOCX, MD)", 
    type=["pdf", "txt", "docx", "md"], 
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Processing your documents..."):
        temp_paths = []
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp:
                tmp.write(file.read())
                temp_paths.append(tmp.name)

        # Process all uploaded files together
        state: RAGState = {
            "uploaded_file_path": temp_paths,
            "pdf_uploaded": True,
            "chunks_count": 0,
            "query": "",
            "retrieved_docs": [],
            "memory": st.session_state.memory,
            "answer": "",
        }

        ingest_state = rag_graph.invoke(state, config={"run_from": "ingest", "run_to": "ingest"})

        st.session_state.docs.extend(temp_paths)
        st.session_state.chunks += ingest_state["chunks_count"]
        st.session_state.memory = ingest_state.get("memory", st.session_state.memory)

        st.success(f" Processed {len(temp_paths)} new document(s)! Total chunks: {st.session_state.chunks}")

        # --- NEW: Summarize each uploaded doc ---
        st.subheader(" Document Summaries")
        for doc_path in temp_paths:
            doc_name = os.path.basename(doc_path)
            try:
                with open(doc_path, "rb") as f:
                    content = f.read(2000)  # read first ~2KB
                # Attempt decoding for text preview
                try:
                    preview = content.decode("utf-8", errors="ignore").strip()
                except Exception:
                    preview = "[Binary file - no preview available]"
                summary = preview[:400] + ("..." if len(preview) > 400 else "")
                st.session_state.summaries[doc_name] = summary
            except Exception as e:
                st.session_state.summaries[doc_name] = f"[Error reading file: {e}]"

        # Display summaries in expandable boxes
        for doc_name, summary in st.session_state.summaries.items():
            with st.expander(f" {doc_name} — Preview"):
                st.write(summary)

# --- Step 2: Chat Interface ---
st.header(" Step 2: Ask Questions")

# Display chat history
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-msg'> You: {msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-msg'> SamRAG: {msg['content']}</div>", unsafe_allow_html=True)

query = st.chat_input("Ask something about your documents...")

if query:
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": query})

    if not st.session_state.docs:
        warning_msg = " Please upload at least one document first."
        st.session_state.chat_history.append({"role": "bot", "content": warning_msg})
        st.warning(warning_msg)
    else:
        with st.spinner("Retrieving context and generating answer..."):
            state: RAGState = {
                "uploaded_file_path": st.session_state.docs,
                "pdf_uploaded": True,
                "chunks_count": st.session_state.chunks,
                "query": query,
                "retrieved_docs": [],
                "memory": st.session_state.memory,
                "answer": "",
            }

             #   Prevent re-ingestion if docs already processed
            run_config = {"run_from": "retrieve", "run_to": "generate"}
            if not st.session_state.docs or not st.session_state.chunks:
                run_config = {"run_from": "ingest", "run_to": "generate"}

            result = rag_graph.invoke(state, config=run_config)

            # Persist updated memory
            st.session_state.memory = result["memory"]

            answer = result["answer"]
            st.session_state.chat_history.append({"role": "bot", "content": answer})

            # Display the bot response
            st.markdown(f"<div class='bot-msg'> SamRAG: {answer}</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("<div class='byline'>Made with  by Samwel — powered by LangGraph + Streamlit</div>", unsafe_allow_html=True)

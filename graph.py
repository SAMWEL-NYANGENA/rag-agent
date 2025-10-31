import os
import hashlib
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone.vectorstores import Pinecone
from pinecone import Pinecone as PC, ServerlessSpec
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_PROJECT = os.getenv("OPENAI_PROJECT")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    default_headers={"OpenAI-Project": OPENAI_PROJECT},
    temperature=0
)

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

pc = PC(api_key=PINECONE_API_KEY)
index_name = "rag-agent-index"

if index_name not in [i["name"] for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

class RAGState(TypedDict):
    uploaded_file_path: list[str]
    pdf_uploaded: bool
    chunks_count: int
    query: str
    retrieved_docs: list[str]
    memory: list[str]
    answer: str


def load_document(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path).load()
    elif ext == ".txt":
        return TextLoader(file_path).load()
    elif ext == ".docx":
        return Docx2txtLoader(file_path).load()
    else:
        return []


def ingest_node(state: RAGState) -> RAGState:
    """Load and embed multiple documents only once per file."""
    if not state.get("uploaded_file_path"):
        return state

    file_paths = state["uploaded_file_path"]
    if not isinstance(file_paths, list):
        file_paths = [file_paths]

    pc_client = PC(api_key=PINECONE_API_KEY)
    index = pc_client.Index(index_name)

    all_splits = []
    ingested_files = []
    total_chunks = 0

    for path in file_paths:
        file_id = os.path.basename(path)
        # Create a unique hash of the file’s contents
        with open(path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

        # Check if this hash already exists in Pinecone
        try:
            res = index.query(
                vector=[0.0] * 1536,
                top_k=1,
                filter={"file_hash": {"$eq": file_hash}},
                include_metadata=True,
            )
            if res and len(res.get("matches", [])) > 0:
                print(f" Skipping already indexed file: {file_id}")
                continue
        except Exception as e:
            print(f"Error checking index for {file_id}: {e}")

        # Load new documents
        docs = load_document(path)
        for doc in docs:
            doc.metadata["source"] = file_id
            doc.metadata["file_hash"] = file_hash

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # Assign unique IDs per chunk
        for i, split in enumerate(splits):
            split.metadata["chunk_id"] = f"{file_hash}_{i}"

        all_splits.extend(splits)
        ingested_files.append(file_id)
        total_chunks += len(splits)

    if not all_splits:
        return {**state, "pdf_uploaded": True, "chunks_count": 0}

    # Insert with IDs to prevent duplicates
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name, embedding=embeddings
    )
    vectorstore.add_documents(
        documents=all_splits,
        ids=[split.metadata["chunk_id"] for split in all_splits],
    )

    print(f" Ingested {len(ingested_files)} new file(s), {total_chunks} chunks total.")
    stats = index.describe_index_stats()
    print(f" Total vectors now: {stats['total_vector_count']}")

    return {**state, "pdf_uploaded": True, "chunks_count": total_chunks}

def retrieve_node(state: RAGState) -> RAGState:
    vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
    docs = vectorstore.similarity_search(state["query"], k=3)
    return {**state, "retrieved_docs": [d.page_content for d in docs]}


def memory_node(state: RAGState) -> RAGState:
    prev_memory = state.get("memory", [])
    context_snippet = "\n\n".join(state.get("retrieved_docs", []))
    new_memory = prev_memory + [f"Q: {state['query']}\nA: {context_snippet}"]
    return {**state, "memory": new_memory}


def generate_node(state: RAGState) -> RAGState:
    memory_context = "\n\n".join(state.get("memory", []))
    current_context = "\n\n".join(state["retrieved_docs"])

    prompt = f"""
You are a helpful assistant. Use both memory and retrieved context below to answer accurately.

Memory (previous Q&A):
{memory_context}

Current Context:
{current_context}

Question:
{state['query']}

Answer:
"""
    response = llm.invoke(prompt)
    return {**state, "answer": response.content}


graph = StateGraph(RAGState)
graph.add_node("ingest", ingest_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("memory", memory_node)
graph.add_node("generate", generate_node)

graph.add_edge(START, "ingest")
graph.add_edge("ingest", "retrieve")
graph.add_edge("retrieve", "memory")
graph.add_edge("memory", "generate")
graph.add_edge("generate", END)

rag_graph = graph.compile()

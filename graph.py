import os
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
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
        dimension=1536,  # for text-embedding-3-small
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

#
class RAGState(TypedDict):
    uploaded_file_path: str
    pdf_uploaded: bool
    chunks_count: int
    query: str
    retrieved_docs: list[str]
    memory: list[str]
    answer: str



def ingest_node(state: RAGState) -> RAGState:
    """Load and embed PDF if not already processed."""
    if state.get("pdf_uploaded"):
        return state

    pdf_path = state["uploaded_file_path"]
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    PineconeVectorStore.from_documents(
        documents=splits, embedding=embeddings, index_name=index_name
    )

    return {**state, "pdf_uploaded": True, "chunks_count": len(splits)}


def retrieve_node(state: RAGState) -> RAGState:
    """Retrieve relevant chunks based on query."""
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name, embedding=embeddings
    )
    docs = vectorstore.similarity_search(state["query"], k=3)
    return {**state, "retrieved_docs": [d.page_content for d in docs]}


def memory_node(state: RAGState) -> RAGState:
    """Keep track of chat memory across queries."""
    prev_memory = state.get("memory", [])
    context_snippet = "\n\n".join(state.get("retrieved_docs", []))
    new_memory = prev_memory + [f"Q: {state['query']}\nA: {context_snippet}"]
    return {**state, "memory": new_memory}


def generate_node(state: RAGState) -> RAGState:
    """Generate answer using context + memory."""
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


# graph
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

"""
Core RAG (Retrieval-Augmented Generation) logic for DocuChat.

Pipeline:
1. Load documents (PDF / TXT)
2. Split into overlapping chunks
3. Embed chunks locally (free, no API key needed) with sentence-transformers
4. Store embeddings in a FAISS vector index
5. On a user query -> embed query -> retrieve top-k similar chunks
6. Pass retrieved chunks + query to Claude, instructed to answer ONLY from context
"""

from typing import List, Tuple

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from anthropic import Anthropic

# ---- Config ----
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # free, local, no API key
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
CLAUDE_MODEL = "claude-sonnet-4-6"


def load_document(file_path: str) -> List[Document]:
    """Load a single document (PDF or TXT) into LangChain Document objects."""
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks suitable for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Load a free, local sentence-transformers embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def build_vector_store(chunks: List[Document], embeddings: HuggingFaceEmbeddings) -> FAISS:
    """Build an in-memory FAISS vector store from document chunks."""
    return FAISS.from_documents(chunks, embeddings)


def retrieve_relevant_chunks(vector_store: FAISS, query: str, k: int = TOP_K) -> List[Document]:
    """Retrieve the top-k most relevant chunks for a query."""
    return vector_store.similarity_search(query, k=k)


def build_prompt(query: str, chunks: List[Document]) -> str:
    """Construct a grounded prompt that instructs Claude to answer only from context."""
    context = "\n\n---\n\n".join(
        f"[Source: {c.metadata.get('source', 'unknown')}, page {c.metadata.get('page', 'N/A')}]\n{c.page_content}"
        for c in chunks
    )
    return f"""You are a helpful assistant that answers questions using ONLY the context provided below.
If the answer is not contained in the context, say "I don't have enough information in the provided documents to answer that."
Do not use outside knowledge. Be concise and mention which source you drew from.

Context:
{context}

Question: {query}

Answer:"""


from langchain_ollama import ChatOllama

def generate_answer(query: str, chunks: List[Document], api_key: str = "") -> str:
    """Generate grounded answer locally using Ollama."""
    llm = ChatOllama(model="llama3.2", temperature=0)
    prompt = build_prompt(query, chunks)
    response = llm.invoke(prompt)
    return response.content


def answer_question(vector_store: FAISS, query: str, api_key: str) -> Tuple[str, List[Document]]:
    """Full RAG pipeline: retrieve relevant chunks, then generate a grounded answer."""
    chunks = retrieve_relevant_chunks(vector_store, query)
    answer = generate_answer(query, chunks, api_key)
    return answer, chunks

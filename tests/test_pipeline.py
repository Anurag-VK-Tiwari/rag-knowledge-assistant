import pytest
from fpdf import FPDF
from langchain.embeddings.base import Embeddings
from src.loader import load_all_pdfs
from src.chunker import chunk_documents
from src.indexer import build_faiss_index
from src.retriever import get_retriever
from src.generator import build_rag_prompt


class MockEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[0.0]*10 for _ in texts]

    def embed_query(self, text):
        return [0.0]*10


@pytest.fixture(scope="session")
def sample_docs(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("docs")
    pdf_file = tmp_dir / "sample.pdf"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10,
                   "LangChain is a framework for building applications with "
                   "LLMs. Retrieval-Augmented Generation (RAG) combines LLMs "
                   "with a knowledge base."
                   )
    pdf.output(pdf_file.as_posix())
    return tmp_dir


@pytest.fixture
def embeddings():
    return MockEmbeddings()


def test_load_documents(sample_docs):
    docs = load_all_pdfs(sample_docs.as_posix())
    assert len(docs) > 0
    assert "LangChain" in docs[0].page_content


def test_chunk_documents(sample_docs):
    docs = load_all_pdfs(sample_docs.as_posix())
    chunks = chunk_documents(docs, chunk_size=30, overlap=5)
    assert len(chunks) >= 1
    assert all(hasattr(c, "page_content") for c in chunks)


def test_embeddings_and_indexing(sample_docs, embeddings):
    docs = load_all_pdfs(sample_docs.as_posix())
    chunks = chunk_documents(docs, chunk_size=30, overlap=5)
    vectorstore = build_faiss_index(chunks, embeddings)
    assert vectorstore is not None
    assert hasattr(vectorstore, "similarity_search")


def test_retriever(sample_docs, embeddings):
    docs = load_all_pdfs(sample_docs.as_posix())
    chunks = chunk_documents(docs, chunk_size=30, overlap=5)
    vectorstore = build_faiss_index(chunks, embeddings)
    retriever = get_retriever(vectorstore, k=2)
    results = retriever.invoke("What is Langchain?")
    assert len(results) > 0
    assert any("LangChain" in r.page_content for r in results)


def test_build_rag_prompt():
    query = "What is LangChain?"
    context = "LangChain is a framework for LLM applications."
    prompt = build_rag_prompt(query, context)
    assert "LangChain" in prompt
    assert query in prompt

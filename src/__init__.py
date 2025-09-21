# src/__init__.py
from .loader import load_all_pdfs
from .chunker import chunk_documents
from .embeddings import get_hf_embeddings
from .indexer import build_faiss_index, load_faiss_index
from .retriever import get_retriever
from .generator import build_rag_prompt, call_gemini_text
from .evaluate import load_qa_pairs, evaluate_predictions

__all__ = [
    "load_all_pdfs",
    "chunk_documents",
    "get_hf_embeddings",
    "build_faiss_index",
    "load_faiss_index",
    "get_retriever",
    "build_rag_prompt",
    "call_gemini_text",
    "load_qa_pairs",
    "evaluate_predictions",
]

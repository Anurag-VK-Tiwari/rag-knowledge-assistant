# Rag-knowledge-assistant
A small, reproducible RAG system using LangChain, FAISS, and Gemini models that compares baseline LLM answers to RAG-augmented answers and reports evaluation metrics (accuracy, hallucination reduction).

**Problem :** Large language models can give fluent answers but often hallucinate or miss domain-specific facts when they don't have access to the user's documents. Retrieval-Augmented Generation (RAG) mitigates this by retrieving relevant document passages and conditioning generation on them.

**Approach :** We build a reproducible pipeline that ingests documents, creates embeddings, stores vectors in FAISS/ChromaDB, retrieves context, and compares two systems: (A) baseline LLM answering without retrieval, (B) RAG pipeline that provides retrieved context to the LLM. We evaluate on a small, curated QA set from the same documents using automatic metrics and a hallucination check.

**Contributions :**

Reproducible repo with scripts to ingest docs, build embeddings, run retrieval, and evaluate results.
Side-by-side comparison of baseline vs RAG with quantitative metrics and short qualitative examples.
A README that doubles as a mini research paper describing experiments, results, and conclusions.

**Keywords :**
LLMs, RAG, embeddings, vector DB, FAISS, LangChain, Gemini, evaluation, hallucination

**Tech Stack :**
Python 3.10+
LangChain (>=0.0.x)
Gemini LLMs
Embedding model: sentence-transformers
Vector DB: FAISS 
Evaluation: scikit-learn, rouge/bert-score (optional), custom hallucination checks
Optional: Docker for containerization, GitHub Actions for CI

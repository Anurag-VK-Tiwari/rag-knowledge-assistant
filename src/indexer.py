# src/indexer.py
from langchain_community.vectorstores import FAISS
import os


def build_faiss_index(docs, embeddings, index_path="faiss_index"):
    vectorstore = FAISS.from_documents(docs, embeddings)
    if not os.path.exists(index_path):
        os.makedirs(index_path)
    vectorstore.save_local(index_path)
    return vectorstore


def load_faiss_index(index_path, embeddings):
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

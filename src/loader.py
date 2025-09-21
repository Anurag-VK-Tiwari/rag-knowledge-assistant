# src/loader.py
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from typing import List
from langchain.schema import Document


def load_pdf(path: str) -> List[Document]:
    loader = PyPDFLoader(path)
    docs = loader.load()
    return docs


def load_all_pdfs(folder: str):
    p = Path(folder)
    docs = []
    for f in p.glob("*.pdf"):
        docs.extend(load_pdf(str(f)))
    return docs

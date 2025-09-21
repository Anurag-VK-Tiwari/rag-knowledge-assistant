# src/retriever.py
def get_retriever(vectorstore, k=4, score_threshold=None):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever

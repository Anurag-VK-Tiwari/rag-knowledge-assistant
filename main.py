from dotenv import load_dotenv
from src import (
    load_all_pdfs,
    chunk_documents,
    get_hf_embeddings,
    build_faiss_index,
    get_retriever,
    build_rag_prompt,
    call_gemini_text,
)

load_dotenv()


def run_rag(query: str, use_gemini: bool = False):
    """End-to-end RAG pipeline: load -> chunk -> embed -> retrieve -> """
    """ -> generate."""

    # 1. Load documents
    docs = load_all_pdfs("data/docs/")

    # 2. Chunk into smaller pieces
    chunks = chunk_documents(docs)

    # 3. Embeddings + index
    embeddings = get_hf_embeddings()
    vectorstore = build_faiss_index(chunks, embeddings)

    # 4. Retriever
    retriever = get_retriever(vectorstore, k=3)

    # 5. Retrieve context
    context_docs = retriever.invoke(query)
    context = "\n".join([d.page_content for d in context_docs])

    # 6. Build RAG prompt
    prompt = build_rag_prompt(query, context)

    # 7. Generate answer
    return call_gemini_text(prompt)


if __name__ == "__main__":
    print("=== RAG Knowledge Assistant ===")
    while True:
        user_input = input("\nAsk a question (or type 'exit'): ")
        if user_input.lower() == "exit":
            break

        # change to True if you want Gemini
        answer = run_rag(user_input, use_gemini=True)
        print(f"\n>>> {answer}\n")

import os
from src.loader import load_all_pdfs
from src.chunker import chunk_documents
from src.embeddings import get_hf_embeddings
from src.indexer import build_faiss_index, load_faiss_index
from src.retriever import get_retriever
from src.generator import build_rag_prompt, call_gemini_text
from src.evaluate import load_qa_pairs, evaluate_predictions, hallucination_estimate


def prepare_index(docs_folder="data/docs", index_path="faiss_index"):
    """
    Load PDFs, chunk them, generate embeddings, and build FAISS index.
    """
    print(f"📄 Loading documents from: {docs_folder}")
    docs = load_all_pdfs(docs_folder)
    chunks = chunk_documents(docs)
    emb = get_hf_embeddings()
    vs = build_faiss_index(chunks, emb, index_path=index_path)
    print(f"🚀 FAISS index ready with {len(chunks)} chunks.")
    return vs, emb


def run_experiment(
    qa_path="data/qa_pairs/test.jsonl",
    index_path="faiss_index",
    baseline_model="gemini-1.5-flash",
    rag_model="gemini-1.5-flash",
    k=4,
    temperature=0.0
):
    """
    Run baseline vs RAG experiments and compute metrics.
    """
    print(f"🔄 Loading FAISS index from {index_path}")
    emb = get_hf_embeddings()
    vs = load_faiss_index(index_path, emb)
    retriever = get_retriever(vs, k=k)

    qa = load_qa_pairs(qa_path)
    questions = [q["question"] for q in qa]
    golds = [q["answer"] for q in qa]

    baseline_preds = []
    rag_preds = []
    rag_hall_estimates = []

    for q in questions:
        # Baseline LLM
        b = call_gemini_text(q, model=baseline_model, temperature=temperature)
        baseline_preds.append(b)

        # RAG
        docs = retriever.invoke(q)
        context_text = " ".join([d.page_content if hasattr(d, "page_content") else str(d) for d in docs])
        prompt = build_rag_prompt(q, context_text)
        r = call_gemini_text(prompt, model=rag_model, temperature=temperature)
        rag_preds.append(r)

        # Hallucination estimation
        rag_hall_estimates.append(hallucination_estimate(r, context_text))

    # Evaluate
    baseline_metrics = evaluate_predictions(baseline_preds, golds)
    rag_metrics = evaluate_predictions(rag_preds, golds)

    results = {
        "baseline": baseline_metrics,
        "rag": rag_metrics,
        "rag_hallucination_mean": sum(rag_hall_estimates)/len(rag_hall_estimates)
    }

    print("✅ Experiment complete.")
    return results


if __name__ == "__main__":
    import json
    res = run_experiment()
    print(json.dumps(res, indent=4))

# RAG-based Knowledge Assistant

## Description
A reproducible RAG pipeline to compare baseline LLM answers vs RAG-grounded answers using FAISS + HuggingFace embeddings and OpenAI/Gemini models.

## Goals / Showcases
- Compare baseline GPT answers vs RAG answers
- Add evaluation metrics: exact match, ROUGE-L, and a simple hallucination estimate
- Publish results in README like a mini research report

## How to run
1. Install requirements:
   pip install -r requirements.txt

2. Put PDFs in `data/docs/` and QA pairs in `data/qa_pairs/test.jsonl`.

3. Build the index:
   python -c "from src.experiment import prepare_index; prepare_index()"

4. Run experiment:
   python -c "from src.experiment import run_experiment; print(run_experiment())"

## Evaluation
- Exact match: strict correctness
- ROUGE-L: overlap measure
- Hallucination estimate: fraction of predicted tokens not present in retrieved context (crude proxy)

## Observations (example)
- RAG should reduce hallucination and increase ROUGE-L for knowledge-specific Qs.
- Baseline better for open-ended creative Qs.

## Notes
- Swap embeddings model or LLM via env vars.
- Replace FAISS with Chroma/Pinecone for production.

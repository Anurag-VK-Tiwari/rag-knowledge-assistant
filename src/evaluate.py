# src/evaluate.py
import json
from typing import List
from rouge_score import rouge_scorer
import numpy as np


def load_qa_pairs(path):
    # expect newline-delimited json: {"id":..., "question":..., "answer":...}
    qa = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            qa.append(json.loads(line))
    return qa


def exact_match_score(pred: str, gold: str):
    return int(pred.strip().lower() == gold.strip().lower())


def rouge_l_score(pred: str, gold: str):
    scorer = rouge_scorer.RougeScorer(['rougeL'])
    score = scorer.score(gold, pred)
    return score['rougeL'].fmeasure


def evaluate_predictions(preds: List[str], golds: List[str]):
    # accuracy with exact match and average rouge-L
    exacts = [exact_match_score(p,g) for p,g in zip(preds,golds)]
    rouge_scores = [rouge_l_score(p,g) for p,g in zip(preds,golds)]
    return {
        "exact_match": np.mean(exacts),
        "rouge_l": np.mean(rouge_scores),
        "n": len(preds)
    }


# crude hallucination estimate: measure proportion of tokens in pred that do NOT overlap with retrieved context
def hallucination_estimate(pred, context):
    context_set = set(context.lower().split())
    pred_tokens = pred.lower().split()
    if not pred_tokens: return 0.0
    non_overlap = sum(1 for t in pred_tokens if t not in context_set)
    return non_overlap / len(pred_tokens)

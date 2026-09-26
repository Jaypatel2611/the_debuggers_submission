"""PRD §17 evaluation framework: macro F0.5 (the leaderboard metric, §1) plus
the diagnostic metrics table. Predictions/truth are
{source1_entity_id: set(matched_ids)}."""
from __future__ import annotations


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def _precision_recall(pred: set[str], truth: set[str]) -> tuple[float, float]:
    if not pred and not truth:
        return 1.0, 1.0
    if not pred:
        return 1.0, 0.0
    if not truth:
        return 0.0, 1.0
    tp = len(pred & truth)
    precision = tp / len(pred)
    recall = tp / len(truth)
    return precision, recall


def macro_f05(predictions: dict[str, set[str]], truth: dict[str, set[str]]) -> float:
    scores = []
    for s1_id, truth_set in truth.items():
        pred_set = predictions.get(s1_id, set())
        p, r = _precision_recall(pred_set, truth_set)
        scores.append(f_beta(p, r, beta=0.5))
    return sum(scores) / len(scores) if scores else 0.0


def evaluate(predictions: dict[str, set[str]], truth: dict[str, set[str]]) -> dict:
    precisions, recalls = [], []
    tp_total = pred_total = truth_total = 0
    singleton_total = singleton_correct = 0
    for s1_id, truth_set in truth.items():
        pred_set = predictions.get(s1_id, set())
        p, r = _precision_recall(pred_set, truth_set)
        precisions.append(p)
        recalls.append(r)
        tp_total += len(pred_set & truth_set)
        pred_total += len(pred_set)
        truth_total += len(truth_set)
        if not truth_set:
            singleton_total += 1
            if not pred_set:
                singleton_correct += 1
    return {
        "macro_f05": macro_f05(predictions, truth),
        "macro_precision": sum(precisions) / len(precisions) if precisions else 0.0,
        "macro_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "micro_precision": tp_total / pred_total if pred_total else 1.0,
        "micro_recall": tp_total / truth_total if truth_total else 1.0,
        "singleton_accuracy": singleton_correct / singleton_total if singleton_total else None,
    }

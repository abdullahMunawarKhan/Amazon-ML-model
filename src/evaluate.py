"""Precision-heavy entity-level F_0.5 evaluation."""

from typing import Dict, Iterable, Set


def entity_f05(predicted: Iterable[str], truth: Iterable[str]) -> float:
    predicted_set, truth_set = set(predicted), set(truth)
    if not truth_set:
        return 1.0 if not predicted_set else 0.0
    if not predicted_set:
        return 0.0
    precision = len(predicted_set & truth_set) / len(predicted_set)
    recall = len(predicted_set & truth_set) / len(truth_set)
    denominator = 0.25 * precision + recall
    return (1.25 * precision * recall / denominator) if denominator else 0.0


def macro_f05(predictions: Dict[str, Iterable[str]], truth: Dict[str, Iterable[str]]) -> float:
    return sum(entity_f05(predictions.get(key, ()), values) for key, values in truth.items()) / max(1, len(truth))
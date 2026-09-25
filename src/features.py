"""Pairwise string features for candidate records."""

import math
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Dict, Iterable, List

import pandas as pd


def _tokens(value: str) -> set[str]:
    return set(value.split())


def _jaccard(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    return len(a & b) / len(a | b) if a or b else 1.0


def _levenshtein_ratio(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, 1):
        current = [left_index]
        for right_index, right_char in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_char != right_char),
            ))
        previous = current
    return 1.0 - previous[-1] / max(len(left), len(right))


def _token_sort_ratio(left: str, right: str) -> float:
    return _levenshtein_ratio(" ".join(sorted(left.split())), " ".join(sorted(right.split())))


def _token_set_ratio(left: str, right: str) -> float:
    left_tokens, right_tokens = _tokens(left), _tokens(right)
    common = left_tokens & right_tokens
    left_remainder = left_tokens - common
    right_remainder = right_tokens - common
    return _levenshtein_ratio(" ".join(sorted(common | left_remainder)), " ".join(sorted(common | right_remainder)))


def _char_ngrams(value: str, size: int = 3) -> set[str]:
    compact = re.sub(r"\s+", " ", value)
    return {compact[index:index + size] for index in range(max(0, len(compact) - size + 1))}


def _cosine(left: str, right: str) -> float:
    a, b = _char_ngrams(left), _char_ngrams(right)
    if not a or not b:
        return 1.0 if a == b else 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))


def _tfidf_cosine(left: str, right: str) -> float:
    """Two-document TF-IDF cosine; IDF is derived only from this pair."""
    left_terms, right_terms = left.split(), right.split()
    vocabulary = set(left_terms) | set(right_terms)
    if not vocabulary:
        return 1.0
    left_counts, right_counts = Counter(left_terms), Counter(right_terms)
    left_vector, right_vector = [], []
    for term in vocabulary:
        document_frequency = int(term in left_counts) + int(term in right_counts)
        weight = math.log(3.0 / (1.0 + document_frequency)) + 1.0
        left_vector.append(left_counts[term] * weight)
        right_vector.append(right_counts[term] * weight)
    numerator = sum(a * b for a, b in zip(left_vector, right_vector))
    denominator = math.sqrt(sum(a * a for a in left_vector) * sum(b * b for b in right_vector))
    return numerator / denominator if denominator else 0.0


def pair_features(left: pd.Series, right: pd.Series) -> Dict[str, float]:
    name_norm = SequenceMatcher(None, left["name_norm"], right["name_norm"]).ratio()
    address_norm = SequenceMatcher(None, left["address_norm"], right["address_norm"]).ratio()
    return {
        "name_jaccard": _jaccard(left["name_norm"], right["name_norm"]),
        "name_core_jaccard": _jaccard(left["name_core"], right["name_core"]),
        "name_levenshtein": _levenshtein_ratio(left["name_norm"], right["name_norm"]),
        "name_token_sort": _token_sort_ratio(left["name_norm"], right["name_norm"]),
        "name_token_set": _token_set_ratio(left["name_norm"], right["name_norm"]),
        "name_sequence": name_norm,
        "name_cosine": _cosine(left["name_norm"], right["name_norm"]),
        "name_tfidf_cosine": _tfidf_cosine(left["name_norm"], right["name_norm"]),
        "address_jaccard": _jaccard(left["address_norm"], right["address_norm"]),
        "address_levenshtein": _levenshtein_ratio(left["address_norm"], right["address_norm"]),
        "address_token_sort": _token_sort_ratio(left["address_norm"], right["address_norm"]),
        "address_token_set": _token_set_ratio(left["address_norm"], right["address_norm"]),
        "address_sequence": address_norm,
        "address_cosine": _cosine(left["address_norm"], right["address_norm"]),
        "address_tfidf_cosine": _tfidf_cosine(left["address_norm"], right["address_norm"]),
        "country_match": float(left["country"] == right["country"]),
        "name_length_delta": abs(len(left["name_norm"]) - len(right["name_norm"])),
        "address_length_delta": abs(len(left["address_norm"]) - len(right["address_norm"])),
    }


def build_feature_frame(source1: pd.DataFrame, target: pd.DataFrame, candidates: Dict[str, Iterable[str]]) -> pd.DataFrame:
    left = source1.set_index("entity_id")
    right = target.set_index("entity_id")
    rows: List[Dict[str, object]] = []
    for source1_id, candidate_ids in candidates.items():
        if source1_id not in left.index:
            continue
        for candidate_id in candidate_ids:
            if candidate_id in right.index:
                row = {"source1_entity_id": source1_id, "candidate_entity_id": candidate_id}
                row.update(pair_features(left.loc[source1_id], right.loc[candidate_id]))
                rows.append(row)
    return pd.DataFrame(rows)
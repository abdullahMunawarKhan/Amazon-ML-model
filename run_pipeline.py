"""Run the baseline or XGBoost entity-resolution pipeline.

The default baseline is intentionally runnable without model training. It uses the
same blocking and feature path as the trained matcher, so its validation score is a
meaningful checkpoint before enabling XGBoost.
"""

import argparse
import sqlite3
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from src.blocking import build_index, candidates_for_row
from src.data_loading import read_ground_truth, read_source
from src.evaluate import macro_f05
from src.features import build_feature_frame
from src.inference import write_results
from src.model import PairMatcher
from src.preprocessing import add_normalized_columns
from src.train_val_split import split_by_source1


def truth_map(frame: pd.DataFrame) -> Dict[str, List[str]]:
    return {
        row.source1_entity_id: [item for item in row.matched_entity_ids.split(",") if item]
        for row in frame.itertuples(index=False)
    }


def fetch_records(connection: sqlite3.Connection, ids: Iterable[str]) -> pd.DataFrame:
    ids = list(dict.fromkeys(ids))
    if not ids:
        return pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country", "name_norm", "name_core", "address_norm"])
    placeholders = ",".join("?" for _ in ids)
    frame = pd.read_sql_query(
        f"SELECT entity_id, business_name, business_address, country, name_norm, name_core, address_norm FROM records WHERE entity_id IN ({placeholders})",
        connection, params=ids,
    )
    return frame


def make_candidates(frame: pd.DataFrame, connection: sqlite3.Connection, limit: int) -> Dict[str, List[str]]:
    normalized = add_normalized_columns(frame)
    return {
        row.entity_id: candidates_for_row(pd.Series(row._asdict()), connection, limit)
        for row in normalized.itertuples(index=False)
    }


def baseline_predictions(features: pd.DataFrame, threshold: float) -> Dict[str, List[str]]:
    if features.empty:
        return {}
    score = (
        0.45 * features["name_core_jaccard"]
        + 0.25 * features["name_sequence"]
        + 0.20 * features["address_jaccard"]
        + 0.10 * features["address_sequence"]
    )
    selected = features.loc[(score >= threshold) & ((features["name_core_jaccard"] >= 0.5) | (features["address_jaccard"] >= 0.5))].copy()
    selected["score"] = score[selected.index]
    selected = selected.sort_values(["source1_entity_id", "score", "candidate_entity_id"], ascending=[True, False, True])
    return selected.groupby("source1_entity_id")["candidate_entity_id"].apply(list).to_dict()


def score_checkpoint(source1: pd.DataFrame, target: pd.DataFrame, candidates: Dict[str, List[str]], truth: Dict[str, List[str]], threshold: float, matcher: PairMatcher | None = None) -> float:
    features = build_feature_frame(source1, target, candidates)
    predictions = matcher.predict_ids(features) if matcher else baseline_predictions(features, threshold)
    score = macro_f05(predictions, truth)
    print(f"validation F_0.5: {score:.6f} ({len(source1):,} Source-1 entities)")
    return score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--work-dir", type=Path, default=Path("work"))
    parser.add_argument("--mode", choices=("baseline", "xgboost"), default="baseline")
    parser.add_argument("--max-candidates", type=int, default=120)
    parser.add_argument("--validation-entities", type=int, default=10000)
    parser.add_argument("--threshold", type=float, default=0.78)
    args = parser.parse_args()

    train_dir, test_dir = args.data_dir / "train", args.data_dir / "test"
    args.work_dir.mkdir(parents=True, exist_ok=True)
    train_s1 = read_source(train_dir / "train_source1.tsv")
    train_truth_frame = read_ground_truth(train_dir / "train_ground_truth.tsv")
    test_s1 = read_source(test_dir / "test_source1.tsv")
    train_truth = truth_map(train_truth_frame)

    train_db = args.work_dir / "train_records.sqlite"
    test_db = args.work_dir / "test_records.sqlite"
    if not train_db.exists():
        print("building training SQLite index")
        build_index([train_dir / "train_source2.tsv", train_dir / "train_source3.tsv"], train_db)
    if not test_db.exists():
        print("building test SQLite index")
        build_index([test_dir / "test_source2.tsv", test_dir / "test_source3.tsv"], test_db)

    train_split_s1, valid_s1, _, valid_truth_frame = split_by_source1(train_s1, train_truth_frame)
    if len(valid_s1) > args.validation_entities:
        valid_s1 = valid_s1.sample(args.validation_entities, random_state=42)
    valid_truth = truth_map(valid_truth_frame)
    valid_truth = {key: valid_truth.get(key, []) for key in valid_s1["entity_id"]}
    with sqlite3.connect(train_db) as connection:
        valid_candidates = make_candidates(valid_s1, connection, args.max_candidates)
        valid_target = fetch_records(connection, {item for values in valid_candidates.values() for item in values})
    matcher = None
    if args.mode == "xgboost":
        # Sample only from the training partition so validation Source-1 entities
        # cannot contribute labeled pairs to the fitted matcher.
        train_s1_sample = train_split_s1.sample(
            min(len(train_split_s1), max(args.validation_entities, 20000)),
            random_state=41,
        )
        with sqlite3.connect(train_db) as connection:
            train_candidates = make_candidates(train_s1_sample, connection, args.max_candidates)
            train_target = fetch_records(connection, {item for values in train_candidates.values() for item in values})
        train_features = build_feature_frame(train_s1_sample, train_target, train_candidates)
        labels = train_features.apply(lambda row: row.candidate_entity_id in set(train_truth.get(row.source1_entity_id, [])), axis=1)
        matcher = PairMatcher(threshold=args.threshold).fit(train_features, labels)
    score_checkpoint(valid_s1, valid_target, valid_candidates, valid_truth, args.threshold, matcher)

    all_candidates: Dict[str, List[str]] = {}
    all_predictions: Dict[str, List[str]] = {}
    with sqlite3.connect(test_db) as connection:
        for start in range(0, len(test_s1), 5000):
            chunk = test_s1.iloc[start:start + 5000].copy()
            candidates = make_candidates(chunk, connection, args.max_candidates)
            target = fetch_records(connection, {item for values in candidates.values() for item in values})
            features = build_feature_frame(chunk, target, candidates)
            predictions = matcher.predict_ids(features) if matcher else baseline_predictions(features, args.threshold)
            all_candidates.update(candidates)
            all_predictions.update(predictions)
            print(f"inference: {min(start + len(chunk), len(test_s1)):,}/{len(test_s1):,}")
    write_results(test_s1["entity_id"], all_candidates, all_predictions, args.output_dir)
    subprocess.run([
        "python", "utils/validate_submission.py", "--matching",
        str(args.output_dir / "matching_results.tsv"), "--candidate",
        str(args.output_dir / "candidate_pairs.tsv"), "--test-dir", str(test_dir),
    ], check=True)


if __name__ == "__main__":
    main()

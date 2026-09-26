"""Run the baseline or XGBoost entity-resolution pipeline.

The default baseline is intentionally runnable without model training. It uses the
same blocking and feature path as the trained matcher, so its validation score is a
meaningful checkpoint before enabling XGBoost.
"""

import argparse
import sqlite3
import subprocess
import sys
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
    frames = []
    for start in range(0, len(ids), 900):
        batch = ids[start:start + 900]
        placeholders = ",".join("?" for _ in batch)
        frames.append(pd.read_sql_query(
            f"SELECT entity_id, business_name, business_address, country, name_norm, name_core, address_norm FROM records WHERE entity_id IN ({placeholders})",
            connection, params=batch,
        ))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def make_candidates(frame: pd.DataFrame, connection: sqlite3.Connection, limit: int) -> Dict[str, List[str]]:
    normalized = frame if {"name_tokens", "address_tokens"}.issubset(frame.columns) else add_normalized_columns(frame)
    return {
        row.entity_id: candidates_for_row(pd.Series(row._asdict()), connection, limit)
        for row in normalized.itertuples(index=False)
    }


def make_features_batched(frame: pd.DataFrame, connection: sqlite3.Connection, limit: int, batch_size: int, label: str) -> tuple[Dict[str, List[str]], pd.DataFrame]:
    candidate_map: Dict[str, List[str]] = {}
    feature_frames = []
    for start in range(0, len(frame), batch_size):
        chunk = frame.iloc[start:start + batch_size].copy()
        normalized_chunk = add_normalized_columns(chunk)
        candidates = make_candidates(normalized_chunk, connection, limit)
        target_ids = {item for values in candidates.values() for item in values}
        target = fetch_records(connection, target_ids)
        feature_frames.append(build_feature_frame(normalized_chunk, target, candidates))
        candidate_map.update(candidates)
        processed = min(start + len(chunk), len(frame))
        print(f"{label}: prepared {processed:,}/{len(frame):,} Source-1 records", flush=True)
    features = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame()
    return candidate_map, features


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


def score_checkpoint(source1: pd.DataFrame, features: pd.DataFrame, truth: Dict[str, List[str]], threshold: float, matcher: PairMatcher | None = None) -> float:
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
    parser.add_argument("--max-candidates", type=int, default=60)
    parser.add_argument("--validation-entities", type=int, default=1000)
    parser.add_argument("--training-entities", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--index-chunk-size", type=int, default=10000)
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
        build_index([train_dir / "train_source2.tsv", train_dir / "train_source3.tsv"], train_db, args.index_chunk_size)
    if not test_db.exists():
        print("building test SQLite index")
        build_index([test_dir / "test_source2.tsv", test_dir / "test_source3.tsv"], test_db, args.index_chunk_size)

    train_split_s1, valid_s1, _, valid_truth_frame = split_by_source1(train_s1, train_truth_frame)
    if len(valid_s1) > args.validation_entities:
        valid_s1 = valid_s1.sample(args.validation_entities, random_state=42)
    valid_truth = truth_map(valid_truth_frame)
    valid_truth = {key: valid_truth.get(key, []) for key in valid_s1["entity_id"]}
    with sqlite3.connect(train_db) as connection:
        connection.execute("PRAGMA cache_size=-65536")
        valid_candidates, valid_features = make_features_batched(
            valid_s1, connection, args.max_candidates, args.batch_size, "validation",
        )
    matcher = None
    if args.mode == "xgboost":
        # Sample only from the training partition so validation Source-1 entities
        # cannot contribute labeled pairs to the fitted matcher.
        train_s1_sample = train_split_s1.sample(min(len(train_split_s1), args.training_entities), random_state=41)
        with sqlite3.connect(train_db) as connection:
            connection.execute("PRAGMA cache_size=-65536")
            _, train_features = make_features_batched(
                train_s1_sample, connection, args.max_candidates, args.batch_size, "training",
            )
        truth_sets = {key: set(values) for key, values in train_truth.items()}
        labels = pd.Series(
            [candidate_id in truth_sets.get(source_id, set()) for source_id, candidate_id in zip(train_features["source1_entity_id"], train_features["candidate_entity_id"])],
            index=train_features.index,
            dtype="int8",
        )
        print(f"fitting XGBoost on {len(train_features):,} candidate pairs", flush=True)
        matcher = PairMatcher(threshold=args.threshold).fit(train_features, labels)
    score_checkpoint(valid_s1, valid_features, valid_truth, args.threshold, matcher)

    for path in (args.output_dir / "matching_results.tsv", args.output_dir / "candidate_pairs.tsv"):
        if path.exists():
            path.unlink()
    with sqlite3.connect(test_db) as connection:
        connection.execute("PRAGMA cache_size=-65536")
        for start in range(0, len(test_s1), args.batch_size):
            chunk = test_s1.iloc[start:start + args.batch_size].copy()
            normalized_chunk = add_normalized_columns(chunk)
            candidates = make_candidates(normalized_chunk, connection, args.max_candidates)
            target = fetch_records(connection, {item for values in candidates.values() for item in values})
            features = build_feature_frame(normalized_chunk, target, candidates)
            predictions = matcher.predict_ids(features) if matcher else baseline_predictions(features, args.threshold)
            write_results(chunk["entity_id"], candidates, predictions, args.output_dir, append=True)
            processed = min(start + len(chunk), len(test_s1))
            if processed % 1000 == 0 or processed == len(test_s1):
                print(f"inference: {processed:,}/{len(test_s1):,} Source-1 records", flush=True)
    subprocess.run([
        sys.executable, "utils/validate_submission.py", "--matching",
        str(args.output_dir / "matching_results.tsv"), "--candidate",
        str(args.output_dir / "candidate_pairs.tsv"), "--test-dir", str(test_dir),
    ], check=True)


if __name__ == "__main__":
    main()

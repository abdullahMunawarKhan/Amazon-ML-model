"""Pairwise matcher. XGBoost is Apache-2.0 licensed; no external data is used."""

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
import xgboost as xgb

FEATURE_COLUMNS = [
    "name_jaccard", "name_core_jaccard", "name_levenshtein", "name_token_sort",
    "name_token_set", "name_sequence", "name_cosine", "name_tfidf_cosine",
    "address_jaccard", "address_levenshtein", "address_token_sort", "address_token_set",
    "address_sequence", "address_cosine", "address_tfidf_cosine", "country_match",
    "name_length_delta", "address_length_delta",
]


class PairMatcher:
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.estimator = None

    def fit(self, features: pd.DataFrame, labels: pd.Series) -> "PairMatcher":
        training_data = xgb.DMatrix(
            features[FEATURE_COLUMNS], label=labels.astype(int),
            feature_names=FEATURE_COLUMNS,
        )
        self.estimator = xgb.train(
            {
                "max_depth": 5,
                "eta": 0.08,
                "subsample": 0.8,
                "colsample_bytree": 0.9,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "device": "cuda",
                "max_bin": 64,
                "nthread": 4,
                "seed": 42,
                "lambda": 2.0,
            },
            training_data,
            num_boost_round=120,
        )
        return self

    def predict_proba(self, features: pd.DataFrame) -> pd.Series:
        if self.estimator is None:
            raise RuntimeError("PairMatcher must be fitted before prediction")
        if features.empty:
            return pd.Series(dtype=float, index=features.index)
        prediction_data = xgb.DMatrix(
            features[FEATURE_COLUMNS], feature_names=FEATURE_COLUMNS,
        )
        return pd.Series(self.estimator.predict(prediction_data), index=features.index)

    def predict_ids(self, features: pd.DataFrame) -> Dict[str, list[str]]:
        if features.empty:
            return {}
        probabilities = self.predict_proba(features)
        selected = features.loc[probabilities >= self.threshold].copy()
        selected["probability"] = probabilities[probabilities >= self.threshold]
        selected = selected.sort_values(["source1_entity_id", "probability", "candidate_entity_id"], ascending=[True, False, True])
        return selected.groupby("source1_entity_id")["candidate_entity_id"].apply(list).to_dict()

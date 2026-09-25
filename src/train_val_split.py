"""Reproducible Source-1-level train/validation splitting."""

from typing import Tuple

import pandas as pd


def split_by_source1(source1: pd.DataFrame, ground_truth: pd.DataFrame, validation_fraction: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ids = source1["entity_id"].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    cutoff = int(len(ids) * (1.0 - validation_fraction))
    train_ids, valid_ids = set(ids.iloc[:cutoff]), set(ids.iloc[cutoff:])
    train_source1 = source1[source1["entity_id"].isin(train_ids)].copy()
    valid_source1 = source1[source1["entity_id"].isin(valid_ids)].copy()
    train_truth = ground_truth[ground_truth["source1_entity_id"].isin(train_ids)].copy()
    valid_truth = ground_truth[ground_truth["source1_entity_id"].isin(valid_ids)].copy()
    return train_source1, valid_source1, train_truth, valid_truth
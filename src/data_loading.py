"""TSV loading and lightweight validation for the challenge data."""

from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

import pandas as pd

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def validate_frame(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    expected = list(columns)
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")
    if frame[expected].isna().any().any():
        raise ValueError(f"{name} contains null values in required columns")
    if frame[expected[0]].duplicated().any():
        raise ValueError(f"{name} contains duplicate {expected[0]} values")


def read_source(path: Path, chunksize: Optional[int] = None) -> pd.DataFrame | Iterator[pd.DataFrame]:
    reader = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunksize)
    if chunksize is not None:
        return reader
    validate_frame(reader, SOURCE_COLUMNS, str(path))
    return reader


def read_ground_truth(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    validate_frame(frame, GROUND_TRUTH_COLUMNS, str(path))
    return frame


def load_split(directory: Path, prefix: str) -> Dict[str, pd.DataFrame]:
    sources = {}
    for source in ("source1", "source2", "source3"):
        sources[source] = read_source(directory / f"{prefix}_{source}.tsv")
    return sources


def summary_stats(sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, frame in sources.items():
        rows.append({
            "source": name,
            "rows": len(frame),
            "countries": frame["country"].nunique(),
            "missing_names": int((frame["business_name"].str.strip() == "").sum()),
            "missing_addresses": int((frame["business_address"].str.strip() == "").sum()),
        })
    return pd.DataFrame(rows)
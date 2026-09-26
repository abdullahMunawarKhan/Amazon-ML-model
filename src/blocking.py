"""Memory-bounded candidate generation using a SQLite inverted token index."""

import sqlite3
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Set, Tuple

import pandas as pd

from .preprocessing import add_normalized_columns


def _tokens(row: pd.Series) -> Set[str]:
    if isinstance(row, pd.Series):
        values = set(row["name_tokens"]) | set(row["address_tokens"])
    else:
        values = set(row.name_tokens) | set(row.address_tokens)
    return {token for token in values if len(token) >= 2}


def build_index(source_paths: Sequence[Path], database_path: Path, chunksize: int = 10_000) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    try:
        connection.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=OFF;
            PRAGMA cache_size=-65536;
            PRAGMA temp_store=FILE;
            CREATE TABLE IF NOT EXISTS records (entity_id TEXT PRIMARY KEY, business_name TEXT, business_address TEXT, country TEXT, name_norm TEXT, name_core TEXT, address_norm TEXT);
            CREATE TABLE IF NOT EXISTS token_index (token TEXT NOT NULL, entity_id TEXT NOT NULL, country TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS token_index_lookup ON token_index(token, country);
        """)
        for path in source_paths:
            for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunksize):
                normalized = add_normalized_columns(chunk)
                records = normalized[["entity_id", "business_name", "business_address", "country", "name_norm", "name_core", "address_norm"]].itertuples(index=False, name=None)
                connection.executemany("INSERT OR REPLACE INTO records VALUES (?, ?, ?, ?, ?, ?, ?)", records)
                token_rows = (
                    (token, row.entity_id, row.country)
                    for row in normalized.itertuples(index=False)
                    for token in _tokens(row)
                )
                connection.executemany("INSERT INTO token_index VALUES (?, ?, ?)", token_rows)
                connection.commit()
                print(f"indexed {path.name}: {chunk.index[-1] + 1:,} rows", flush=True)
        connection.execute("ANALYZE")
    finally:
        connection.close()


def candidates_for_row(row: pd.Series, connection: sqlite3.Connection, max_candidates: int = 250) -> List[str]:
    if "name_tokens" in row.index and "address_tokens" in row.index:
        normalized = row
    else:
        normalized = add_normalized_columns(pd.DataFrame([row])).iloc[0]
    tokens = sorted(_tokens(normalized))
    if not tokens:
        return []
    placeholders = ",".join("?" for _ in tokens)
    query = f"""
        SELECT entity_id, COUNT(*) AS overlap
        FROM token_index
        WHERE country = ? AND token IN ({placeholders})
        GROUP BY entity_id
        ORDER BY overlap DESC, entity_id
        LIMIT ?
    """
    values = [normalized["country"], *tokens, max_candidates]
    return [item[0] for item in connection.execute(query, values).fetchall()]


def generate_candidates(source1: pd.DataFrame, database_path: Path, max_candidates: int = 250) -> Dict[str, List[str]]:
    connection = sqlite3.connect(database_path)
    try:
        return {row.entity_id: candidates_for_row(pd.Series(row._asdict()), connection, max_candidates) for row in source1.itertuples(index=False)}
    finally:
        connection.close()


def recall_ceiling(candidate_map: Dict[str, Iterable[str]], truth: Dict[str, Iterable[str]], target_count: int) -> Tuple[float, float]:
    total = 0
    recovered = 0
    candidate_count = 0
    for source1_id, matches in truth.items():
        match_set = set(matches)
        candidate_set = set(candidate_map.get(source1_id, ()))
        total += len(match_set)
        recovered += len(match_set & candidate_set)
        candidate_count += len(candidate_set)
    recall = recovered / total if total else 1.0
    reduction = 1.0 - candidate_count / max(1, len(candidate_map) * target_count)
    return recall, reduction

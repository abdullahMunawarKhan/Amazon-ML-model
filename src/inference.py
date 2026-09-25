"""Output writing with the exact challenge format."""

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


def write_results(source1_ids: Iterable[str], candidates: Dict[str, Iterable[str]], predictions: Dict[str, Iterable[str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows = []
    result_rows = []
    for source1_id in source1_ids:
        candidate_ids = sorted(set(candidates.get(source1_id, ())))
        match_ids = sorted(set(predictions.get(source1_id, ())) & set(candidate_ids))
        candidate_rows.append({"source1_entity_id": source1_id, "candidate_entity_ids": ",".join(candidate_ids)})
        result_rows.append({"source1_entity_id": source1_id, "matched_entity_ids": ",".join(match_ids)})
    pd.DataFrame(candidate_rows).to_csv(output_dir / "candidate_pairs.tsv", sep="\t", index=False)
    pd.DataFrame(result_rows).to_csv(output_dir / "matching_results.tsv", sep="\t", index=False)
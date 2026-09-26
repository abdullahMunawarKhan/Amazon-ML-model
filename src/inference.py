"""Output writing with the exact challenge format."""

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


def write_results(source1_ids: Iterable[str], candidates: Dict[str, Iterable[str]], predictions: Dict[str, Iterable[str]], output_dir: Path, append: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    source1_ids = list(source1_ids)
    if not source1_ids and not candidates and not predictions:
        for path in (output_dir / "candidate_pairs.tsv", output_dir / "matching_results.tsv"):
            if path.exists():
                path.unlink()
        return

    candidate_rows = []
    result_rows = []
    for source1_id in source1_ids:
        candidate_ids = sorted(set(candidates.get(source1_id, ())))
        match_ids = sorted(set(predictions.get(source1_id, ())) & set(candidate_ids))
        candidate_rows.append({"source1_entity_id": source1_id, "candidate_entity_ids": ",".join(candidate_ids)})
        result_rows.append({"source1_entity_id": source1_id, "matched_entity_ids": ",".join(match_ids)})
    mode = "a" if append else "w"
    header = not append or not (output_dir / "candidate_pairs.tsv").exists()
    pd.DataFrame(candidate_rows, columns=["source1_entity_id", "candidate_entity_ids"]).to_csv(
        output_dir / "candidate_pairs.tsv", sep="\t", index=False, mode=mode, header=header,
    )
    pd.DataFrame(result_rows, columns=["source1_entity_id", "matched_entity_ids"]).to_csv(
        output_dir / "matching_results.tsv", sep="\t", index=False, mode=mode, header=header,
    )

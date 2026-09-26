import tempfile
import unittest
from pathlib import Path

from src.inference import write_results


class WriteResultsTests(unittest.TestCase):
    def test_empty_inputs_do_not_create_stale_header_only_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            write_results([], {}, {}, output_dir)
            self.assertFalse((output_dir / "matching_results.tsv").exists())
            self.assertFalse((output_dir / "candidate_pairs.tsv").exists())

    def test_writes_expected_rows_for_valid_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            write_results(
                ["S1-1", "S1-2"],
                {"S1-1": ["S2-10", "S2-12"], "S1-2": ["S3-20"]},
                {"S1-1": ["S2-10"], "S1-2": []},
                output_dir,
            )
            self.assertTrue((output_dir / "matching_results.tsv").exists())
            self.assertTrue((output_dir / "candidate_pairs.tsv").exists())

            matching = (output_dir / "matching_results.tsv").read_text(encoding="utf-8")
            candidate = (output_dir / "candidate_pairs.tsv").read_text(encoding="utf-8")
            self.assertIn("S1-1\tS2-10", matching)
            self.assertIn("S1-2\t", matching)
            self.assertIn("S1-1\tS2-10,S2-12", candidate)
            self.assertIn("S1-2\tS3-20", candidate)


if __name__ == "__main__":
    unittest.main()

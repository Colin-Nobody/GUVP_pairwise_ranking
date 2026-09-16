from __future__ import annotations

import argparse
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import vlm_pairwise


class EvaluationParsingTests(unittest.TestCase):
    def test_loads_custom_prompt_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prompt_file = Path(directory) / "prompt.txt"
            prompt_file.write_text("custom prompt\n", encoding="utf-8")
            prompt, digest = vlm_pairwise.load_prompt(prompt_file)

        self.assertEqual(prompt, "custom prompt")
        self.assertEqual(
            digest,
            "a81519c3dc469bc8049048b4ed471f6ed87aba13b2effbba7d3f4580344222c9",
        )

    def test_parses_evidence_before_scores(self) -> None:
        response = (
            "Evaluation evidence: A has scattered litter; B has clear pavement.\n"
            "Score of Image A: 4.5\nScore of Image B: 8"
        )
        self.assertEqual(vlm_pairwise.parse_evaluation(response), (4.5, 8.0))

    def test_rejects_scores_before_evidence(self) -> None:
        response = (
            "Score of Image A: 4\nScore of Image B: 8\n"
            "Evaluation evidence: B looks cleaner."
        )
        self.assertIsNone(vlm_pairwise.parse_evaluation(response))

    def test_rejects_missing_evidence_or_invalid_range(self) -> None:
        self.assertIsNone(vlm_pairwise.parse_evaluation(
            "Score of Image A: 4\nScore of Image B: 8"
        ))
        self.assertIsNone(vlm_pairwise.parse_evaluation(
            "Evaluation evidence: test\nScore of Image A: 0\nScore of Image B: 8"
        ))


class CalibrationTests(unittest.TestCase):
    def test_lower_visual_pollution_score_wins(self) -> None:
        result = vlm_pairwise.calibrate_scores(
            "one.jpg", "two.jpg",
            evaluations_ab=[(2, 8)] * 3,
            evaluations_ba=[(8, 2)] * 3,
            expected_samples=3,
            score_direction="lower_is_cleaner",
        )
        self.assertEqual(result["winner"], "one.jpg")
        self.assertEqual(result["status"], "calibrated")
        self.assertAlmostEqual(result["calibrated_score_a"], 2.0)
        self.assertAlmostEqual(result["calibrated_score_b"], 8.0)
        self.assertAlmostEqual(result["calibrated_margin"], 6.0)

    def test_bpc_realigns_swapped_positions_and_averages_2k_scores(self) -> None:
        result = vlm_pairwise.calibrate_scores(
            "one.jpg", "two.jpg",
            evaluations_ab=[(8, 3), (7, 4), (9, 2)],
            # BA position is two.jpg first, one.jpg second.
            evaluations_ba=[(4, 7), (3, 8), (2, 9)],
            expected_samples=3,
        )
        self.assertEqual(result["winner"], "one.jpg")
        self.assertEqual(result["status"], "calibrated")
        self.assertAlmostEqual(result["calibrated_score_a"], 8.0)
        self.assertAlmostEqual(result["calibrated_score_b"], 3.0)
        self.assertEqual(json.loads(result["scores_ba_image_a"]), [7, 8, 9])

    def test_position_bias_cancels_to_tie(self) -> None:
        result = vlm_pairwise.calibrate_scores(
            "one.jpg", "two.jpg",
            evaluations_ab=[(8, 4)] * 3,
            evaluations_ba=[(8, 4)] * 3,
            expected_samples=3,
        )
        self.assertEqual(result["winner"], "")
        self.assertEqual(result["status"], "calibrated_tie")
        self.assertAlmostEqual(result["bpde"], 0.6931471805599453)

    def test_incomplete_mec_direction_is_invalid(self) -> None:
        result = vlm_pairwise.calibrate_scores(
            "one.jpg", "two.jpg", [(8, 3)] * 2, [(3, 8)] * 3, expected_samples=3,
        )
        self.assertEqual(result["status"], "invalid_ab")
        self.assertEqual(result["winner"], "")


class InferenceOptimizationTests(unittest.TestCase):
    @staticmethod
    def response(score_a: int, score_b: int) -> str:
        return (
            "Evaluation evidence: visible evidence.\n"
            f"Score of Image A: {score_a}\nScore of Image B: {score_b}"
        )

    def test_repeat_inputs_preserves_complete_multimodal_blocks(self) -> None:
        inputs = {
            "input_ids": torch.tensor([[1, 2]]),
            "pixel_values": torch.tensor([[10], [20], [30]]),
            "image_grid_thw": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        }
        repeated = vlm_pairwise.repeat_inputs(inputs, 2, torch.device("cpu"))
        self.assertEqual(repeated["input_ids"].tolist(), [[1, 2], [1, 2]])
        self.assertEqual(
            repeated["pixel_values"].flatten().tolist(), [10, 20, 30, 10, 20, 30]
        )
        self.assertEqual(repeated["image_grid_thw"].shape, (4, 3))

    def test_independent_multinomial_sampling_is_reproducible_per_row(self) -> None:
        input_ids = torch.ones((2, 1), dtype=torch.long)
        scores = torch.zeros((2, 256), dtype=torch.float32)
        first = vlm_pairwise.IndependentMultinomialSampler([101, 202], 1.0)(input_ids, scores)
        second = vlm_pairwise.IndependentMultinomialSampler([101, 202], 1.0)(input_ids, scores)
        self.assertTrue(torch.equal(first, second))
        self.assertNotEqual(first[0].argmax().item(), first[1].argmax().item())

    def test_mec_batch_two_plus_one_prepares_once_and_preserves_order(self) -> None:
        args = argparse.Namespace(
            mec_samples=3, mec_batch_size=2, max_pixels=401408, temperature=1.0,
            max_new_tokens=192, seed=42,
        )
        batch_one = [self.response(8, 3), self.response(7, 4)]
        batch_two = [self.response(9, 2)]
        with patch.object(vlm_pairwise, "prepare_pair_inputs", return_value={}) as prepare, \
                patch.object(
                    vlm_pairwise, "generate_prepared", side_effect=[batch_one, batch_two]
                ) as generate:
            evaluations, responses = vlm_pairwise.infer_mec(
                object(), object(), Path("a.jpg"), Path("b.jpg"), args, seed_offset=100,
            )
        prepare.assert_called_once()
        self.assertEqual(responses, batch_one + batch_two)
        self.assertEqual(evaluations, [(8.0, 3.0), (7.0, 4.0), (9.0, 2.0)])
        self.assertEqual(generate.call_args_list[0].args[5], [142, 143])
        self.assertEqual(generate.call_args_list[1].args[5], [144])

    def test_decode_mode_records_batched_sampling_implementation(self) -> None:
        self.assertEqual(
            vlm_pairwise.inference_decode_mode(argparse.Namespace(mec_batch_size=1)),
            vlm_pairwise.DECODE_MODE,
        )
        self.assertIn(
            "batch2", vlm_pairwise.inference_decode_mode(argparse.Namespace(mec_batch_size=2))
        )
        self.assertIn(
            "reused_prefix", vlm_pairwise.inference_decode_mode(
                argparse.Namespace(mec_batch_size=1, reuse_prefix_cache=True)
            )
        )


class OutputAndResumeTests(unittest.TestCase):
    @staticmethod
    def row(comparison_id: str, winner: str, status: str) -> dict[str, str]:
        row = {field: "" for field in vlm_pairwise.AUDIT_FIELDS}
        row.update({
            "comparison_id": comparison_id, "image_a": "one.jpg", "image_b": "two.jpg",
            "winner": winner, "calibrated_score_a": "8", "calibrated_score_b": "3",
            "calibrated_margin": "5", "scores_ab_image_a": "[8, 8, 8]",
            "scores_ab_image_b": "[3, 3, 3]", "scores_ba_image_a": "[8, 8, 8]",
            "scores_ba_image_b": "[3, 3, 3]", "valid_samples_ab": "3",
            "valid_samples_ba": "3", "bpde": "0", "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "prompt_version": vlm_pairwise.PROMPT_VERSION, "prompt_sha256": vlm_pairwise.PROMPT_SHA256,
            "label_layout": vlm_pairwise.LABEL_LAYOUT, "max_pixels": "401408",
            "quantization": vlm_pairwise.QUANTIZATION, "cpu_offload": "false",
            "decode_mode": vlm_pairwise.DECODE_MODE, "mec_samples": "3", "temperature": "1.0",
            "max_new_tokens": "192", "seed": "42", "timestamp_utc": "2026-08-31T00:00:00+00:00",
            "raw_responses_ab": "[]", "raw_responses_ba": "[]", "status": status,
        })
        return row

    def test_vote_file_contains_only_non_tie_calibrated_rows(self) -> None:
        rows = [self.row("vlm-00001", "one.jpg", "calibrated"),
                self.row("vlm-00002", "", "calibrated_tie")]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "votes.csv"
            accepted = vlm_pairwise.rebuild_votes(rows, output)
            with output.open(newline="", encoding="utf-8") as handle:
                written = list(csv.DictReader(handle))
        self.assertEqual(accepted, 1)
        self.assertEqual([row["comparison_id"] for row in written], ["vlm-00001"])

    def test_complete_audit_resume_rebuilds_votes_without_loading_model(self) -> None:
        audit_row = self.row("vlm-00001", "one.jpg", "calibrated")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            (images / "one.jpg").write_bytes(b"placeholder")
            (images / "two.jpg").write_bytes(b"placeholder")
            pairs = root / "pairs.csv"
            with pairs.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["image_a", "image_b"])
                writer.writeheader()
                writer.writerow({"image_a": "one.jpg", "image_b": "two.jpg"})
            audit = root / "audit.csv"
            with audit.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=vlm_pairwise.AUDIT_FIELDS)
                writer.writeheader()
                writer.writerow(audit_row)
            votes = root / "votes.csv"
            argv = [
                "vlm_pairwise.py", "--images-dir", str(images), "--pairs", str(pairs),
                "--audit-output", str(audit), "--output", str(votes),
                "--max-pixels", "401408", "--resume",
            ]
            with patch("sys.argv", argv), patch.object(
                vlm_pairwise, "load_model", side_effect=AssertionError("model must not load")
            ), patch("sys.stdout", new_callable=io.StringIO):
                vlm_pairwise.main()
            with votes.open(newline="", encoding="utf-8") as handle:
                written = list(csv.DictReader(handle))
        self.assertEqual(len(written), 1)
        self.assertEqual(written[0]["winner"], "one.jpg")

    def test_parameter_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "mec-samples"):
            vlm_pairwise.validate_args(argparse.Namespace(
                resume=False, overwrite=False, mec_samples=0, temperature=1.0, max_new_tokens=192,
            ))
        with self.assertRaisesRegex(ValueError, "mec-batch-size"):
            vlm_pairwise.validate_args(argparse.Namespace(
                resume=False, overwrite=False, mec_samples=3, mec_batch_size=4,
                temperature=1.0, max_new_tokens=192,
            ))
        with self.assertRaisesRegex(ValueError, "reuse-prefix-cache"):
            vlm_pairwise.validate_args(argparse.Namespace(
                resume=False, overwrite=False, mec_samples=3, mec_batch_size=2,
                reuse_prefix_cache=True, temperature=1.0, max_new_tokens=192,
            ))


if __name__ == "__main__":
    unittest.main()

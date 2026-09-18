import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from UNIV_adaptor.data_protocol import canonical_sha256, write_json_atomic
from UNIV_adaptor.scripts.data.phase2_analysis import DIMENSIONS
from UNIV_adaptor.scripts.data.score_phase2_dataset import SCORE_SCHEMA, collect
from UNIV_adaptor.scripts.router.phase2_utility_model import (
    choose_actions,
    cross_validate_utility,
    utility_matrix,
)
from UNIV_adaptor.scripts.router.train_phase2_gain_prior import ACTION_SETS, load_data
from UNIV_adaptor.scripts.router.train_phase2_utility_prior import (
    evaluate_policy,
    lambda_diagnostics,
    oracle_rows,
    train,
    train_latency_profile,
)
from UNIV_adaptor.tests.test_phase2_quality import fixture, synthetic_scores


class UtilityModelTests(unittest.TestCase):
    def test_utility_and_tie_break_use_normalized_cost(self):
        quality = np.asarray([[0.8, 0.81, 0.82]])
        cost = np.asarray([0.2, 0.4, 0.6])
        utility = utility_matrix(quality, cost, 0.05)
        np.testing.assert_allclose(utility, [[0.79, 0.79, 0.79]])
        np.testing.assert_array_equal(choose_actions(utility, cost), [0])

    def test_cv_selects_prompt_signal_by_policy_regret(self):
        texts = [
            "fast motion" if index % 2 else "detailed portrait" for index in range(40)
        ]
        utility = np.zeros((40, 3), dtype=np.float64)
        for index in range(40):
            utility[index] = [0.0, 0.12, -0.1] if index % 2 else [0.0, -0.1, 0.12]
        model, state, oof, folds, rows = cross_validate_utility(
            texts, utility, np.asarray([0.2, 0.3, 0.4]), folds=5
        )
        selected = next(row for row in rows if row["selected"])
        self.assertIsNotNone(selected["alpha"])
        self.assertLess(selected["mean_policy_regret"], 0.01)
        self.assertGreater(selected["oracle_exact_action_rate"], 0.95)
        self.assertEqual(oof.shape, utility.shape)
        self.assertEqual(set(folds), set(range(5)))
        self.assertIsNotNone(state)
        self.assertEqual(model["weights"].shape[1], 2)

    def test_constant_utility_selects_mean_control(self):
        utility = np.tile(np.asarray([[0.0, 0.1, -0.1]]), (20, 1))
        _, _, prediction, _, rows = cross_validate_utility(
            [f"prompt {index}" for index in range(20)],
            utility,
            np.asarray([0.2, 0.3, 0.4]),
        )
        selected = next(row for row in rows if row["selected"])
        self.assertIsNone(selected["alpha"])
        np.testing.assert_allclose(prediction, utility)


class UtilityPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        fixture(self.root)
        rows, identity = collect(self.root)
        digest = canonical_sha256({"identity": identity, "rows": rows})
        write_json_atomic(
            self.root / "evaluation_inputs.json",
            dict(input_sha256=digest, **identity, rows=rows),
        )
        body = dict(
            schema=SCORE_SCHEMA,
            input_sha256=digest,
            dimensions=list(DIMENSIONS),
            scores=synthetic_scores(rows),
            provenance={},
        )
        write_json_atomic(
            self.root / "scores.json",
            dict(**body, payload_sha256=canonical_sha256(body)),
        )

    @staticmethod
    def args(out):
        return argparse.Namespace(
            action_set="three",
            out_dir=str(out),
            features="tfidf",
            t5_dir=None,
            folds=2,
            seed=10,
            alphas=[0.1, 1.0],
            max_features=100,
            utility_lambda=0.05,
            harm_epsilon=0.001,
            diagnostic_lambdas=[0.0, 0.05, 0.1],
        )

    def test_profile_is_train_only_paired_and_oracle_has_margin(self):
        samples, rows, _ = load_data(self.root, ACTION_SETS["three"])
        cost, profile = train_latency_profile(rows, ACTION_SETS["three"])
        self.assertEqual(profile["source_split"], "train")
        self.assertEqual(profile["pair_count"], 2)
        self.assertEqual(cost.shape, (3,))
        labels = oracle_rows(samples, ACTION_SETS["three"], cost, 0.05)
        self.assertEqual(len(labels), 4)
        self.assertTrue(all(label["oracle_margin"] >= 0 for label in labels))
        self.assertTrue(
            all(label["oracle_action"] in ACTION_SETS["three"] for label in labels)
        )
        diagnostics = lambda_diagnostics(
            samples, ACTION_SETS["three"], cost, [0.0, 0.05, 0.1], 0.05
        )
        self.assertEqual(len(diagnostics), 3)
        self.assertEqual(sum(row["active_run_lambda"] for row in diagnostics), 1)
        self.assertTrue(all(row["prompts"] == 2 for row in diagnostics))

    def test_prompt_matching_beats_same_histogram_control(self):
        actions = ACTION_SETS["three"]
        samples, _, _ = load_data(self.root, actions)
        selected = copy.deepcopy(samples[:2])
        selected[0]["quality"] = [0.9, 0.1, 0.1]
        selected[1]["quality"] = [0.1, 0.9, 0.1]
        prediction = np.asarray([[0.0, -1.0, -1.0], [0.0, 1.0, -1.0]])
        summary, _ = evaluate_policy(
            selected,
            prediction,
            actions,
            np.asarray([0.2, 0.3, 0.4]),
            0.05,
            0,
            "test",
            42,
            0.001,
        )
        by_name = {row["policy"]: row for row in summary}
        self.assertAlmostEqual(
            by_name["prompt_utility"]["mean_normalized_cost"],
            by_name["shuffled_router_hist_expected"]["mean_normalized_cost"],
        )
        self.assertGreater(by_name["prompt_utility"]["gain_vs_histogram_shuffle"], 0.3)

    def test_validation_changes_do_not_change_model_or_latency_profile(self):
        samples, rows, identity = load_data(self.root, ACTION_SETS["three"])
        train(self.args(self.root / "first"), samples, rows, identity)
        changed = copy.deepcopy(samples)
        changed_rows = copy.deepcopy(rows)
        for sample in changed:
            if sample["split"] == "validation":
                sample["quality"] = [0.99, 0.01, 0.02]
                sample["seconds"] = [999.0, 999.0, 999.0]
        for row in changed_rows:
            if row["split"] == "validation":
                row["pipeline_seconds"] = 999.0
        train(self.args(self.root / "second"), changed, changed_rows, identity)
        with (
            np.load(self.root / "first/model.npz") as first,
            np.load(self.root / "second/model.npz") as second,
        ):
            for key in first.files:
                np.testing.assert_array_equal(first[key], second[key])
        first_profile = json.loads(
            (self.root / "first/latency_profile.json").read_text()
        )
        second_profile = json.loads(
            (self.root / "second/latency_profile.json").read_text()
        )
        self.assertEqual(first_profile, second_profile)
        summary = json.loads((self.root / "first/policy_summary.json").read_text())
        self.assertFalse(summary["factorial_claim"])
        self.assertFalse(summary["test_accessed"])
        for name in (
            "oracle_labels.csv",
            "train_lambda_diagnostics.csv",
            "utility_predictions.csv",
            "policy_summary.csv",
            "policy_by_prompt.csv",
            "report.md",
        ):
            self.assertTrue((self.root / "first" / name).is_file())


if __name__ == "__main__":
    unittest.main()

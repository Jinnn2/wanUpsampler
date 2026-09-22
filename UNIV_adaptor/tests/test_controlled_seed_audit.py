"""Scientific checks: stable signal versus winner selection from seed noise."""
import csv
import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from UNIV_adaptor.data_protocol import canonical_sha256
from UNIV_adaptor.scripts.data.audit_controlled_factor_seed_value import (
    ACTIONS, SEEDS, load_cube, oracle_values, pair_diagnostics, run,
)


class SeedValueTest(unittest.TestCase):
    def test_stable_prompt_preference_survives_holdout(self):
        u = np.array([[[0., 1.]]*3, [[1., 0.]]*3])
        values, _, _, _ = oracle_values(u, 0)
        self.assertAlmostEqual(values['cross_seed_selector'].mean(), 1.)
        np.testing.assert_array_equal(values['cross_seed_selector'], values['prompt_oracle_in_sample'])

    def test_seed_winners_are_not_prompt_value(self):
        u = np.array([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]])
        values, _, _, choices = oracle_values(u, 0)
        self.assertAlmostEqual(values['instance_oracle_in_sample'].mean(), 1.)
        self.assertAlmostEqual(values['prompt_oracle_in_sample'].mean(), 1/3)
        self.assertAlmostEqual(values['cross_seed_selector'].mean(), 0.)
        self.assertTrue(np.all(choices != np.array([[0, 1, 2]])))

    def test_pair_difference_cancels_reference(self):
        a = np.array([[1., 2., 3.], [3., 2., 1.]])
        b = a + np.array([[1.], [-1.]])
        ref = np.array([[0., 10., -8.], [8., 7., 2.]])
        np.testing.assert_allclose((a-ref)-(b-ref), a-b)
        self.assertEqual(pair_diagnostics(a-b)['icc_unclipped'], 1.)

    def test_loader_skips_test_scores_and_rejects_duplicates(self):
        rows = []
        for p, split in enumerate(('train', 'validation')):
            for seed in SEEDS:
                for action in ACTIONS:
                    rows.append(dict(prompt_id=p, split=split, family_id=split,
                                     prompt=split, prompt_sha256=canonical_sha256(split),
                                     base_seed=seed, seed=seed+p, action_id=action,
                                     full_vbench5=.8, action_vbench5=.7, delta_vbench5=-.1,
                                     full_seconds=10., action_seconds=5., time_ratio_to_full=.5))
        rows.append({**rows[0], 'split': 'test', 'full_vbench5': 'DO_NOT_PARSE'})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'relative.csv'
            def save():
                with path.open('w', newline='', encoding='utf8') as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(rows)
            save()
            meta, q, _ = load_cube(path)
            self.assertEqual(len(meta), 2)
            self.assertEqual(q.shape, (2, 3, 4))
            path.rename(Path(directory)/'relative_to_full.csv')
            out = Path(directory)/'audit'
            run(argparse.Namespace(scored_dir=directory, out_dir=str(out),
                                   lambdas=[0., .05], epsilon=.001, bootstrap=100, seed=42))
            report = json.loads((out/'audit.json').read_text(encoding='utf8'))
            self.assertFalse(report['test_scores_analyzed'])
            self.assertEqual(len(report['oracle_summary']), 20)
            rows.append(rows[0])
            save()
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                load_cube(path)


if __name__ == '__main__':
    unittest.main()

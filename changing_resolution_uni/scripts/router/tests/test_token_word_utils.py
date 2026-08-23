from __future__ import annotations

import unittest

import numpy as np

from changing_resolution_uni.scripts.router.token_word_utils import (
    merge_subtokens_to_words,
    summarize_attributions,
)


class TokenWordUtilsTest(unittest.TestCase):
    def test_sentencepiece_subtokens_become_natural_words(self) -> None:
        tokens = ["▁A", "▁vibr", "ant", "▁scene", ",", "</s>"]
        scores = np.asarray([1.0, 2.0, 4.0, -1.0, 99.0, 99.0])
        words = merge_subtokens_to_words(tokens, scores)
        self.assertEqual([word["word"] for word in words], ["vibrant", "scene"])
        self.assertEqual(words[0]["subtokens"], ["vibr", "ant"])
        self.assertEqual(words[0]["mean_piece_attribution"], 3.0)
        self.assertAlmostEqual(words[0]["additive_contribution"], 1.0)

    def test_wordpiece_markers_are_merged(self) -> None:
        tokens = ["play", "##ing", "▁outside"]
        scores = np.asarray([1.0, 3.0, -2.0])
        words = merge_subtokens_to_words(tokens, scores)
        self.assertEqual([word["word"] for word in words], ["playing", "outside"])

    def test_summary_applies_occurrence_threshold(self) -> None:
        values = {
            "camera": [
                {
                    "mean_piece_attribution": 1.0,
                    "additive_contribution": 0.1,
                    "subtoken_count": 1.0,
                },
                {
                    "mean_piece_attribution": 3.0,
                    "additive_contribution": 0.3,
                    "subtoken_count": 2.0,
                },
            ],
            "rare": [
                {
                    "mean_piece_attribution": 5.0,
                    "additive_contribution": 0.5,
                    "subtoken_count": 1.0,
                }
            ],
        }
        rows = summarize_attributions(values, minimum_count=2)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["word"], "camera")
        self.assertEqual(rows[0]["mean_attribution"], 2.0)
        self.assertEqual(rows[0]["mean_subtokens"], 1.5)


if __name__ == "__main__":
    unittest.main()

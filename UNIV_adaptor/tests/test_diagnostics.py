from __future__ import annotations

import unittest

import numpy as np

from UNIV_adaptor.diagnostics import (
    spectral_metrics,
    state_distance,
    state_moments,
    temporal_difference_metrics,
    transition_state_diagnostics,
)


class TransitionDiagnosticsTest(unittest.TestCase):
    def test_moments_and_temporal_difference(self):
        time = np.arange(5, dtype=np.float32)[None, :, None, None]
        state = np.broadcast_to(time, (2, 5, 3, 4)).copy()
        moments = state_moments(state)
        difference = temporal_difference_metrics(state)
        self.assertEqual(moments["shape"], [2, 5, 3, 4])
        self.assertAlmostEqual(moments["mean"], 2.0)
        self.assertAlmostEqual(moments["std"], np.sqrt(2.0))
        self.assertAlmostEqual(difference["mean_abs"], 1.0)
        self.assertAlmostEqual(difference["rms"], 1.0)

    def test_temporal_spectrum_detects_nyquist_signal(self):
        alternating = ((-1.0) ** np.arange(8))[None, :, None, None]
        state = np.broadcast_to(alternating, (1, 8, 4, 4)).copy()
        temporal = spectral_metrics(state)["temporal"]
        self.assertAlmostEqual(temporal["centroid_nyquist"], 1.0)
        self.assertAlmostEqual(temporal["high_frequency_ratio"], 1.0)

    def test_spatial_spectrum_detects_checkerboard(self):
        height, width = np.indices((8, 8))
        checkerboard = ((-1.0) ** (height + width))[None, None]
        state = np.broadcast_to(checkerboard, (1, 3, 8, 8)).copy()
        spatial = spectral_metrics(state)["spatial"]
        self.assertAlmostEqual(spatial["centroid_nyquist"], 1.0)
        self.assertAlmostEqual(spatial["high_frequency_ratio"], 1.0)

    def test_orthonormal_spectral_mean_power_matches_signal_energy(self):
        rng = np.random.default_rng(8)
        state = rng.normal(size=(2, 5, 6, 7)).astype(np.float32)
        channel_mean = state.mean(axis=0)
        expected = float(np.mean(np.square(channel_mean)))
        spectrum = spectral_metrics(state)
        self.assertAlmostEqual(spectrum["temporal"]["mean_power"], expected)
        self.assertAlmostEqual(spectrum["spatial"]["mean_power"], expected)

    def test_native_distance_is_zero_for_identical_states(self):
        state = np.arange(48, dtype=np.float32).reshape(1, 3, 4, 4)
        distance = state_distance(state, state.copy())
        self.assertEqual(distance["rmse"], 0.0)
        self.assertEqual(distance["mae"], 0.0)
        self.assertAlmostEqual(distance["relative_l2"], 0.0)
        self.assertAlmostEqual(distance["cosine_distance"], 0.0)

    def test_missing_native_state_is_explicit(self):
        low = np.zeros((1, 2, 2, 2), dtype=np.float32)
        high = np.zeros((1, 3, 4, 4), dtype=np.float32)
        diagnostics = transition_state_diagnostics(
            clean_lr=low,
            clean_hr=high,
            renoised_hr=high,
        )
        self.assertFalse(diagnostics["native_hr_state_distance"]["available"])
        self.assertEqual(
            diagnostics["native_hr_state_distance"]["reason"],
            "native_hr_state_not_provided",
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np

from UNIV_adaptor.flow import wan_clean_from_velocity, wan_renoise


class WanFlowAnalyticTest(unittest.TestCase):
    def test_clean_endpoint_identity_for_arbitrary_sigma(self):
        rng = np.random.default_rng(1234)
        clean = rng.normal(size=(4, 5, 7)).astype(np.float64)
        noise = rng.normal(size=clean.shape).astype(np.float64)
        velocity = noise - clean
        for sigma in (0.0, 0.01, 0.25, 0.6, 0.999, 1.0):
            with self.subTest(sigma=sigma):
                noisy = wan_renoise(clean, noise, sigma)
                recovered = wan_clean_from_velocity(noisy, velocity, sigma)
                np.testing.assert_allclose(recovered, clean, rtol=1e-12, atol=1e-12)

    def test_renoise_mean_and_variance_match_independent_mixture(self):
        rng = np.random.default_rng(2026)
        sample_count = 1_000_000
        clean_mean, clean_std = 1.25, 0.7
        noise_mean, noise_std = -0.2, 1.1
        sigma = 0.65
        clean = rng.normal(clean_mean, clean_std, sample_count)
        noise = rng.normal(noise_mean, noise_std, sample_count)

        renoised = wan_renoise(clean, noise, sigma)
        expected_mean = (1.0 - sigma) * clean_mean + sigma * noise_mean
        expected_variance = (
            (1.0 - sigma) ** 2 * clean_std**2 + sigma**2 * noise_std**2
        )

        self.assertAlmostEqual(float(renoised.mean()), expected_mean, delta=0.004)
        self.assertAlmostEqual(float(renoised.var()), expected_variance, delta=0.006)

    def test_renoise_endpoints_are_exact(self):
        clean = np.asarray([-2.0, 0.5, 3.0])
        noise = np.asarray([4.0, -1.0, 2.0])
        np.testing.assert_array_equal(wan_renoise(clean, noise, 0.0), clean)
        np.testing.assert_array_equal(wan_renoise(clean, noise, 1.0), noise)


if __name__ == "__main__":
    unittest.main()

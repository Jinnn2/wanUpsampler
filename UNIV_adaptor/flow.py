from __future__ import annotations


def wan_clean_from_velocity(noisy_state, velocity, sigma):
    """Recover Wan's clean endpoint from ``x_sigma = z0 + sigma * v``."""

    return noisy_state - sigma * velocity


def wan_renoise(clean_state, noise, sigma):
    """Move a clean state onto Wan's rectified-flow interpolation path.

    Wan uses ``v = epsilon - z0`` and therefore
    ``x_sigma = z0 + sigma*v = (1-sigma)*z0 + sigma*epsilon``.
    """

    return (1.0 - sigma) * clean_state + sigma * noise

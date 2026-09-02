"""Universal spatio-temporal Wan generation pipeline.

The package keeps its planning utilities importable without Torch or LightX2V.
Runtime integration is loaded explicitly from :mod:`UNIV_adaptor.wan_runner`.
"""

from .core import ResolvedSchedule, UniversalAction
from .schedule import resolve_schedule

__all__ = ["ResolvedSchedule", "UniversalAction", "resolve_schedule"]

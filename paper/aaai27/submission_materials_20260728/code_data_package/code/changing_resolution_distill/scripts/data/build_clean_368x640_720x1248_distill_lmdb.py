"""Construct Distill4 ITU clean-latent supervision, paper Sec. 3.2."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from changing_resolution.scripts.data.build_480p720p_lmdb import main


if __name__ == "__main__":
    main()

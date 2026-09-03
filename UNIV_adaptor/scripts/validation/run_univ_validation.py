from __future__ import annotations

import sys
from pathlib import Path


repo_root = str(Path(__file__).resolve().parents[3])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from UNIV_adaptor.validation import main


if __name__ == "__main__":
    main()

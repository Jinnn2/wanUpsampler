#!/usr/bin/env python3
"""Create the canonical Figure 2 from the selected ImageGen overview.

The complete inference band is already reproduced in Figure 1(b).  Figure 2
therefore retains only the model-internal supervision band and removes the
redundant footer claim.  This is a deterministic crop: labels, arrows, and
diagram pixels are not redrawn or generatively modified.
"""

from pathlib import Path

from PIL import Image


FIGURE_DIR = Path(__file__).resolve().parent
SOURCE = FIGURE_DIR / "fig_talh_overview_imagegen.png"
OUTPUT_PNG = FIGURE_DIR / "fig_talh_overview.png"
OUTPUT_PDF = FIGURE_DIR / "fig_talh_overview.pdf"
# Locked coordinates in the selected 1774 x 887 ImageGen rendering.  Rows
# 424--774 contain the complete bordered supervision panel.
CROP_BOX = (0, 423, 1774, 776)


def main() -> None:
    source = Image.open(SOURCE).convert("RGB")
    if source.size != (1774, 887):
        raise ValueError(f"Unexpected selected overview size: {source.size}")

    supervision = source.crop(CROP_BOX)
    supervision.save(OUTPUT_PNG, dpi=(400, 400), optimize=True)
    supervision.save(OUTPUT_PDF, "PDF", resolution=400.0)

    print(f"Saved: {OUTPUT_PNG}")
    print(f"Saved: {OUTPUT_PDF}")
    print(f"Crop box: {CROP_BOX}; output size: {supervision.size}")


if __name__ == "__main__":
    main()

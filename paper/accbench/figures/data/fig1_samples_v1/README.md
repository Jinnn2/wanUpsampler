# Figure 1a samples

Rule (fixed before rendering): dev2 prompts, seed 42; within each (arm, cell) the clips are sorted by min(retention, quality_kept) (ties: prompt id) and the lower median (index (n-1)//2) is taken. No hand picking.

Frames: 5 evenly spaced (first and last included); the arm's frames are resized to FULL's size. Flow: OpenCV Farneback, mean magnitude in pixels at 416 px width, one value per consecutive frame pair (not the RAFT flow of the fixed layer).

## RCM4_hh: prompt 127 (rank 5 of 11)

Hundreds of bees swarm rapidly over a detailed honeycomb in front of a fixed macro camera, their veined wings blurring as they dart across hexagonal cells and glistening honey, constantly crossing and occluding one another.

retention 0.500, quality_kept 0.200, min 0.200, dVBench5 -5.00 pp

| rank | prompt id | min(ret, Q) | retention | quality_kept | dVBench5 pp |
| --- | --- | --- | --- | --- | --- |
| 0 | 107 | 0.000 | 0.222 | 0.000 | -1.34 |
| 1 | 111 | 0.000 | 0.000 | 0.600 | +8.85 |
| 2 | 123 | 0.000 | 0.833 | 0.000 | -3.90 |
| 3 | 147 | 0.000 | 1.000 | 0.000 | +0.69 |
| 4 | 119 | 0.167 | 0.167 | 0.200 | -5.31 |
| 5 (chosen) | 127 | 0.200 | 0.500 | 0.200 | -5.00 |
| 6 | 115 | 0.400 | 0.400 | 1.000 | +13.34 |
| 7 | 143 | 0.400 | 0.800 | 0.400 | -4.13 |
| 8 | 163 | 0.400 | 0.714 | 0.400 | -3.25 |
| 9 | 155 | 0.500 | 0.500 | 1.000 | +0.35 |
| 10 | 135 | 0.800 | 0.857 | 0.800 | -0.97 |

## RCM4_hl: prompt 134 (rank 7 of 16)

A single large plain white drum bounces and rolls rapidly across an empty dark stage, wobbling and changing direction. Nothing else is present and the camera stays locked.

retention 0.000, quality_kept 0.000, min 0.000, dVBench5 +1.50 pp

| rank | prompt id | min(ret, Q) | retention | quality_kept | dVBench5 pp |
| --- | --- | --- | --- | --- | --- |
| 0 | 102 | 0.000 | 0.000 | 0.000 | +9.62 |
| 1 | 106 | 0.000 | 0.500 | 0.000 | -2.80 |
| 2 | 110 | 0.000 | 0.125 | 0.000 | +10.60 |
| 3 | 114 | 0.000 | 0.000 | 0.000 | -3.55 |
| 4 | 118 | 0.000 | 0.500 | 0.000 | -5.73 |
| 5 | 122 | 0.000 | 0.200 | 0.000 | -2.63 |
| 6 | 130 | 0.000 | 0.500 | 0.000 | -4.44 |
| 7 (chosen) | 134 | 0.000 | 0.000 | 0.000 | +1.50 |
| 8 | 138 | 0.000 | 0.500 | 0.000 | +3.98 |
| 9 | 142 | 0.000 | 0.417 | 0.000 | +8.07 |
| 10 | 146 | 0.000 | 0.625 | 0.000 | -0.24 |
| 11 | 150 | 0.000 | 0.375 | 0.000 | -5.13 |
| 12 | 154 | 0.000 | 0.000 | 1.000 | +7.54 |
| 13 | 158 | 0.000 | 0.000 | 0.000 | -0.30 |
| 14 | 162 | 0.000 | 0.562 | 0.000 | -3.77 |
| 15 | 126 | 0.500 | 0.500 | 1.000 | +8.39 |

## RCM4_lh: prompt 105 (rank 7 of 16)

A fixed overhead camera shows a carved wooden chessboard mid-game with all pieces in place: intricate knights with fine manes, crenellated rooks, tiny crosses on the kings, and visible wood grain on every square. Everything remains perfectly still.

retention 0.300, quality_kept 0.600, min 0.300, dVBench5 +0.04 pp

| rank | prompt id | min(ret, Q) | retention | quality_kept | dVBench5 pp |
| --- | --- | --- | --- | --- | --- |
| 0 | 125 | 0.000 | 0.167 | 0.000 | -3.76 |
| 1 | 133 | 0.000 | 0.000 | 0.400 | +0.29 |
| 2 | 113 | 0.167 | 0.167 | 0.400 | -3.92 |
| 3 | 137 | 0.167 | 0.167 | 0.200 | -5.39 |
| 4 | 117 | 0.200 | 0.500 | 0.200 | -2.34 |
| 5 | 121 | 0.200 | 0.600 | 0.200 | -0.38 |
| 6 | 129 | 0.200 | 0.750 | 0.200 | -1.93 |
| 7 (chosen) | 105 | 0.300 | 0.300 | 0.600 | +0.04 |
| 8 | 145 | 0.333 | 0.333 | 0.400 | -2.69 |
| 9 | 109 | 0.400 | 0.619 | 0.400 | -1.35 |
| 10 | 141 | 0.417 | 0.417 | 0.600 | -5.74 |
| 11 | 101 | 0.500 | 0.500 | 0.800 | -7.19 |
| 12 | 149 | 0.500 | 0.500 | 0.800 | -3.26 |
| 13 | 153 | 0.500 | 0.500 | 0.600 | -0.40 |
| 14 | 161 | 0.500 | 0.500 | 0.800 | -4.97 |
| 15 | 157 | 0.667 | 0.667 | 1.000 | -1.76 |

## RCM4_ll: prompt 112 (rank 7 of 16)

One large smooth orange maple leaf lies flat and motionless on a plain gray concrete floor. The camera is locked, the leaf has broad simple color without visible veins, and the empty background never changes.

retention 0.571, quality_kept 0.200, min 0.200, dVBench5 -2.16 pp

| rank | prompt id | min(ret, Q) | retention | quality_kept | dVBench5 pp |
| --- | --- | --- | --- | --- | --- |
| 0 | 100 | 0.000 | 0.333 | 0.000 | +1.19 |
| 1 | 108 | 0.000 | 0.312 | 0.000 | +0.02 |
| 2 | 120 | 0.000 | 0.500 | 0.000 | -2.91 |
| 3 | 128 | 0.000 | 0.611 | 0.000 | -7.87 |
| 4 | 144 | 0.000 | 0.714 | 0.000 | +1.55 |
| 5 | 148 | 0.000 | 0.700 | 0.000 | +4.99 |
| 6 | 124 | 0.188 | 0.188 | 1.000 | -2.34 |
| 7 (chosen) | 112 | 0.200 | 0.571 | 0.200 | -2.16 |
| 8 | 104 | 0.214 | 0.214 | 0.800 | -2.55 |
| 9 | 152 | 0.476 | 0.476 | 0.600 | +3.17 |
| 10 | 116 | 0.500 | 0.500 | 0.600 | -2.74 |
| 11 | 136 | 0.500 | 0.500 | 0.800 | +3.75 |
| 12 | 140 | 0.500 | 0.500 | 1.000 | -1.49 |
| 13 | 132 | 0.800 | 0.833 | 0.800 | +2.64 |
| 14 | 160 | 0.929 | 0.929 | 1.000 | -3.21 |
| 15 | 156 | 1.000 | 1.000 | 1.000 | -1.40 |

## LR_FLASH_hh: prompt 163 (rank 5 of 11)

Hundreds of old books with gilded spines tumble rapidly off tall library shelves, pages fluttering open, books colliding and occluding one another as they fall past brass rails and ornate wood carvings. Fixed camera.

retention 0.571, quality_kept 0.000, min 0.000, dVBench5 -1.62 pp

| rank | prompt id | min(ret, Q) | retention | quality_kept | dVBench5 pp |
| --- | --- | --- | --- | --- | --- |
| 0 | 115 | 0.000 | 0.000 | 0.200 | +1.88 |
| 1 | 127 | 0.000 | 0.500 | 0.000 | -2.33 |
| 2 | 135 | 0.000 | 0.714 | 0.000 | -2.14 |
| 3 | 143 | 0.000 | 0.000 | 0.000 | -9.88 |
| 4 | 147 | 0.000 | 0.625 | 0.000 | +1.16 |
| 5 (chosen) | 163 | 0.000 | 0.571 | 0.000 | -1.62 |
| 6 | 107 | 0.200 | 0.222 | 0.200 | +2.93 |
| 7 | 111 | 0.200 | 1.000 | 0.200 | +15.29 |
| 8 | 119 | 0.200 | 0.667 | 0.200 | -0.98 |
| 9 | 155 | 0.200 | 0.200 | 0.800 | -1.24 |
| 10 | 123 | 0.333 | 0.333 | 0.400 | +4.65 |

## LR_FLASH_hl: prompt 114 (rank 7 of 16)

One large smooth orange maple leaf is blown violently across a plain gray concrete floor, flipping and skidding quickly from side to side. Nothing else is in the frame and the camera does not move.

retention 1.000, quality_kept 0.600, min 0.600, dVBench5 +2.03 pp

| rank | prompt id | min(ret, Q) | retention | quality_kept | dVBench5 pp |
| --- | --- | --- | --- | --- | --- |
| 0 | 102 | 0.000 | 1.000 | 0.000 | +5.81 |
| 1 | 122 | 0.000 | 0.600 | 0.000 | +1.44 |
| 2 | 134 | 0.000 | 1.000 | 0.000 | -5.36 |
| 3 | 138 | 0.000 | 0.500 | 0.000 | +10.10 |
| 4 | 150 | 0.000 | 0.875 | 0.000 | -0.35 |
| 5 | 158 | 0.000 | 0.143 | 0.000 | -3.41 |
| 6 | 162 | 0.000 | 0.625 | 0.000 | -1.86 |
| 7 (chosen) | 114 | 0.600 | 1.000 | 0.600 | +2.03 |
| 8 | 118 | 0.600 | 1.000 | 0.600 | +1.99 |
| 9 | 126 | 0.600 | 1.000 | 0.600 | +7.67 |
| 10 | 130 | 0.600 | 1.000 | 0.600 | -1.36 |
| 11 | 146 | 0.600 | 0.625 | 0.600 | +2.28 |
| 12 | 142 | 0.667 | 0.667 | 1.000 | +11.76 |
| 13 | 154 | 0.800 | 1.000 | 0.800 | +2.13 |
| 14 | 106 | 0.875 | 0.875 | 1.000 | -0.07 |
| 15 | 110 | 0.875 | 0.875 | 1.000 | +4.26 |

## LR_FLASH_lh: prompt 129 (rank 7 of 16)

A locked wide shot shows dozens of hot air balloons with intricate patchwork patterns, thin ropes, wicker baskets, and tiny figures, parked above a field of fine grass. All balloons are tethered and completely still.

retention 0.375, quality_kept 0.200, min 0.200, dVBench5 -0.60 pp

| rank | prompt id | min(ret, Q) | retention | quality_kept | dVBench5 pp |
| --- | --- | --- | --- | --- | --- |
| 0 | 113 | 0.000 | 0.333 | 0.000 | -0.75 |
| 1 | 149 | 0.000 | 0.667 | 0.000 | -3.10 |
| 2 | 161 | 0.000 | 0.786 | 0.000 | -1.21 |
| 3 | 141 | 0.167 | 0.167 | 0.200 | -1.26 |
| 4 | 101 | 0.200 | 1.000 | 0.200 | -3.66 |
| 5 | 105 | 0.200 | 0.700 | 0.200 | +1.84 |
| 6 | 125 | 0.200 | 0.500 | 0.200 | -2.80 |
| 7 (chosen) | 129 | 0.200 | 0.375 | 0.200 | -0.60 |
| 8 | 145 | 0.200 | 0.750 | 0.200 | -3.69 |
| 9 | 153 | 0.200 | 0.500 | 0.200 | -3.27 |
| 10 | 157 | 0.200 | 0.333 | 0.200 | -0.74 |
| 11 | 117 | 0.333 | 0.333 | 0.800 | -0.54 |
| 12 | 121 | 0.400 | 0.400 | 0.800 | +0.54 |
| 13 | 133 | 0.600 | 1.000 | 0.600 | -1.08 |
| 14 | 137 | 0.750 | 0.750 | 1.000 | -1.43 |
| 15 | 109 | 0.800 | 0.905 | 0.800 | -5.13 |

## LR_FLASH_ll: prompt 124 (rank 7 of 16)

A single large smooth golden wax block rests motionless on an empty beige table. The locked camera shows broad plain surfaces with no cells, no bees, and no change in lighting.

retention 0.875, quality_kept 0.800, min 0.800, dVBench5 -0.47 pp

| rank | prompt id | min(ret, Q) | retention | quality_kept | dVBench5 pp |
| --- | --- | --- | --- | --- | --- |
| 0 | 140 | 0.000 | 1.000 | 0.000 | -5.23 |
| 1 | 120 | 0.200 | 1.000 | 0.200 | -0.94 |
| 2 | 136 | 0.500 | 0.500 | 0.800 | +2.38 |
| 3 | 100 | 0.600 | 1.000 | 0.600 | -0.37 |
| 4 | 116 | 0.600 | 0.875 | 0.600 | +1.38 |
| 5 | 108 | 0.800 | 0.875 | 0.800 | +2.90 |
| 6 | 112 | 0.800 | 1.000 | 0.800 | +0.42 |
| 7 (chosen) | 124 | 0.800 | 0.875 | 0.800 | -0.47 |
| 8 | 148 | 0.800 | 0.800 | 1.000 | +3.34 |
| 9 | 152 | 0.810 | 0.810 | 1.000 | +4.57 |
| 10 | 128 | 0.861 | 0.861 | 1.000 | -0.69 |
| 11 | 104 | 0.929 | 0.929 | 1.000 | -1.26 |
| 12 | 132 | 0.958 | 0.958 | 1.000 | +1.59 |
| 13 | 144 | 1.000 | 1.000 | 1.000 | +1.13 |
| 14 | 156 | 1.000 | 1.000 | 1.000 | -1.49 |
| 15 | 160 | 1.000 | 1.000 | 1.000 | +0.21 |


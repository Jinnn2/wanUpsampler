# Figure 1 samples v2

Rule (fixed before rendering): dev2 prompts, seed 42; within each cell the prompts are sorted by the mean over RCM4 and LR_FLASH of min(retention, quality_kept) (ties: prompt id) and the lower median (index (n-1)//2) is taken. No hand picking. Each chosen prompt is exported for FULL, RCM4, LR_FLASH and the tau=0.80 dev-fit routed arm (hh->RCM4, hl->SCALING10, lh->F41_RIFE, ll->TEA015; one copy when it is already listed).

Frames: 5 evenly spaced (first and last included), resized to FULL's size. Flow: OpenCV Farneback, mean magnitude in pixels at 416 px width, one value per consecutive frame pair (not the RAFT flow of the fixed layer). FULL rows have no retention / quality_kept (FULL is the reference).

## hh: prompt 119 (rank 5 of 11), routed arm RCM4

An elaborate tower of hundreds of tiny grained wooden blocks collapses rapidly in front of a fixed camera, blocks with small painted windows and thin arches tumbling, bouncing, and occluding one another as they scatter.

| video | retention | quality_kept | min(ret, Q) | dVBench5 pp |
| --- | --- | --- | --- | --- |
| RCM4 (routed) | 0.167 | 0.200 | 0.167 | -5.31 |
| LR_FLASH | 0.667 | 0.200 | 0.200 | -0.98 |

| rank | prompt id | mean min | min RCM4 | min LR_FLASH |
| --- | --- | --- | --- | --- |
| 0 | 147 | 0.000 | 0.000 | 0.000 |
| 1 | 107 | 0.100 | 0.000 | 0.200 |
| 2 | 111 | 0.100 | 0.000 | 0.200 |
| 3 | 127 | 0.100 | 0.200 | 0.000 |
| 4 | 123 | 0.167 | 0.000 | 0.333 |
| 5 (chosen) | 119 | 0.183 | 0.167 | 0.200 |
| 6 | 115 | 0.200 | 0.400 | 0.000 |
| 7 | 143 | 0.200 | 0.400 | 0.000 |
| 8 | 163 | 0.200 | 0.400 | 0.000 |
| 9 | 155 | 0.350 | 0.500 | 0.200 |
| 10 | 135 | 0.400 | 0.800 | 0.000 |

## hl: prompt 114 (rank 7 of 16), routed arm SCALING10

One large smooth orange maple leaf is blown violently across a plain gray concrete floor, flipping and skidding quickly from side to side. Nothing else is in the frame and the camera does not move.

| video | retention | quality_kept | min(ret, Q) | dVBench5 pp |
| --- | --- | --- | --- | --- |
| RCM4 | 0.000 | 0.000 | 0.000 | -3.55 |
| LR_FLASH | 1.000 | 0.600 | 0.600 | +2.03 |
| SCALING10 (routed) | 1.000 | 1.000 | 1.000 | -0.55 |

| rank | prompt id | mean min | min RCM4 | min LR_FLASH |
| --- | --- | --- | --- | --- |
| 0 | 102 | 0.000 | 0.000 | 0.000 |
| 1 | 122 | 0.000 | 0.000 | 0.000 |
| 2 | 134 | 0.000 | 0.000 | 0.000 |
| 3 | 138 | 0.000 | 0.000 | 0.000 |
| 4 | 150 | 0.000 | 0.000 | 0.000 |
| 5 | 158 | 0.000 | 0.000 | 0.000 |
| 6 | 162 | 0.000 | 0.000 | 0.000 |
| 7 (chosen) | 114 | 0.300 | 0.000 | 0.600 |
| 8 | 118 | 0.300 | 0.000 | 0.600 |
| 9 | 130 | 0.300 | 0.000 | 0.600 |
| 10 | 146 | 0.300 | 0.000 | 0.600 |
| 11 | 142 | 0.333 | 0.000 | 0.667 |
| 12 | 154 | 0.400 | 0.000 | 0.800 |
| 13 | 106 | 0.438 | 0.000 | 0.875 |
| 14 | 110 | 0.438 | 0.000 | 0.875 |
| 15 | 126 | 0.550 | 0.500 | 0.600 |

## lh: prompt 145 (rank 7 of 16), routed arm F41_RIFE

A locked overhead camera shows the playfield of a vintage pinball machine with intricate illustrated artwork, tiny bumpers, thin metal rails, small star rollovers, and dense rows of inserts. The machine is off and nothing moves.

| video | retention | quality_kept | min(ret, Q) | dVBench5 pp |
| --- | --- | --- | --- | --- |
| RCM4 | 0.333 | 0.400 | 0.333 | -2.69 |
| LR_FLASH | 0.750 | 0.200 | 0.200 | -3.69 |
| F41_RIFE (routed) | 1.000 | 1.000 | 1.000 | +0.61 |

| rank | prompt id | mean min | min RCM4 | min LR_FLASH |
| --- | --- | --- | --- | --- |
| 0 | 113 | 0.083 | 0.167 | 0.000 |
| 1 | 125 | 0.100 | 0.000 | 0.200 |
| 2 | 129 | 0.200 | 0.200 | 0.200 |
| 3 | 105 | 0.250 | 0.300 | 0.200 |
| 4 | 149 | 0.250 | 0.500 | 0.000 |
| 5 | 161 | 0.250 | 0.500 | 0.000 |
| 6 | 117 | 0.267 | 0.200 | 0.333 |
| 7 (chosen) | 145 | 0.267 | 0.333 | 0.200 |
| 8 | 141 | 0.292 | 0.417 | 0.167 |
| 9 | 133 | 0.300 | 0.000 | 0.600 |
| 10 | 121 | 0.300 | 0.200 | 0.400 |
| 11 | 101 | 0.350 | 0.500 | 0.200 |
| 12 | 153 | 0.350 | 0.500 | 0.200 |
| 13 | 157 | 0.433 | 0.667 | 0.200 |
| 14 | 137 | 0.458 | 0.167 | 0.750 |
| 15 | 109 | 0.600 | 0.400 | 0.800 |

## ll: prompt 112 (rank 7 of 16), routed arm TEA015

One large smooth orange maple leaf lies flat and motionless on a plain gray concrete floor. The camera is locked, the leaf has broad simple color without visible veins, and the empty background never changes.

| video | retention | quality_kept | min(ret, Q) | dVBench5 pp |
| --- | --- | --- | --- | --- |
| RCM4 | 0.571 | 0.200 | 0.200 | -2.16 |
| LR_FLASH | 1.000 | 0.800 | 0.800 | +0.42 |
| TEA015 (routed) | 1.000 | 1.000 | 1.000 | +0.67 |

| rank | prompt id | mean min | min RCM4 | min LR_FLASH |
| --- | --- | --- | --- | --- |
| 0 | 120 | 0.100 | 0.000 | 0.200 |
| 1 | 140 | 0.250 | 0.500 | 0.000 |
| 2 | 100 | 0.300 | 0.000 | 0.600 |
| 3 | 108 | 0.400 | 0.000 | 0.800 |
| 4 | 148 | 0.400 | 0.000 | 0.800 |
| 5 | 128 | 0.431 | 0.000 | 0.861 |
| 6 | 124 | 0.494 | 0.188 | 0.800 |
| 7 (chosen) | 112 | 0.500 | 0.200 | 0.800 |
| 8 | 136 | 0.500 | 0.500 | 0.500 |
| 9 | 144 | 0.500 | 0.000 | 1.000 |
| 10 | 116 | 0.550 | 0.500 | 0.600 |
| 11 | 104 | 0.571 | 0.214 | 0.929 |
| 12 | 152 | 0.643 | 0.476 | 0.810 |
| 13 | 132 | 0.879 | 0.800 | 0.958 |
| 14 | 160 | 0.964 | 0.929 | 1.000 |
| 15 | 156 | 1.000 | 1.000 | 1.000 |


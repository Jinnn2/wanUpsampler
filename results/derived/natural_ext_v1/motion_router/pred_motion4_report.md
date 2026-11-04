# Router from prompt vectors vs hand-made cells

250 prompts, 8-fold by family x 10 repeats. Each feature set predicts per-prompt (time, retention, quality_kept) for every arm in-fold; one LP over the held-out fold's predictions picks per-prompt arm distributions (internal floor on a grid). Speed is compared at matched realized quality: the fastest policy (or two-policy mix) whose out-of-fold min(mean ret, mean Q) reaches tau, so a predictor that overpromises gets no credit. 95% CI: family bootstrap.

## speed at realized quality tau, gain vs blind

| features | tau 0.95 | tau 0.90 | tau 0.85 | tau 0.80 | tau 0.70 | tau 0.60 |
| --- | --- | --- | --- | --- | --- | --- |
| blind | 1.62x +0% [+0%, +0%] | 2.15x +0% [+0%, +0%] | 2.68x +0% [+0%, +0%] | 3.56x +0% [+0%, +0%] | 7.34x +0% [+0%, +0%] | 7.34x +0% [+0%, +0%] |
| cell | 1.64x +1% [-2%, +6%] | 2.18x +2% [-1%, +5%] | 2.81x +5% [-1%, +8%] | 3.70x +4% [-4%, +11%] | 7.34x +0% [-0%, +3%] | 7.34x +0% [+0%, +0%] |
| bow | 1.64x +1% [-1%, +5%] | 2.21x +3% [-0%, +5%] | 2.76x +3% [-2%, +8%] | 3.62x +1% [-6%, +8%] | 7.34x +0% [-2%, +1%] | 7.34x +0% [+0%, +0%] |

## speed at realized quality tau, gain vs cell

| features | tau 0.95 | tau 0.90 | tau 0.85 | tau 0.80 | tau 0.70 | tau 0.60 |
| --- | --- | --- | --- | --- | --- | --- |
| blind | 1.62x -1% [-5%, +2%] | 2.15x -2% [-5%, +1%] | 2.68x -4% [-7%, +1%] | 3.56x -4% [-10%, +4%] | 7.34x +0% [-3%, +0%] | 7.34x +0% [+0%, +0%] |
| cell | 1.64x +0% [+0%, +0%] | 2.18x +0% [+0%, +0%] | 2.81x +0% [+0%, +0%] | 3.70x +0% [+0%, +0%] | 7.34x +0% [+0%, +0%] | 7.34x +0% [+0%, +0%] |
| bow | 1.64x +0% [-2%, +2%] | 2.21x +1% [-2%, +3%] | 2.76x -2% [-4%, +4%] | 3.62x -2% [-8%, +4%] | 7.34x +0% [-3%, +0%] | 7.34x +0% [+0%, +0%] |

## nominal: internal floor = tau (speed, realized min(ret, Q) - tau)

| features | tau 0.95 | tau 0.90 | tau 0.85 | tau 0.80 | tau 0.70 | tau 0.60 |
| --- | --- | --- | --- | --- | --- | --- |
| blind | 1.84x s-0.012 | 2.16x s-0.003 | 2.84x s-0.013 | 3.57x s-0.001 | 7.30x s+0.004 | 7.34x s+0.104 |
| cell | 1.85x s-0.016 | 2.25x s-0.008 | 3.02x s-0.016 | 3.78x s-0.003 | 7.25x s+0.005 | 7.34x s+0.104 |
| bow | 1.84x s-0.013 | 2.23x s-0.003 | 2.99x s-0.016 | 3.73x s-0.005 | 7.07x s+0.007 | 7.34x s+0.104 |

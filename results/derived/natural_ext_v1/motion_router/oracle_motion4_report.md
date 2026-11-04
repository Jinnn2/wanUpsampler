# Router from prompt vectors vs hand-made cells

250 prompts, 8-fold by family x 10 repeats. Each feature set predicts per-prompt (time, retention, quality_kept) for every arm in-fold; one LP over the held-out fold's predictions picks per-prompt arm distributions (internal floor on a grid). Speed is compared at matched realized quality: the fastest policy (or two-policy mix) whose out-of-fold min(mean ret, mean Q) reaches tau, so a predictor that overpromises gets no credit. 95% CI: family bootstrap.

## speed at realized quality tau, gain vs blind

| features | tau 0.95 | tau 0.90 | tau 0.85 | tau 0.80 | tau 0.70 | tau 0.60 |
| --- | --- | --- | --- | --- | --- | --- |
| blind | 1.62x +0% [+0%, +0%] | 2.15x +0% [+0%, +0%] | 2.68x +0% [+0%, +0%] | 3.56x +0% [+0%, +0%] | 7.34x +0% [+0%, +0%] | 7.34x +0% [+0%, +0%] |
| cell | 1.62x +0% [-5%, +7%] | 2.29x +7% [+3%, +10%] | 2.93x +9% [+3%, +13%] | 3.93x +10% [+1%, +18%] | 7.34x +0% [+0%, +4%] | 7.34x +0% [+0%, +0%] |
| bow | 1.53x -6% [-9%, +0%] | 2.21x +3% [-0%, +5%] | 2.78x +3% [-2%, +7%] | 3.66x +3% [-5%, +8%] | 7.34x +0% [-1%, +2%] | 7.34x +0% [+0%, +0%] |

## speed at realized quality tau, gain vs cell

| features | tau 0.95 | tau 0.90 | tau 0.85 | tau 0.80 | tau 0.70 | tau 0.60 |
| --- | --- | --- | --- | --- | --- | --- |
| blind | 1.62x -0% [-7%, +5%] | 2.15x -6% [-9%, -3%] | 2.68x -8% [-12%, -3%] | 3.56x -9% [-15%, -1%] | 7.34x +0% [-4%, +0%] | 7.34x +0% [+0%, +0%] |
| cell | 1.62x +0% [+0%, +0%] | 2.29x +0% [+0%, +0%] | 2.93x +0% [+0%, +0%] | 3.93x +0% [+0%, +0%] | 7.34x +0% [+0%, +0%] | 7.34x +0% [+0%, +0%] |
| bow | 1.53x -6% [-10%, -1%] | 2.21x -3% [-6%, -1%] | 2.78x -5% [-9%, -2%] | 3.66x -7% [-12%, -2%] | 7.34x +0% [-3%, +0%] | 7.34x +0% [+0%, +0%] |

## nominal: internal floor = tau (speed, realized min(ret, Q) - tau)

| features | tau 0.95 | tau 0.90 | tau 0.85 | tau 0.80 | tau 0.70 | tau 0.60 |
| --- | --- | --- | --- | --- | --- | --- |
| blind | 1.84x s-0.012 | 2.16x s-0.003 | 2.84x s-0.013 | 3.57x s-0.001 | 7.30x s+0.004 | 7.34x s+0.104 |
| cell | 1.89x s-0.012 | 2.32x s-0.002 | 3.13x s-0.012 | 3.96x s-0.001 | 7.23x s+0.005 | 7.34x s+0.104 |
| bow | 1.88x s-0.017 | 2.30x s-0.010 | 3.07x s-0.020 | 3.85x s-0.008 | 6.94x s+0.010 | 7.34x s+0.104 |

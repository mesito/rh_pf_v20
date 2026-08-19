# Part A — full-ensemble floor fits (all tight pairs)

Data: `floors_all_8p4e9.npy` (2736 pairs, floor = max|Z_rs2| over the gap, paper convention of the floor convention of Section 52 eq. floor-def, 33-pt grid + parabolic refine), `LX_all_8p4e9.npy` (L_X at 6 cutoffs, one cumsum pass over 664,579 primes), `floors_all_2M.npy` (6088 pairs, fp64 RS, max and min122 conventions), `LX_all_2M.npy`.

Convention check: on the 300-pair subset, max-convention floors are ~29.4x the archived min-122-sample floors (constant factor, absorbed in c); refit at X=1e4 gives a=2.006/1.978, b=0.772/0.783, R2=0.9442/0.9447 — no material difference in a, b, R2 (all scale-invariant).

## Table 52.1 (full): 8.4e9, 2736 pairs vs 300-pair values

| X | a (full) | se(a) | b (full) | se(b) | R2 (full) | a/b/R2 (300) | move? |
|---|---|---|---|---|---|---|---|
| 1e2 | 1.9917 | 0.0137 | 0.7493 | 0.0053 | 0.9369 | 1.847/0.752/0.9339 | **YES** |
| 1e3 | 1.9979 | 0.0133 | 0.7385 | 0.0050 | 0.9410 | 1.978/0.733/0.9348 | no |
| 1e4 | 1.9925 | 0.0134 | 0.7776 | 0.0053 | 0.9396 | 1.978/0.783/0.9447 | no |
| 1e5 | 1.9647 | 0.0131 | 0.8276 | 0.0055 | 0.9421 | 1.969/0.829/0.9505 | no |
| 1e6 | 1.9595 | 0.0128 | 0.8945 | 0.0058 | 0.9447 | 1.932/0.899/0.9439 | no |
| 1e7 | 1.9372 | 0.0121 | 0.9271 | 0.0056 | 0.9510 | 1.868/0.944/0.9546 | **YES** |

## Table 52.2(a) (full): 2M, 6088 pairs, X=1e4

| convention | a | se(a) | b | se(b) | R2 | 300-pair a/b/R2 |
|---|---|---|---|---|---|---|
| max | 1.9488 | 0.0070 | 0.8935 | 0.0047 | 0.9489 | 1.892/0.906/0.9463 |
| min122 | 1.9266 | 0.0072 | 0.9032 | 0.0048 | 0.9454 | 1.892/0.906/0.9463 |

## Full-set statistics (8.4e9, X=1e4 unless stated)

| statistic | full (2736) | 300-pair value |
|---|---|---|
| R2(g only) | 0.4673 | 0.4556 |
| R2(g+far) | 0.7474 | 0.7615 |
| R2(g+L_X) | 0.9396 | 0.9447 |
| R2(g+far+L_X) | 0.9402 | 0.9447 |
| corr(resid, L_X) | 0.9414 | 0.946 |
| partial corr(resid, far | L_X) | 0.0267 | 0.004 |
| partial corr(resid, L_X | far) | 0.9196 | 0.873 |
| K5 corr(resid, log far) | -0.7251 | -0.746 |
| corr(L_X, log far) | -0.7930 | -0.641..-0.791 |

Euler deficit at Lehmer midpoints vs 300 controls (seed 11, s in [0.95,1.05]):

| X | shift (sigma) full | 300-pair | ratio P_X smaller full | 300-pair |
|---|---|---|---|---|
| 1e2 | -0.48 | -0.54 | 1.5x | - |
| 1e3 | -0.87 | -0.96 | 2.1x | - |
| 1e4 | -1.44 | -1.54 | 3.0x | - |
| 1e5 | -2.17 | -2.28 | 4.6x | - |
| 1e6 | -3.14 | -3.13 | 7.1x | 7.4x |
| 1e7 | -3.89 | -3.90 | 11.4x | 12x |

## Full-set statistics (2M, 6088 pairs, X=1e4, max convention)

| statistic | full | 300-pair |
|---|---|---|
| corr(L_X, resid) | 0.9262 | 0.916 |
| corr(L_X, log far) | -0.4262 | -0.422 |
| K5 corr(resid, log far) | -0.4983 | -0.50 |

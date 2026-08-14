# Moment-level mediation: does the Euler–Hadamard dependence survive averaging?

**Ensemble.** All consecutive-zero gaps of the LMFDB 8.4e9 window (772,718 gaps),
subsampled to every 4th gap (offset 3, seed `default_rng(20260808)`): **193,179
gaps**. For each gap: midpoint `t_i`, gap `g_i`, floor = max|Z(t)| over 12 interior
samples (`Z_rs2`, re-validated against mpmath dps=50 to ~5e-9 after replacing the
float64-sourced π in the phase reduction with a 20-digit longdouble π — see note
below), and `L_X(t_i)` at X=10⁶ (k=1,2,3 terms; spot-checked vs the canonical
`LX_batch` to ~1e-8). Data: `data/moment_allgaps.npy` (gap_idx, t_mid, g, floor, L_X).

**Test.** Gaps sorted by t; blocks of B consecutive gaps; block means of log floor,
log g, L_X; OLS `log floor = a·log g + b·L_X + c` per B
(`moment_mediation_scan.csv`).

## Results

| B | n_blocks | a | b | se_b | R² | partial corr(log floor, L_X \| log g) |
|---|---|---|---|---|---|---|
| 1 | 193,179 | 0.6535 | 0.6900 | 0.0010 | 0.9209 | 0.8391 |
| 16 | 12,073 | 1.5836 | 0.2143 | 0.0040 | 0.9542 | 0.4383 |
| 256 | 754 | 1.7636 | 0.1485 | 0.0130 | 0.9817 | 0.3837 |
| 4096 | 47 | 1.8615 | 0.0744 | 0.0517 | 0.9762 | 0.2120 |

Auxiliary diagnostics:

| B | b/se_b | p(b) | std. beta(b) | sd(L_X block) | raw corr(log floor, L_X) | corr(L_X, log g) blocks |
|---|---|---|---|---|---|---|
| 1 | 678 | ≈0 | 0.752 | 1.406 | 0.950 | 0.817 |
| 16 | 54 | ≈0 | 0.240 | 0.277 | 0.920 | 0.901 |
| 256 | 11.4 | 8e-28 | 0.169 | 0.0867 | 0.951 | 0.943 |
| 4096 | 1.44 | 0.16 | 0.084 | 0.0153 | 0.920 | 0.918 |

**B=1 sanity (task item 5).** The all-gaps pointwise fit (a=0.654, b=0.690,
R²=0.9209) differs from the paper's 300-tight-pair fit (a=1.932, b=0.899,
R²=0.9439) — honestly expected: the g² law is a *small-gap asymptotic*; for a
generic gap the hump height is O(1) and only weakly g-dependent, so the ensemble
log-log slope flattens to 0.65. Restricting the same B=1 data to tight gaps
recovers the paper fit: s<0.15 (n=680): **a=1.909, b=0.878, R²=0.9427**; s<0.30
(n=5,005): a=1.878, b=0.888, R²=0.9479. The instrument and functional are
consistent with the paper's; the difference is the ensemble, not the method.

## Verdict: the conditional Euler coefficient DECAYS under averaging — but the coupling survives in the spacing channel

The multivariate Euler coefficient decays monotonically, b(B) = 0.690 → 0.214 →
0.149 → 0.074, and by B=4096 (blocks of ≈1,230 t-units, n=47) it is
statistically indistinguishable from zero (1.4σ, p=0.16); the partial correlation
falls 0.84 → 0.21 over the same range. Read narrowly, this is the Heap-splitting
expectation: the Euler and zero factors decorrelate in averaged moments. Two
quantitative caveats qualify the picture. (i) The decay is slow: at B=256
(≈77 t-units) the partial correlation is still 0.38 with t=11.4 — the pointwise
coupling persists robustly over mesoscopic windows and is not a tight-pair or
B=1 artifact. (ii) The decay of b is mechanically driven by self-averaging of
L_X (block sd 1.41 → 0.015) plus near-perfect block-level collinearity between
mean L_X and mean log g (corr 0.82 → 0.94): the Euler information does not
vanish, it migrates into the zero-spacing channel — a(B) rises 0.65 → 1.86
toward the small-gap asymptotic slope 2, and the raw block correlation
corr(log floor, L_X) stays ≈0.92–0.95 at every B. In the language of the
splitting conjecture: the independence of Euler and zero factors holds, if at
all, only after conditioning on the local zero-density mode; unconditionally the
Euler–zero correlation (block-mean L_X vs block-mean spacing ≈ 0.92 even at
B=4096) does not decay with averaging, so the moment-level independence assumed
by the hybrid model is at best an asymptotic, conditional approximation.

## Instrument note (flag for verify_all.py)

The canonical `Z_rs2`/`LX_batch` in `verify_all.py` reduce phases modulo
`2·np.longdouble(np.pi)`, i.e. a float64-sourced π; at t≈8.4e9 the quotient
k≈1.4·10¹⁰ amplifies the π rounding to phase errors up to ~6e-6 rad, giving
|Z_rs2 − mpmath(dps=50)| ≈ 2–8e-6 at random mid-gap points (T9's quoted 1.5e-7
happens to be at a low-error point). With a 20-digit longdouble π the error
drops to ~5e-9. The impact on the published regressions is negligible
(L_X errors ~1e-5 vs σ≈0.6–1.4), but future floor work at tighter gaps should
use the corrected constant (implemented in `scan_floors.py`, `compute_lx.py`).

## Provenance

- `scan_floors.py` — floors, ~2×49 min on 2 cores, checkpoints in
  `data/moment_floors_half{0,1}.npy`
- `compute_lx.py` — L_X(10⁶) at 193,179 midpoints (~35 min)
- `analyze_moments.py` — blocking test, writes `data/moment_mediation_scan.csv`

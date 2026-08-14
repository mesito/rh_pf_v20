# Speiser-detector + depth-capacity sweep — LMFDB ensemble at t ~ 8.436e9

**Ensemble:** 772,719 certified Platt zeros, t in [8,436,146,000.02, 8,436,376,999.73],
dens = log(mean(t)/2pi)/(2pi) = 3.3451071 (evaluated at mean zero height).
**Tight-pair criterion:** consecutive pair with normalized spacing s = g*dens < 0.15.
**Window convention:** W = 250 zeros on each side of the pair, indices [i-250, i) and
[i+2, i+252), clipped at the ensemble edges (2 pairs clipped at the top edge) — identical
to verify_all.py T8. All ordinate arithmetic in 80-bit longdouble.

## Detector (Lemma D transposed to the gap interior)
For t in (gamma_n, gamma_{n+1}),
  G(t) = sum_{m outside pair, window} 1/(t - gamma_m)^2  +  2/((t-gamma_n)(t-gamma_{n+1})).
The first term is the positive on-line quadratic form; the straddling pair enters with the
negative Speiser sign (equals -8/g^2 at the midpoint), the transpose of the
exceptional-orbit term 2/(h'^2 - h0^2) of Lemma D. G sampled at 257 interior points per
pair; a "Speiser sign change" = G attaining a non-negative value.

## Headline A — Speiser sign changes
- Tight pairs scanned: **2736**
- Pairs exhibiting a sign change: **0**   (total sign changes: 0)
- Largest G value seen anywhere: -3934.5 (pair idx 171536, the widest gap g=0.04478);
  i.e. G(t) <= -3.9e3 on every gap interior. The certificate margin is enormous:
  the far-field sum (O(10^1)-O(10^2) on the gap) can never overcome the pair term
  >= 8/g^2 >= 3.98e3.

## Headline B — depth capacity H = sqrt(2/far) vs half-gap
- H/(g/2): min **3.4857**, median 16.0055, mean 18.6113, max 142.0029
- Interpretation: even the worst pair admits a hidden companion at depth 3.49 half-gaps
  before the Speiser threshold would be visible; the median pair admits 16.0.

### Worst 5 pairs by H/(g/2)
| pair_idx | t0 | g | far | S_on | H | r | H/(g/2) |
|---|---|---|---|---|---|---|---|
| 655666 | 8436342007.330303 | 0.042279 | 3.683543e+02 | 4843.7925 | 0.0737 | 0.961225 | 3.4857 |
| 655665 | 8436342007.293220 | 0.031887 | 3.081449e+02 | 8176.0875 | 0.0806 | 0.980975 | 5.0530 |
| 500176 | 8436295524.700836 | 0.037208 | 1.353791e+02 | 5914.0318 | 0.1215 | 0.988488 | 6.5334 |
| 771913 | 8436376758.773354 | 0.034958 | 1.508036e+02 | 6697.1604 | 0.1152 | 0.988677 | 6.5886 |
| 154759 | 8436192264.499482 | 0.039591 | 1.153875e+02 | 5219.2698 | 0.1317 | 0.988884 | 6.6508 |

## Theorem-A ratio r = h_thr/(g/2) = sqrt(2/S_on)/(g/2), full ensemble
- min 0.961225, median 0.998054, mean 0.997909, max 0.999975  (paper: median 0.9981)
- Theorem A(a) identity r^2(1 + 1/(8 C_mid)) = 1: max deviation 2.71e-19 (80-bit; pure identity)
- Endpoint identity E = 1/(t0-g_n)^2 + 1/(t0-g_{n+1})^2 = 8/g^2: exact (max rel dev 0.0)

## far / S_on statistics and validation vs paper values
| quantity | this run | paper/stored | status |
|---|---|---|---|
| # tight pairs | 2736 | 2736 | match |
| far median | 24.842 | 24.84 | match |
| far mean | 27.98 | 27.98 | match |
| far max | 368.35 | 368.35 | match |
| mean far / (2.5 dens^2) | 1.0003 | 1.000 | match |
| median r | 0.998054 | 0.9981 | match |
| max rel dev far vs lehmer_rows_8p4e9 | 1.7e-05 | — | stored rows in fp64; this run in 80-bit (more accurate) |
| max rel dev S_on vs stored | 1.3e-06 | — | match |
| max abs dev r vs stored | 6.4e-07 | — | match |

Shield cross-check (T5 of verify_all): mean prev/next gap of tight pairs = 1.354/1.360 of
the global mean gap — a *different* statistic from H/(g/2) (the shield is a gap-ratio; H is
an absolute depth scale ~ 0.28 ~ mean gap, while g/2 <= 0.0224 at tight pairs, so
H/(g/2) >= 3.49 necessarily).

**Headline claim:** across all 2,736 tight pairs at height 8.436e9 the Speiser detector
shows zero sign changes (G < -3.9e3 everywhere on every gap interior), and every pair has
depth capacity H >= 3.49 half-gaps (median 16.0) with h_thr/(g/2) in [0.9612, 0.99997]
(median 0.9981): no hidden companion anywhere near a tight pair can evade detection, and
none is present.

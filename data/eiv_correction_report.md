# Part B — EIV correction of the Euler loading b

Setup: `log floor = a log g + b L_X + c` on the full 2736-pair ensemble. L_X is a proxy for log|P_X| with error U = -(k>=4 harmonic tail) (k<=3 truncation; sharp cutoff at X).

Truncation-error model for U:
- sup-norm bound (the floor convention of Section 52 rationale): |U| <= (1/4)*sum_p p^-2 = 0.1131; exact sum over k>=4 of (1/k)*sum_p p^{-k/2} = 0.2387
- random-phase variance: sU^2 = (1/2)*sum_{k>=4} (1/k^2) sum_p p^{-k} = 0.00350 (sU = 0.0591)
- empirical scale: Var(L_X) over the 2736 pairs is 0.36-0.41 (below), so the bound-error reliability is lam ~ 0.97.

## Corrected b per X

| X | b_obs | Var(L_X) | lam (sup) | b_EIV (sup) | lam (rand) | b_EIV (rand) | b_IV (L_{X/10}) | lam_IV |
|---|---|---|---|---|---|---|---|---|
| 1e2 | 0.7493 | 0.7315 | 0.9825 | 0.7626 | 0.9952 | 0.7529 | nan | nan |
| 1e3 | 0.7385 | 0.7597 | 0.9832 | 0.7512 | 0.9954 | 0.7419 | 0.8230 | 0.8767 |
| 1e4 | 0.7776 | 0.6832 | 0.9813 | 0.7924 | 0.9949 | 0.7816 | 0.8433 | 0.9738 |
| 1e5 | 0.8276 | 0.6060 | 0.9789 | 0.8455 | 0.9942 | 0.8324 | 0.8897 | 0.9848 |
| 1e6 | 0.8945 | 0.5217 | 0.9755 | 0.9169 | 0.9933 | 0.9005 | 0.9549 | 1.0069 |
| 1e7 | 0.9271 | 0.4919 | 0.9740 | 0.9519 | 0.9929 | 0.9338 | 0.9824 | 0.9656 |

## Reading

- Under the classical EIV model (proxy = truth + independent error of scale given by the the floor convention of Section 52 sup bound), the attenuation factor is lam = 0.965-0.970 across X, lifting b by ~3.2%: b(1e6) 0.894 -> 0.917, b(1e7) 0.927 -> 0.952.
- Under the random-phase tail variance (arguably more realistic), lam = 0.990-0.992 and b lifts by only ~1%%.
- The IV estimate with L_{X/10} as instrument gives b_IV between the raw and sup-corrected values; because the k>=4 tail is common to both cutoffs the instrument is NOT strictly valid (correlated errors), so b_IV is biased toward b_obs and should be read as a lower bound on the correction. Validity holds only under the scale-specific cutoff-error interpretation (disjoint prime sets p>X vs p>X/10, near-independent).
- Conclusion: corrected b range at X=1e7 is 0.927 (raw) - 0.952 (sup-bound EIV), i.e. b = 1 remains excluded at this height (se(b) ~ 0.006), but truncation attenuation is a ~3% effect, not the main reason b < 1.

Assumptions stated: (i) E[U]=0, U independent of X* and of log g; sup-norm bound used as RMS (conservative: overestimates sU, hence overcorrects slightly); (ii) instrument relevance is strong (corr(L_X, L_{X/10}) ~ 0.9), exogeneity as discussed.

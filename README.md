# Verification suite for *Off-line zeros of the Riemann ξ-function* (version 20)

Code and data accompanying

> M. Ismail, *Off-line zeros of the Riemann ξ-function: a constraint network,
> an exactly solved four-body collision, an energy–budget divergence, a
> lifetime–deficit dictionary with negative controls, and an explicit bound for
> S(T) from validated inputs*, 10 August 2026. `Ismail_rh_pf_v20.tex` / `.pdf`.

Every theorem whose statement contains an explicit constant, each numbered
Numerical Observation, all 32 tables and all 7 figures of the paper
are recomputed here, independently of the text, with pass/fail criteria stated
in the scripts.

## 1. Requirements and use

Python ≥ 3.10 with `numpy`, `scipy`, `sympy`, `mpmath` (1.3.0). No network
access is required; all input data are included.

```
python3 run_all.py            # the suite for the paper        (~40 min)
python3 run_all.py --fast     # fast certificates only         (~6 min)
python3 run_all.py --all      # adds the supplementary suite   (~3 min)
```

This archive contains code and data only. The manuscript source and its
figures are distributed separately as `Ismail_rh_pf_v20_latex.zip`
(`Ismail_rh_pf_v20.tex` plus `figs/`); compile it with `pdflatex` run twice.

## 2. Contents of the package

| Path | Covers | Scripts |
|---|---|---|
| `parts_I_VI/` | Sections 4–32 | 25 (+ `config.py`, `ic_core.py`) |
| `part_VII/` | Sections 33–40 | 6 |
| `part_VIII/` | Sections 41–47 | 3 |
| `part_IX/` | Sections 48–57 | 10 |
| `data/` | certified zero ensembles (36 files, 51 MB) | — |
| `SHA256SUMS.txt` | checksums for every file above | — |

Each of `parts_I_VI/`, `part_VII/` and `part_VIII/` is self-contained: the
required zero caches (`zeta_zeros.npy`, `zeros_cache.pkl`, `dh_online_true.pkl`,
`_cache_*.npy`) ship with them. `part_IX/` and `parts_I_VI/speiser_threshold_verify.py`
read `data/` through the environment variable `RH_DATA` (default: the sibling
`data/` directory, i.e. `../data` relative to the script); the latter uses
`data/zeros_window_7005_592.npy` (zeros 6414–7005 near t ≈ 7005) and rebuilds it
from `data/zeros6.gz` if absent.

`SHA256SUMS.txt` records the **shipped state**. The pipeline scripts of
`part_IX/` write derived arrays back into `data/`, so after any pipeline run the
affected checksums no longer match; regenerate from the repository root with
`find . -type f ! -name 'SHA256SUMS.txt' ! -name '*.pyc' -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS.txt`.

## 3. Sections 4–32 — `parts_I_VI/`

One row per section. A dash in the *Table* column means the section states no
tabulated quantity.

| § | Verified objects | Table | Script |
|---|---|---|---|
| 4 | Thm 4.1 (fundamental positivity of V′ off the exceptional set) | — | `verify_04_positivity.py`, `verify_23_floor_core.py` |
| 5 | Thm 5.1, Cor 5.2 (pole confinement; the ε-brackets) | — | `verify_23_floor_core.py` |
| 6 | Prop 6.4 (sign barrier); NO 6.6 (Mh₀ ∈ [0.37, 8.68] over all 1999 gaps) | — | `verify_12_tunneling.py` |
| 7 | Thms 7.1, 7.2, 7.4 (measure collapse, monotonicity, antisymmetry) | — | `verify_partI_constraints.py` |
| 8 | Thm 8.1, NO 8.2 (Hadamard curvature detector; S/N > 10⁴) | — | `verify_08_curvature.py` |
| 9 | Thm 9.1 (\|Φ_off\| = 1 on the critical line, to machine precision) | — | `verify_09_inner_function.py` |
| 10 | Thm 10.1 (Speiser threshold: monotonicity and the IVT branch); NO 10.2 (square-root law, C̄ = 1.18 ± 0.08) | 10.1 | `speiser_threshold_verify.py`, `verify_10_speiser.py`, `verify_10_self_consistency.py` |
| 11 | Prop 11.3, NO 11.4 (concavity; exactly one turning point per gap, certified) | — | `verify_11_concavity.py` |
| 12 | Thm 12.1, Cor 12.2 (T-independent tunneling invariant; C_n ∈ [1.002, 2.306]) | — | `verify_12_tunneling.py` |
| 13 | Thm 13.2, Lemma 13.3; NO 13.5–13.6 (η = 0.458 ± 0.013, CV 2.7 %) | 13.1 | `verify_13_collision_eta.py`, `verify_paper_numerics.py` |
| 14 | Lemma 14.1, Thm 14.2, Cor 14.3 (invariants; closed-form collision time) | — | `verify_14_15_flow_bounds.py` |
| 15 | Lemma 15.1, (15.2)–(15.5) (argument bound; τ*(T₀) = 0.2147, x₁* = 2.7588, B_S = 11.61); Thm 15.6 | 15.1 | `verify_kusmin_correction.py`, `verify_optimized_bound.py`, `verify_14_15_flow_bounds.py` |
| 16 | Problem 16.1 — resolved in Section 36 | — | `part_VII/04_counterexample_16_1.py` |
| 17 | Defs 17.1–17.2, Lemma 17.3 (weight classes; regularized-gamma values) | 17.1 | `verify_v20_corrections.py` |
| 18 | Lemma 18.2 (Poisson capacity budget, averaged asymptotics) | — | `verify_19_energy_budget.py` |
| 19 | Thm 19.2 (E/B divergence, averaged and pointwise); NO 19.3 | 19.1, 19.2 | `verify_19_energy_budget.py` |
| 20 | Lemma 20.2, Prop 20.3; NO 20.4 (first-moment sign via the recursion (20.2)) | 20.1 | `verify_paper_numerics.py` |
| 21 | Prop 21.3; NO 21.2 (cap 0.977, factor 3.0×, sliver 0.453) | 21.1 | `verify_21_supply_envelope.py` |
| 22 | Def 22.1 (windowed Weil functional and the deficit) | — | `verify_22_24_dictionary.py` |
| 23 | Lemmas 23.1, 23.2′, 23.3, 23.5; NO 23.2 (u_c map, κ = 0.700, spill bound, sparse floor ≥ y²/6) | — | `verify_v20_corrections.py`, `verify_23_floor_core.py` |
| 24 | Thm 24.1, Prop 24.3, Cor 24.5, Rem 24.1′ (cluster counterexample 0.571 vs 0.130); NO 24.2, 24.4 | 24.1 | `verify_22_24_dictionary.py`, `verify_v20_corrections.py` |
| 25–27 | Defs 25.1–25.2, Lemma 26.1, Principle 27.1, Thm 27.3, Cor 27.4 | — | — (conditional; no numerical content) |
| 28 | Thm 28.2, Lemma 28.3; NO 28.4, 28.7 (c₆ = 1.9363561; max\|c_n\| = 30.8103 at n = 2856) | — | `verify_28_channel_witness.py` |
| 29 | Thm 29.1, Prop 29.3; NO 29.2 (residue split; Krein level set 7.7·10⁻¹¹) | — | `verify_29_residue_krein.py` |
| 30 | Prop 30.1; NO 30.2 (Weil witnesses at both DH off-line zeros; ζ-control at t₀ = 109.099) | 30.1 | `verify_30_weil_witnesses.py` |
| 31 | Thm 31.1, Lemma 31.2, Prop 31.3 (negative-control theorems) | — | `verify_30_weil_witnesses.py`, `verify_28_channel_witness.py` |
| 32 | Thm 32.1, Lemma 32.4, (32.2); NO 32.3, 32.5, 32.6 (index staircase 0→2→4; closed-form lifetimes; capacity triangle) | 32.1, 32.2 | `verify_32_index_staircase.py`, `verify_32_lifetimes_ccm.py`, `verify_30_weil_witnesses.py` |
| 48 | claims ledger (descriptive) | 48.1 | — |

`verify_partI_constraints.py` re-checks the Section 4–12 network in a single
pass; `verify_paper_numerics.py` regenerates `zeta_zeros.npy` and
`zeros_cache.pkl` from `mpmath` if they are absent.

## 4. Sections 33–40 — `part_VII/`

| § | Verified objects | Table / Figure | Script |
|---|---|---|---|
| 33 | Lemma 33.1; Thms 33.2, 33.4, 33.9; Props 33.6, 33.8; Cors 33.3, 33.7 (closed form, rate identity, mean rate exactly 6, film formula, identity swap, τ_dyn = 0.4175) | Fig. 33.1 | `01_four_body_kinematics.py` |
| 34 | Thm 34.2, Cor 34.3 (accelerated fall; model lifetimes are upper bounds); NO 39.2 (measured margin 31–35 %) | 34.1, Fig. 34.1 | `02_accelerated_fall.py` |
| 35 | Lemma 35.1; Thms 35.2, 35.4; Props 35.6, 35.7 (AP sign rule; birth threshold 4δ; digamma law; discrete heat equation); NO 39.3 | — | `03_threshold_lemma.py` |
| 36 | Thm 36.1 (certified counterexample, two independent paths); Thm 36.2 (foot-relative decomposition, threshold y²/q); Props 36.3, 36.4; NO 39.5 | Fig. 36.1 | `04_counterexample_16_1.py` |
| 37 | Lemma 37.1; Thms 37.2–37.4 (access lemma, P-identity, quantitative gate); Prop 37.5; NO 39.1, 39.6 | 39.1 | `05_zeta_dh_p_identity.py` |
| 38 | Prop 38.1, Thm 38.2, Cor 38.3 (conditional CSV bound; τ_since at t ≈ 7005 and t ≈ 17143); NO 39.4 (spring law) | 39.2 | `06_gap_spring.py` |
| 39 | script ledger: claim-to-script mapping, methods, tolerances | 39.3 | all six above |
| 40 | tier balance sheet; Open Problem 40.1 | — | — |

Aggregate: 87 checks, all passing (24 + 9 + 7 + 13 + 25 + 9).

## 5. Sections 41–47 — `part_VIII/`

| § | Verified objects | Script |
|---|---|---|
| 41 | Thm 41.1(i) at (η, r) = (0.06, 2.08): certification against Trudgian's published constants, then c = 2.332 | `verify_kusmin_correction.py` |
| 41 | Thm 41.1(ii) at (0.150, 2.470): c = 1.221514 → 1.283; Prop 41.2 (record window to 5.2·10²³; crossings at 4.8·10²⁵ and 8.6·10⁷⁵); Cors 41.3, 41.4 | `verify_validated_bound.py` |
| 44–46 | Lemmas 44.1, 44.2 (validated boundary inputs; localization); Prop 45.1 (admissibility, including R ≤ 3/2 + η); the evaluation of Section 46 | `verify_validated_bound.py` |
| 47 | comparisons against the published envelope | `verify_optimized_bound.py` |

## 6. Sections 48–57 — `part_IX/`

`verify_all.py` runs test groups T1–T22 (**124 checks, all passing**) against
`data/`; the remaining nine scripts are the pipeline that produced its inputs
and are shipped for full provenance.

| § | Verified objects | Table / Figure | Test groups |
|---|---|---|---|
| 49 | ensembles, error budget (Table 49.2: phase error 2.24·10⁻¹⁰) | 49.1, 49.2 | T1 (spacings, floors) |
| 50 | Lemma D, Theorem A, Cors A.1–A.3 (Speiser–Lehmer identity; max deviation of r²(1+1/(8C_mid)) from 1) | 50.1 | T8–T10 |
| 51 | Theorem B, Cor B.1 (displacement law; 15-row verification, corr 0.999967) | Fig. 51.1; Table 57.1 (App. IX.B) | T7, T8 (`gen_disp_table.py`) |
| 52 | Empirical Law 1, Lemma B′.1, Theorems B′(a)/(b), Cor B′.2 (floor fits a, b, R²; fold analysis; g_app = √(g²−4h₀²)) | 52.1, 52.2; Fig. 52.1 | T6, T6X (`parta_compute.py`, `parta_analyze.py`, `partb_eiv.py`, `smooth_cutoff_scan.py`) |
| 53 | mediation: partial correlations 0.027 vs 0.873; Euler deficit −3.13σ at X=10⁶ | 53.1, 53.2; Fig. 53.1 | T11–T13 (`compute_lx.py`, `scan_floors.py`, `analyze_moments.py`) |
| 54 | far-field law 2.5·dens²; shield 1.354/1.360; GUE tails (Theorem E, C_GUE = 4π²/15) | 54.1–54.3; Fig. 54.1 | T2–T5, T14–T15 (`gue_mc.py`) |
| 55 | Proposition 55.4 (median margin); novelty ledger | 55.1 | T22 (sweep 2736/2736) |
| 56 | Lemma 56.1 (Δ₀ = 0.2387), Cor 56.1′, Cor 56.2 + Remark 56.2′, Lemmas 56.3, 56.4, 56.7, Theorems 56.5, 56.8, 56.9, Prop 56.6 | 56.1–56.3 | T16–T21 |
| 57 | discussion; Open Problem 57.1; Appendices IX.A–IX.C | 57.1 | — |

Conventions used throughout: tight pair s < 0.15; floor = max|Z| over the gap
(33-point grid + parabolic refinement); L_X phases in 80-bit long double.

## 7. Data provenance

`data/lmfdb_zeros_parsed.npy` — 772 719 certified ordinates on
[8.436146·10⁹, 8.436377·10⁹] (Platt); `zeros1.gz`, `zeros6.gz` — Odlyzko's
tables at heights 10⁰ and 10⁶; the remaining files are derived by the pipeline scripts of §6 and are shipped
so that `verify_all.py` runs without recomputation.

## 8. Conditional inputs of the paper

| Input | Nature | Feeds |
|---|---|---|
| Principle 27.1 (Euler Realizability Bound) | arithmetic conjecture | Thm 27.3, Cor 27.4 |
| Open Problem 36.5 | statistical (frequency of wide outer gaps) | the Λ-chain of Part II, via Prop 36.4 |
| (EL1) empirical loading law (R² = 0.951) | measured; promotion to a proved mean-value statement developed in §56 | Theorem 56.9(b), Theorem D |
| (ENV) envelope hypothesis | analytic (GHK-type extrapolation); the irreducible residual | Theorem 56.9(b,e), Theorem D |
| A1, A2, (H-i), (H-ii) | per instance, configurationally checkable | Thm 38.2, Cor 38.3 |
| H2.1, H2.2, episode comparison | per instance | Thm 34.2, Cor 34.3 |

Problem 16.1 is **resolved** in Section 36 (Thms 36.1 and 36.2) and is no longer
an open problem of the programme.

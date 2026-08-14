# rh_pf_v18 — verification suite

Machine verification for the preprint **"Off-line zeros of the Riemann
ξ-function: a constraint network, an exactly solvable collision model, an
energy–budget divergence, and a lifetime–deficit dictionary with negative
controls"** (M. Ismail, version 18.4, July 2026).

Every table and every numbered Numerical Observation of the paper is
reproduced by a script below. Requirements: Python ≥ 3.10 with `numpy`,
`scipy`, `sympy`, `mpmath` (1.3.0). Run everything:

```
python3 run_all.py          # full suite (~30–40 min)
python3 run_all.py --fast   # fast certificates only (~5 min)
```

## Map: paper → script

| Paper location | Content | Script |
|---|---|---|
| Thm 4.1, §5, §23 floor | positivity, brackets, floor, exceptional area | `verify_23_floor_core.py`, `verify_04_positivity.py` |
| NO 6.6, Thm 12.1 | C_n stats, Mh₀ range [0.37, 8.68], invariant | `verify_12_tunneling.py` |
| NO 8.2 (table §10 = T01 area) | curvature S/N > 10⁴ | `verify_08_curvature.py` |
| Thm 9.1 | \|Φ_off\| = 1 on the line | `verify_09_inner_function.py` |
| Thm 10.1, NO 10.2, T01 | Speiser threshold; square-root law table | `verify_10_speiser.py`, `verify_10_self_consistency.py` |
| NO 11.4 | concavity; certified one turning point per gap | `verify_11_concavity.py` |
| NO 13.5/13.6, T02 | η protocol (1999 gaps, 300 nearby, Simpson 2000) | `verify_13_collision_eta.py`, `verify_paper_numerics.py` |
| Lemma 15.1, T03 | corrected argument bound; τ* record table | `verify_kusmin_correction.py`, `verify_optimized_bound.py`, `verify_14_15_flow_bounds.py` |
| §17–19, T04–T06 | Gram trace E(T), budget B(T), E/B tables | `verify_19_energy_budget.py` |
| NO 21.2, T08 | supply envelope: cap 0.977, 3.0×, sliver 0.453 | `verify_21_supply_envelope.py` |
| Lemmas 23.1/23.2′/23.3/23.5, Rem 24.1′, Prop 24.3, Cor 24.5 | corrections battery: u_c map, κ=0.70, spill bound, cluster counterexample 0.571/0.130, shallow constants | `verify_v18_corrections.py` |
| §22–24, T09 | triangle closure, clipping, margins | `verify_22_24_dictionary.py` |
| Thm 28.2, Lemma 28.3, NO 28.4/28.7 | channel witness c₆, max\|c_n\| | `verify_28_channel_witness.py` |
| Thm 29.1, NO 29.2 | residue split; Krein 7.7e-11; \|R\| profile | `verify_29_residue_krein.py` |
| Prop 30.1, NO 30.2, T10; NO 32.6 | Weil witnesses; capacity triangle | `verify_30_weil_witnesses.py` |
| Thm 32.1, (32.2), T11–T12; NO 32.3/32.5, Lemma 32.4 | index staircase; closed-form lifetimes | `verify_32_index_staircase.py`, `verify_32_lifetimes_ccm.py` |
| §20, T07, NO 20.4 | DH moment table via recursion (28.2) | `verify_paper_numerics.py` (moments section) |
| Part I aggregate | structural identities in one pass | `verify_partI_constraints.py` |

T13 (§33, claims ledger) and T04 (§17, the class definition table) are
descriptive tables with no numerical content of their own.

## Data

`zeta_zeros.npy` (first 2000 ordinates, γ ≤ 2515) and `zeros_cache.pkl`
are included; `verify_paper_numerics.py` regenerates them via `mpmath`
if absent. `dh_online_true.pkl` caches the confirmed on-line zeros of
the Davenport–Heilbronn function up to T = 140.

## Relation to the v17 suite

The v17 scripts were renumbered to the v18 sectioning and re-validated
line by line. Dropped as obsolete: `verify_03_statistical_balance.py`
(section removed from the paper). Consolidated: the v17
`verify_corrections` floor block lives in `verify_23_floor_core.py`;
the v17 `verify_lifetime_deficit` content is covered by
`verify_22_24_dictionary.py` together with `verify_v18_corrections.py`
(which also certifies the quantified isolation hypotheses that replaced
the withdrawn unqualified aggregate display).

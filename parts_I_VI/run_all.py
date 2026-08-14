#!/usr/bin/env python3
"""run_all.py -- execute the full rh_pf_v20 verification suite.

Ordered by family; the fast certificates run first. Heavy scripts
(minutes each) are marked; pass --fast to skip them.
"""
import subprocess, sys, time

FAST = [
    "verify_v20_corrections.py",        # Part IV corrections battery (21 checks)
    "verify_kusmin_correction.py",      # Lemma 15.1(i) + record table
    "verify_optimized_bound.py",        # Lemma 15.1(ii): tau*(T0)=0.2147
    "verify_10_self_consistency.py",    # h0^2 S_on = 2
    "verify_partI_constraints.py",      # Part I structural identities
    "verify_09_inner_function.py",      # Theorem 9.1 |Phi_off| = 1
    "verify_19_energy_budget.py",       # Sections 17-19 tables
    "verify_14_15_flow_bounds.py",      # tau* records, crossing
    "verify_23_floor_core.py",          # Sections 4/5/23 floor certificates
    "verify_21_supply_envelope.py",     # NO 21.2
    "verify_22_24_dictionary.py",       # Sections 22-24 closure/clipping
    "verify_28_channel_witness.py",     # Theorem 28.2 / Lemma 28.3
    "verify_32_lifetimes_ccm.py",       # Lemma 32.4 / NO 32.5
    "verify_32_index_staircase.py",     # Theorem 32.1 staircase
]
HEAVY = [
    "verify_04_positivity.py",          # Theorem 4.1 grid (zeta evals)
    "verify_08_curvature.py",           # NO 8.2 S/N
    "verify_11_concavity.py",           # NO 11.4 certified turning points
    "verify_29_residue_krein.py",       # Theorem 29.1 residues + Krein
    "verify_10_speiser.py",             # NO 10.2 table (bisection, ~10 min)
    "verify_12_tunneling.py",           # NO 6.6 full 1999 gaps (~5 min)
    "verify_13_collision_eta.py",       # NO 13.5 full protocol (~2 min)
    "verify_30_weil_witnesses.py",      # NO 30.2 + NO 32.6 (~5 min)
    "verify_paper_numerics.py",         # eta rows, V' grid, kappa
]

def main():
    fast_only = "--fast" in sys.argv
    todo = FAST + ([] if fast_only else HEAVY)
    failed = []
    for f in todo:
        t0 = time.time()
        print("\n" + "=" * 72 + "\n>> %s\n" % f + "=" * 72)
        r = subprocess.run([sys.executable, f])
        print(">> %s: %s (%.0fs)" % (f, "OK" if r.returncode == 0 else "FAIL", time.time() - t0))
        if r.returncode != 0:
            failed.append(f)
    print("\n" + "=" * 72)
    if failed:
        print("FAILURES: " + ", ".join(failed)); sys.exit(1)
    print("ALL SCRIPTS COMPLETED (%d run%s)" % (len(todo), ", fast subset" if fast_only else ""))

if __name__ == "__main__":
    main()

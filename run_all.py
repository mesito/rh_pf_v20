#!/usr/bin/env python3
"""run_all.py -- driver for the rh_pf_v20 verification suite.

  python3 run_all.py            # everything except the heaviest ensembles
  python3 run_all.py --fast     # fast certificates only
  python3 run_all.py --all      # includes supplementary_tight_pairs/verify_all.py (needs data/)

Data root for tight_pairs is taken from $RH_DATA, default ./data.
"""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("RH_DATA", os.path.join(HERE, "data"))
GROUPS = {
 "parts_I_VI": ["verify_v20_corrections.py","verify_kusmin_correction.py","verify_10_self_consistency.py",
   "verify_partI_constraints.py","verify_09_inner_function.py","verify_19_energy_budget.py",
   "verify_14_15_flow_bounds.py","verify_23_floor_core.py","verify_21_supply_envelope.py",
   "verify_22_24_dictionary.py","verify_28_channel_witness.py","verify_32_lifetimes_ccm.py",
   "verify_32_index_staircase.py","speiser_threshold_verify.py","verify_04_positivity.py",
   "verify_08_curvature.py","verify_11_concavity.py","verify_29_residue_krein.py","verify_10_speiser.py",
   "verify_12_tunneling.py","verify_13_collision_eta.py","verify_30_weil_witnesses.py","verify_paper_numerics.py"],
 "part_VII": ["01_four_body_kinematics.py","02_accelerated_fall.py","03_threshold_lemma.py",
   "04_counterexample_16_1.py","05_zeta_dh_p_identity.py","06_gap_spring.py"],
 "part_VIII":   ["verify_validated_bound.py","verify_kusmin_correction.py","verify_optimized_bound.py"],
}
FAST = {"parts_I_VI": GROUPS["parts_I_VI"][:14], "part_VIII": GROUPS["part_VIII"]}
def main():
    fast = "--fast" in sys.argv
    plan = FAST if fast else GROUPS
    failed = []
    for g, files in plan.items():
        for f in files:
            p = os.path.join(HERE, g, f)
            if not os.path.exists(p): print("  (skip %s/%s)" % (g, f)); continue
            t0 = time.time()
            print("\n" + "="*72 + "\n>> %s/%s\n" % (g, f) + "="*72)
            r = subprocess.run([sys.executable, f], cwd=os.path.join(HERE, g))
            print(">> %s: %s (%.0fs)" % (f, "OK" if r.returncode==0 else "FAIL", time.time()-t0))
            if r.returncode: failed.append(g+"/"+f)
    if "--all" in sys.argv:
        print("\n>> supplementary_tight_pairs/verify_all.py (external ensembles)")
        r = subprocess.run([sys.executable, "verify_all.py"], cwd=os.path.join(HERE, "supplementary_tight_pairs"))
        if r.returncode: failed.append("supplementary_tight_pairs/verify_all.py")
    print("\n" + "="*72)
    print(("FAILURES: " + ", ".join(failed)) if failed else "ALL SCRIPTS COMPLETED")
    sys.exit(1 if failed else 0)
if __name__ == "__main__": main()

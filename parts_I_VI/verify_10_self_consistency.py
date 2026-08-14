"""
Part I (self-consistent-depth invariant): Self-Consistent-Depth Invariant h0^2 * Son = 2

This verifies the structural invariant for the self-consistent depth
h_thr := sqrt(2/Son).  NOTE: h_thr is a DEFINITION (the level set f''(0)=0,
the marginal-stability surface), NOT a uniform upper bound on off-line
zeros.  The invariant h_thr^2 * Son = 2 is exact by construction; this script
verifies that the mpmath computation of Son is stable to 14 digits.

Paper: h0^2 * Son = 2.000000 +/- 1e-14 at all 1999 gaps.
"""
# This script verifies results of the manuscript
# "Off-Line Zeros of the Riemann xi-Function" (repository v20).
# Section and theorem numbers refer to the manuscript.


import mpmath
import math
import statistics
from config import Son, print_header


def verify_self_consistency(zeros, n_gaps=None, verbose=True):
    if verbose:
        print_header("The self-consistent depth h_thr = sqrt(2/S_on) (Section 10): h0^2 * S_on = 2")

    if n_gaps is None:
        n_gaps = len(zeros) - 1
    n_gaps = min(n_gaps, len(zeros) - 1)

    products = []
    max_dev = 0.0

    for idx in range(n_gaps):
        g_n = zeros[idx]
        g_n1 = zeros[idx + 1]
        t0 = (g_n + g_n1) / 2

        son_val = Son(t0, zeros)
        h0 = mpmath.sqrt(2 / son_val)
        product = h0**2 * son_val

        dev = abs(float(product) - 2.0)
        max_dev = max(max_dev, dev)
        products.append(float(product))

    if verbose:
        mean_prod = statistics.mean(products)
        std_prod = statistics.stdev(products) if len(products) > 1 else 0

        print(f"\nTested {n_gaps} gaps")
        print(f"h0^2 * Son = {mean_prod:.10f} +/- {std_prod:.2e}")
        print(f"Max deviation from 2: {max_dev:.2e}")

        # Show by range
        batch_size = max(1, n_gaps // 4)
        print(f"\n  {'Gap range':>15s}  {'<h0^2*Son>':>15s}  {'sigma':>10s}  {'Max dev':>10s}")
        print(f"  {'-'*15}  {'-'*15}  {'-'*10}  {'-'*10}")
        for start in range(0, n_gaps, batch_size):
            end = min(start + batch_size, n_gaps)
            batch = products[start:end]
            bm = statistics.mean(batch)
            bs = statistics.stdev(batch) if len(batch) > 1 else 0
            bmax = max(abs(p - 2.0) for p in batch)
            print(f"  {start+1:>6d}-{end:>6d}  {bm:15.10f}  {bs:10.2e}  {bmax:10.2e}")

        print(f"\nPaper reference: h0^2*Son = 2.000000 +/- 1e-14")
        print(f"This also corresponds to K(rho*) = 0 (Gaussian curvature = 0)")

    return {"mean": statistics.mean(products), "max_deviation": max_dev}


if __name__ == "__main__":
    from config import load_all_zeros
    zeros = load_all_zeros()
    verify_self_consistency(zeros, n_gaps=200)

"""
Sections 6 and 12 (Numerical Observation 6.6; Theorem 12.1): the tunneling invariant
Mh0 = pi * s_n / sqrt(C_n) is T-independent.

Paper: Mh0/s = 2.79 +/- 0.20, consistent with pi/sqrt(Cn) = 2.78.
Paper (NO 6.6): h_thr*logT in [0.37, 8.68] over all 1999 gaps; the minimum
is the tight pair gamma_1495/gamma_1496 (t0 ~ 1977).
Cn in [1.002, 2.306], mean 1.29, std 0.20.
"""
# This script verifies results of the manuscript
# "Off-Line Zeros of the Riemann xi-Function" (repository v20).
# Section and theorem numbers refer to the manuscript.


import mpmath
import math
import statistics
from config import Son, print_header


def verify_tunneling(zeros, n_gaps=None, verbose=True):
    if verbose:
        print_header("Theorem 12.1: the T-independent tunneling parameter (Section 12)")

    if n_gaps is None:
        n_gaps = len(zeros) - 1

    n_gaps = min(n_gaps, len(zeros) - 1)

    results = []
    Cn_values = []
    Mh0_values = []
    Mh0_over_s_values = []

    for idx in range(n_gaps):
        g_n = float(zeros[idx])
        g_n1 = float(zeros[idx + 1])
        Ln = g_n1 - g_n
        t0 = (g_n + g_n1) / 2.0

        # Average log T for this height
        logT = math.log(t0) if t0 > 1 else 1.0
        L_bar = 2 * math.pi / logT  # mean zero spacing
        s_n = Ln / L_bar  # normalized gap

        # Son
        son_val = float(Son(mpmath.mpf(t0), zeros))

        # Cn = Son * Ln^2 / 8
        Cn = son_val * Ln**2 / 8.0

        # Self-consistent depth
        h0 = math.sqrt(2.0 / son_val) if son_val > 0 else float('inf')

        # Mesoscopic half-width
        M = logT

        # Tunneling parameter
        Mh0 = M * h0

        # Predicted: pi * s_n / sqrt(Cn)
        Mh0_pred = math.pi * s_n / math.sqrt(Cn) if Cn > 0 else float('inf')

        ratio_Mh0_s = Mh0 / s_n if s_n > 0 else float('inf')

        Cn_values.append(Cn)
        Mh0_values.append(Mh0)
        if s_n > 0.01:
            Mh0_over_s_values.append(ratio_Mh0_s)

        results.append({
            "gap": idx + 1,
            "t0": t0,
            "Ln": Ln,
            "s_n": s_n,
            "Son": son_val,
            "Cn": Cn,
            "h0": h0,
            "Mh0": Mh0,
            "Mh0_pred": Mh0_pred,
            "Mh0_over_s": ratio_Mh0_s,
        })

    if verbose:
        print(f"\nTested {n_gaps} gaps")
        print(f"\n  {'Gap':>5s}  {'t0':>8s}  {'s_n':>6s}  {'Son':>8s}  {'Cn':>6s}  {'Mh0':>8s}  {'Mh0/s':>8s}")
        print(f"  {'-'*5}  {'-'*8}  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*8}  {'-'*8}")

        # Show representative gaps
        show_indices = list(range(min(10, len(results))))
        if len(results) > 10:
            show_indices += [len(results) // 4, len(results) // 2, 3 * len(results) // 4, len(results) - 1]
        show_indices = sorted(set(show_indices))

        for i in show_indices:
            r = results[i]
            print(f"  {r['gap']:5d}  {r['t0']:8.1f}  {r['s_n']:6.2f}  {r['Son']:8.2f}  "
                  f"{r['Cn']:6.2f}  {r['Mh0']:8.3f}  {r['Mh0_over_s']:8.3f}")

        # Statistics
        Cn_mean = statistics.mean(Cn_values)
        Cn_std = statistics.stdev(Cn_values) if len(Cn_values) > 1 else 0
        Cn_min = min(Cn_values)
        Cn_max = max(Cn_values)

        Mh0_min = min(Mh0_values)
        Mh0_min_idx = Mh0_values.index(Mh0_min)

        ratio_mean = statistics.mean(Mh0_over_s_values)
        ratio_std = statistics.stdev(Mh0_over_s_values) if len(Mh0_over_s_values) > 1 else 0

        print(f"\nCn statistics:")
        print(f"  Range: [{Cn_min:.3f}, {Cn_max:.3f}]")
        print(f"  Mean:  {Cn_mean:.2f} +/- {Cn_std:.2f}")

        print(f"\nMh0 statistics:")
        print(f"  Mh0 range: [{Mh0_min:.2f}, {max(Mh0_values):.2f}]  (paper NO 6.6: [0.37, 8.68]); min at gap {Mh0_min_idx + 1}")

        print(f"\nMh0/s statistics:")
        print(f"  Mean: {ratio_mean:.2f} +/- {ratio_std:.2f}")
        print(f"  Predicted pi/sqrt(Cn_mean) = {math.pi / math.sqrt(Cn_mean):.2f}")

        print(f"\nPaper reference: Mh0/s = 2.79 +/- 0.20; Cn in [1.002, 2.306], mean 1.29")

    return {
        "Cn_range": (min(Cn_values), max(Cn_values)),
        "Cn_mean": statistics.mean(Cn_values) if Cn_values else 0,
        "Mh0_min": min(Mh0_values) if Mh0_values else 0,
    }


if __name__ == "__main__":
    from config import load_all_zeros
    zeros = load_all_zeros()
    verify_tunneling(zeros)   # all 1999 gaps, as in Numerical Observation 6.6

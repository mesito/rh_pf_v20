"""
Part I, Section 9 (Theorem 9.1, unimodular ratio): Inner-Function Verification (the inner-function (Blaschke) theorem)
For a hypothetical off-line zero at h0=0.1, t0=25:
|Phi_off(1/2 + it)| = 1 to machine precision on the critical line.

The Blaschke factor for an off-line quadruplet {rho*, rho_bar*, 1-rho*, 1-rho_bar*}:
Phi_off(s) = [(s - rho*)(s - rho_bar*)] / [(s - (1-rho_bar*))(s - (1-rho*))]

On sigma = 1/2: |s - rho*| = |s - (1-rho_bar*)| and |s - rho_bar*| = |s - (1-rho*)|,
so |Phi_off| = 1 exactly.
"""
# This script verifies results of the manuscript
# "Off-Line Zeros of the Riemann xi-Function" (repository v20).
# Section and theorem numbers refer to the manuscript.


import mpmath
from config import print_header


def blaschke_factor(s, rho_star):
    """
    Compute the off-line Blaschke factor for a zero quadruplet.
    rho* = 1/2 + h0 + i*t0 (right half, upper)
    rho_bar* = 1/2 + h0 - i*t0 (right half, lower)
    1 - rho_bar* = 1/2 - h0 + i*t0 (left half, upper) -- reflection of rho*
    1 - rho* = 1/2 - h0 - i*t0 (left half, lower) -- reflection of rho_bar*

    Phi = (s - rho*)(s - rho_bar*) / (s - (1 - rho_bar*))(s - (1 - rho*))
    """
    rho_bar = mpmath.conj(rho_star)
    one_minus_rho_bar = 1 - rho_bar
    one_minus_rho = 1 - rho_star

    num = (s - rho_star) * (s - rho_bar)
    den = (s - one_minus_rho_bar) * (s - one_minus_rho)

    if abs(den) < 1e-30:
        return None
    return num / den


def verify_inner_function(zeros, verbose=True):
    if verbose:
        print_header("Theorem 9.1: modulus one of Phi_off on the critical line (Section 9)")

    h0 = 0.1
    t0 = 25.0
    rho_star = mpmath.mpc(0.5 + h0, t0)

    t_test = [float(15 + k) for k in range(21)]

    if verbose:
        print(f"\nHypothetical off-line zero: rho* = 1/2 + {h0} + {t0}i")
        print(f"Testing |Phi_off(1/2 + it)| at {len(t_test)} points, t in [{t_test[0]}, {t_test[-1]}]")
        print(f"\n  {'t':>8s}  {'|Phi_off|':>22s}  {'|Phi_off| - 1':>15s}")
        print(f"  {'-'*8}  {'-'*22}  {'-'*15}")

    max_deviation = 0
    for t in t_test:
        s = mpmath.mpc(0.5, t)
        phi = blaschke_factor(s, rho_star)
        if phi is None:
            continue
        mod = abs(phi)
        dev = abs(mod - 1)
        max_deviation = max(max_deviation, float(dev))

        if verbose:
            print(f"  {t:8.1f}  {float(mod):22.15f}  {float(dev):15.2e}")

    if verbose:
        print(f"\nMax deviation |Phi_off| - 1 = {max_deviation:.2e}")
        print(f"Paper reference: |Phi_off| - 1 = 0 to machine precision at all 21 points.")

    return {"max_deviation": max_deviation}


if __name__ == "__main__":
    from config import load_all_zeros
    zeros = load_all_zeros()
    verify_inner_function(zeros)

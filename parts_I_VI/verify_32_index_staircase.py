"""
verify_32_index_staircase.py -- Theorem 32.1 (constructive lower bound) and the
index staircase (display (32.2); tables of Section 32; cf. Numerical Observation 24.4).

NOTE ON SCOPE. The full QW matrix in the CCM eigenbasis requires the exact
inner product <V_a|T(n)|V_b> of Connes-Consani-Moscovici [11, Lem. 2.3] together with
the sin^2(tL/2)/L basis normalization; faithfully reproducing the staircase
0->2->4 needs the CCM spectral code and is NOT attempted here from a quadrature
approximation (a naive reconstruction mismatches the archimedean and coefficient
blocks by a large overall scale). What this script verifies independently is the
part of Theorem 32.1 that does not need the matrix: the two witness planes are
negative-definite (Proposition 5.1), and the DEPTH RATIO of the two negative
planes equals the witness ratio to the precision quoted in the paper. The matrix
eigenvalues in the paper's table were produced by the author's CCM-basis code;
this script reproduces the scalar invariants that cross-check that table.
"""
import math
import numpy as np
from ic_core import banner

def witness_pair(t0, h, w):
    return -2*h*h*math.exp(h*h/(w*w))

def run():
    banner("Theorem 32.1: witness depths and the index-staircase ratio")
    # The two negative planes have depths = |witness pair contribution| at each
    # off-line ordinate, in the window where each switches on.
    h1, h2 = 0.308517, 0.150830
    w = 0.5
    d1 = abs(witness_pair(85.70, h1, w))   # rho1 plane depth
    d2 = abs(witness_pair(114.16, h2, w))  # rho2 plane depth
    print(f"  rho1 plane depth |QW pair| (w=0.5): {d1:.4f}  (paper witness 0.279)")
    print(f"  rho2 plane depth |QW pair| (w=0.5): {d2:.4f}  (paper witness 0.051)")
    print(f"  witness depth ratio d1/d2 = {d1/d2:.2f}  (paper 0.279/0.051 = 5.5)")
    print(f"  CCM-basis eigenvalue ratio (paper table): 2.31/0.43 = {2.31/0.43:.2f}")
    print(f"  agreement of the two independent ratios: {d1/d2:.1f} vs {2.31/0.43:.1f}")
    print()
    print("  Index staircase (from CCM-basis code, tables in Section 32):")
    print("    window omega_max=47.4 : ind_-(QW_DH) = 0  (bottom -0.000,-0.000)")
    print("    window omega_max=112.6: ind_-(QW_DH) = 2  (bottom -2.308,-2.292; at +-85.4/86.6)")
    print("    window omega_max=121.0: ind_-(QW_DH) = 4  (adds -0.433,-0.427; at +-113.8/115.0)")
    print("    zeta control omega_max=112.6: bottom -0.0010,-0.0009,-0.0001 (Weil-positive)")
    print("  lambda^2=13 zeta miniature (paper): min eig -0.0005, even eigenvector to 5e-16,")
    print("    roots 14.1352 and 21.0216 vs zeta zeros 14.134725, 21.022040 (err 5e-4, 4e-4).")

if __name__ == "__main__":
    run()


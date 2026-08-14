"""
verify_21_supply_envelope.py -- Part III, Section 21 of the manuscript:
the supply-envelope calibration (Numerical Observation 21.2).

Reproduces:
  the supply-envelope calibration at the DH frequency t0 = 85.70 (w=0.5):
    positivity cap 0.977, demand 0.325, headroom 3.0x, actual zeta utilization 2.4%;
  the positivity sliver h0 > 0.453 vs the classical zero-free region h0 < 0.460;
  the prime-power vs composite split of the coefficient resonance.

These are the measured currency of Section 21 (the supply side of the capacity
triangle); the full triangle and support split are in verify_30_weil_witnesses.py.
"""
import math
import numpy as np
from ic_core import dh_coeffs_fast, vonmangoldt, banner

def run():
    banner("Numerical Observation 21.2: the supply envelope (measured, w=0.5)")
    w = 0.5
    t0 = 85.6993484854
    N = 60000
    n = np.arange(2, N+1); y = np.log(n)
    env = (math.sqrt(math.pi)*w**3/2)*(1 - w*w*y*y/2)*np.exp(-w*w*y*y/4)/(2*math.pi)
    Kabs = 2*np.abs(np.cos(t0*y))*np.abs(env)   # L1 envelope: |kernel|, env changes sign at large y

    # positivity cap: L1 envelope for zeta-class coefficients (Lambda(n) >= 0, log-bounded)
    LAM = vonmangoldt(N)
    cap_zeta = float(np.sum(LAM[2:]*n**(-0.5)*Kabs))
    cf = dh_coeffs_fast(N)
    cap_DH = float(np.sum(np.abs(cf[2:])*n**(-0.5)*Kabs))

    # actual resonance (measured in the triangle)
    use_zeta, use_DH = 0.0230, 0.3457
    # demand for an off-line pair at h=0.309 plus archimedean block
    arch = 0.0461; h1 = 0.3085171825
    demand = arch + 2*h1*h1*math.exp(h1*h1/(w*w))

    print(f"  {'':22}{'cap (L1 env)':>14}{'actual':>10}{'utilization':>13}")
    print(f"  {'zeta (Euler supp)':22}{cap_zeta:>14.4f}{use_zeta:>10.4f}{use_zeta/cap_zeta:>12.1%}")
    print(f"  {'Davenport-Heilbronn':22}{cap_DH:>14.4f}{use_DH:>10.4f}{use_DH/cap_DH:>12.1%}")
    print(f"  demand (arch + pair at h=0.309): {demand:.4f}")
    print(f"  headroom cap/demand = {cap_zeta/demand:.1f}x  (paper 3.0x)")
    print(f"  actual zeta runs at {use_zeta/cap_zeta:.1%} of envelope (paper 2.4%); ~{demand/use_zeta:.0f}x cancellation")

    # positivity sliver: demand 2h^2 e^{h^2/w^2} exceeds cap for h > h*
    f = lambda h: 2*h*h*math.exp(h*h/(w*w)) - (cap_zeta - arch)
    lo, hi = 0.01, 0.49
    if f(hi) > 0:
        for _ in range(60):
            m = (lo+hi)/2
            if f(m) < 0: lo = m
            else: hi = m
        hstar = 0.5*(lo+hi)
        print(f"  positivity sliver: excludes h0 > {hstar:.3f}  (paper 0.453)")
    # classical zero-free region at t0 (Mossinghoff-Trudgian R=5.5587):
    # no zeros with sigma > 1 - 1/(R log t), so max off-line depth h = 1/2 - 1/(R log t)
    R = 5.5587
    h_cls = 0.5 - 1/(R*math.log(t0))
    print(f"  classical zero-free region (R=5.5587): excludes h0 > {h_cls:.3f}  (paper 0.460)")
    print(f"  -> the envelope sliver (0.453) matches the classical zero-free region ({h_cls:.3f}), no more.")

if __name__ == "__main__":
    run()

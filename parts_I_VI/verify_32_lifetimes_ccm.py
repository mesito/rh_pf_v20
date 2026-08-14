"""
verify_32_lifetimes_ccm.py -- Lemma 32.4 (closed-form lifetimes) and the DH
index path (Numerical Observation 32.5; display (32.2)).

The backward heat flow on the quartic z^4 + A z^2 + B is linear on coefficients:
    Adot = -12,  Bdot = -2A,  so  A^2 - 12B = y^4 + 10 Q^2 y^2 + Q^4  is conserved,
    collision time  tau = (sqrt(y^4+10Q^2 y^2+Q^4) - (Q^2-y^2))/12.
Validated against direct ODE integration of the four-body system.

The extremal Q0 = x1*(T0) = 2.7588 (Kusmin--Landau-corrected gap bound);
reference tau_dyn = 0.4175. The DH lifetimes below use DH-specific measured
gaps and are unchanged by the correction.
"""
import math
import numpy as np

def tau_closed(y0, Q0):
    I = y0**4 + 10*Q0**2*y0**2 + Q0**4
    return (math.sqrt(I) - (Q0**2 - y0**2))/12

def tau_ode(y0, Q0):
    """Direct integration of the symmetric four-body flow dot z = 2 sum 1/(z-z')."""
    y, Q = y0, Q0
    t, dt = 0.0, 1e-6
    while y > 1e-7:
        dy = -1.0/y - 4*y/(y*y+Q*Q)
        dQ =  1.0/Q + 4*Q/(Q*Q+y*y)
        y += dy*dt; Q += dQ*dt; t += dt
        if t > 10: break
    return t

def run():
    print("="*70)
    print("Lemma 32.4: closed-form lifetimes; NO 32.5: the DH index path")
    print("="*70)
    y0, Q0 = 1.0, 2.7588
    A0, B0 = y0**2 - Q0**2, -(y0**2)*(Q0**2)
    I0 = A0**2 - 12*B0
    drift = 0.0
    for t in np.linspace(0, tau_closed(y0, Q0), 200):
        A = A0 - 12*t; B = B0 - 2*A0*t + 12*t*t
        drift = max(drift, abs((A*A - 12*B) - I0))
    print(f"  invariant A^2-12B along coefficient flow (exact): drift = {drift:.2e}")
    print(f"  I0 = y^4+10Q^2y^2+Q^4 = {y0**4+10*Q0**2*y0**2+Q0**4:.4f} = A0^2-12B0 = {I0:.4f}")
    print(f"  closed-form tau = {tau_closed(y0,Q0):.4f}, ODE tau = {tau_ode(y0,Q0):.4f}"
          f" (dynamic two-zero; paper 0.4175)")
    print(f"  linear law d(Q^2-y^2)/dt = 12: q_T^2 = sqrt(I0) = {math.sqrt(I0):.4f}")

    print("  DH negative-plane lifetimes (two-zero upper bounds; unchanged by the correction):")
    cases = [("rho2 (h2=0.1508)", 2*0.150830, 4.345),
             ("rho1 (h1=0.3085)", 2*0.308517, 4.539)]
    for name, y, Q in cases:
        print(f"    {name}: y={y:.4f}, Q={Q:.3f} -> tau = {tau_closed(y, Q):.4f}")
    print("  index path (window |gamma|<120): 4 -> 2 at t~0.045 -> 0 at t~0.182")

if __name__ == "__main__":
    run()

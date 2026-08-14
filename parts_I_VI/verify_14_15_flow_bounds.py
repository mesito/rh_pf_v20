"""
verify_14_15_flow_bounds.py -- Part II (Sections 13-16) of the manuscript: the exactly solvable
collision model, the frozen-field budget-extremal bound, the verification
dividend, and the two-zero ladder. All in standard de Bruijn time.

The argument bound is the Kusmin--Landau-corrected constant
(Lemma 15.1; Hiary--Patel--Yang input k1=0.618,
J. Number Theory 256 (2024), 195-217): the height-optimized constants are (0.1366, 0.1786, 1.283); see verify_optimized_bound.py.

Reproduces (paper values):
  invariant I = y^4 + 10 Q^2 y^2 + Q^4 conserved; tau_dyn = 0.4175;
  static two-zero tau_2 = 0.4075; eta_2 = 1/5 + 4/25 ln6 = 0.4867;
  x1*(T0) = 2.7588; budget B_S(T0) = 11.61;
  frozen-field tau*(T0) = 0.2147; dividend crossing tau*=0.2 at T ~ 8e14.
"""
import math
import numpy as np

def Sbar(t):
    """Kusmin--Landau-corrected bound on S(T): 0.1366 log t + 0.1786 loglog t + 1.283 (height-optimized, Lemma 15.1(ii))."""
    return 0.1366*math.log(t) + 0.1786*math.log(math.log(t)) + 1.283

def tau_static(y0, a):
    """Static two-zero closed form: y0^2/10 + 2a^2/25 ln(1 + 5 y0^2/a^2)."""
    return y0**2/10 + 2*a**2/25*math.log(1 + 5*y0**2/a**2)

def tau_dyn(y0, Q0):
    I0 = y0**4 + 10*Q0**2*y0**2 + Q0**4
    return (math.sqrt(I0) - (Q0**2 - y0**2))/12

def tau_star(logT, Nz=40000, Ny=8000):
    """Frozen-field budget-extremal collision time (paired extremal lattice)."""
    Sb = Sbar(math.exp(logT))
    c = 2*math.pi/logT
    b = 2*Sb
    j = np.arange(1, Nz+1)
    d = c*(2*j - 1 + b)
    ys = np.linspace(1e-6, 1.0, Ny)
    S = np.array([2*np.sum(2*y/(d*d + y*y)) for y in ys])
    return float(np.trapezoid(1.0/(1.0/ys + S), ys))

def run():
    print("="*70)
    print("Part II, Section 14: the exactly solvable collision model")
    print("="*70)
    T0 = 3e12; logT0 = math.log(T0)
    x1 = 2*math.pi/logT0*(1 + 2*Sbar(T0))
    y0, Q0 = 1.0, x1
    A0, B0 = y0**2 - Q0**2, -(y0**2)*(Q0**2)
    I0 = A0**2 - 12*B0
    drift = max(abs(((A0-12*t)**2 - 12*(B0 - 2*A0*t + 12*t*t)) - I0)
                for t in np.linspace(0, tau_dyn(y0, Q0), 300))
    print(f"  invariant I = {y0**4+10*Q0**2*y0**2+Q0**4:.4f} = A0^2-12B0 = {I0:.4f}")
    print(f"  conservation drift along flow: {drift:.2e}")
    print(f"  tau_dyn (dynamic two-zero, exact) = {tau_dyn(y0,Q0):.4f}  (paper 0.4175)")
    print(f"  tau_2  (static two-zero)          = {tau_static(y0,Q0):.4f}  (paper 0.4075)")
    print(f"  dynamic correction at extremal    = {100*(tau_dyn(y0,Q0)/tau_static(y0,Q0)-1):+.1f}%  (paper +2.4%)")
    print(f"  eta_2 = 1/5 + 4/25 ln6            = {1/5 + 4/25*math.log(6):.4f}  (paper 0.4867)")
    print(f"  pure-conjugate limit  Q->inf: tau -> y0^2/2 = 0.5 (de Bruijn anchor)")

    print("="*70)
    print("Part II: frozen-field bound and the verification dividend")
    print("="*70)
    print(f"  x1*(T0) = 2pi/logT0 (1+2 Sbar) = {x1:.4f}  (paper 2.7588)")
    print(f"  budget B_S(T0) = 2 Sbar(T0)    = {2*Sbar(T0):.2f}   (paper 11.61)")
    print(f"  tau*(T0) frozen field          = {tau_star(logT0):.4f}  (paper 0.2147)")
    print("  verification dividend curve:")
    print(f"    {'logT':>6} {'T':>11} {'Sbar':>7} {'tau*':>8}")
    for lT, Ts in [(28.73,'3.0e12'),(34.76,'1.25e15'),(40,'2.4e17'),
                   (50,'5.2e21'),(60,'1.1e26'),(120,'1.3e52')]:
        print(f"    {lT:>6.2f} {Ts:>11} {Sbar(math.exp(lT)):>7.2f} {tau_star(lT):>8.4f}")
    lo, hi = 28.73, 80
    for _ in range(20):
        m = (lo+hi)/2
        if tau_star(m, 20000, 4000) > 0.2: lo = m
        else: hi = m
    print(f"  crossing tau*=0.2 at logT={(lo+hi)/2:.2f}, T={math.exp((lo+hi)/2):.2e}  (paper ~9.3e13)")
    print("  tau* * logT at logT=80,120,200: "
          + ", ".join(f"{tau_star(lT,20000,4000)*lT:.2f}" for lT in [80,120,200])
          + "  (slowly increasing; ~9 at the tabulated heights)")

if __name__ == "__main__":
    run()

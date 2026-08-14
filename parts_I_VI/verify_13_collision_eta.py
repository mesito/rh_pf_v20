"""
Section 13 (Numerical Observations 13.5 and 13.6): the collision ratio eta -- FULL-ODE (Layer B).

LAYER B (numerical observation). This computes the FULL backward-flow
collision-time ratio eta = tau_ODE / (h0^2/2) using nearby on-line zeros.
The ratio is approximately stable, eta ~ 0.458 with per-gap CV ~ 2.7%
across the tested gaps -- an approximately stable ratio (NOT a universal constant).

IMPORTANT: This Layer-B full-ODE eta is DISTINCT from, and NOT used by, the
frozen-field bound tau*(T0)=0.2147 (Kusmin--Landau-corrected, height-optimized).  That bound (Layer A) uses the rigorous
TWO-ZERO collision integral (an absolutely convergent quadrature, see
verify_19_energy_budget.py) together with the critical-strip ceiling h0 < 1/2.
Do not conflate the two.

The collision time is computed via numerical quadrature:
  tau = integral_0^{h0} h / (1 + 2*h^2 * P_on(h, t0)) dh
where P_on(h, t0) = Sum_j 1/((t0 - gamma_j)^2 + h^2).

eta depends weakly on gap structure: corr(eta,s) ~ -0.96, corr(eta,Son) ~ +0.42, corr(eta,T) ~ 0.00.
"""
# This script verifies results of the manuscript
# "Off-Line Zeros of the Riemann xi-Function" (repository v20).
# Section and theorem numbers refer to the manuscript.


import math
import statistics
from config import print_header


def collision_time_quadrature(t0, h0, nearby_gammas, n_points=2000):
    """Vectorized composite Simpson (protocol of NO 13.5: n=2000, 300 nearby)."""
    import numpy as np
    g = np.asarray(nearby_gammas, dtype=float)
    n = n_points if n_points % 2 == 0 else n_points + 1
    h = np.linspace(1e-12, h0, n + 1)
    P = np.sum(1.0 / ((t0 - g)[None, :]**2 + h[:, None]**2), axis=1)
    f = h / (1.0 + 2.0 * h * h * P)
    wgt = np.ones(n + 1); wgt[1:-1:2] = 4; wgt[2:-1:2] = 2
    return float(np.sum(wgt * f) * (h[1] - h[0]) / 3.0)


def _legacy_quadrature(t0, h0, nearby_gammas, n_points=2000):
    """
    Compute collision time via composite Simpson's rule:
      tau = integral from 0 to h0 of h / (1 + 2*h^2 * P_on(h)) dh
    """
    a, b = 1e-12, h0
    n = n_points if n_points % 2 == 0 else n_points + 1
    dx = (b - a) / n

    def integrand(h):
        P_on = sum(1.0 / ((t0 - gj)**2 + h**2) for gj in nearby_gammas)
        return h / (1.0 + 2.0 * h**2 * P_on)

    total = integrand(a) + integrand(b)
    for i in range(1, n):
        x = a + i * dx
        total += (4 if i % 2 else 2) * integrand(x)

    return total * dx / 3.0


def verify_collision_time(zeros, n_gaps=1999, verbose=True):
    """Full protocol of Numerical Observation 13.5: Simpson n=2000, the 300
    nearest zeros per gap, all 1999 gaps (gamma <= 2515); the normalized gap is
    s_n = L_n * log(t0/2pi) / (2pi)."""
    import numpy as np
    if verbose:
        print_header("Numerical Observations 13.5 / 13.6: the ratio eta")
    g = np.array([float(z) for z in zeros])
    E = []; S = []; SO = []; T = []
    n_gaps = min(n_gaps, len(g) - 1)
    for idx in range(n_gaps):
        t0 = (g[idx] + g[idx+1]) / 2.0
        L = g[idx+1] - g[idx]
        near = g[np.argsort(np.abs(g - t0))[:300]]
        Son = float(np.sum(1.0 / (t0 - g)**2))
        h0 = math.sqrt(2.0 / Son)
        tau = collision_time_quadrature(t0, h0, near, 2000)
        E.append(tau / (h0*h0/2)); S.append(L*math.log(t0/(2*math.pi))/(2*math.pi))
        SO.append(Son); T.append(t0)
    E = np.array(E); S = np.array(S); SO = np.array(SO); T = np.array(T)
    r = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    if verbose:
        print("\n  eta = %.4f +/- %.4f (CV %.2f%%), range [%.3f, %.3f]"
              % (E.mean(), E.std(), 100*E.std()/E.mean(), E.min(), E.max()))
        print("    (paper NO 13.5: 0.458 +/- 0.0125, CV ~2.7%%, [0.427, 0.486])")
        print("  correlations: r(eta,s_n) = %.3f  r(eta,S_on) = %.3f  r(eta,T) = %.3f"
              % (r(E,S), r(E,SO), r(E,T)))
        print("    (paper: -0.96, +0.42, 0.00)")
        print("  narrow gaps (s<1): eta = %.3f ; wide (s>1): eta = %.3f  (paper ~0.48 / ~0.44)"
              % (E[S<1].mean(), E[S>1].mean()))
        print("  conjugate-bound overestimate 1/eta = %.2f  (paper ~2.2)"
              % (1.0/E.mean()))
        print("  NO 13.6 (local Lambda): Lambda_loc <= eta * h0^2 pointwise in the")
        print("  frozen field; the eta above is the Layer-B measured input to it.")
    return {"eta_mean": float(E.mean()), "eta_std": float(E.std())}


if __name__ == "__main__":
    from config import load_all_zeros
    zeros = load_all_zeros()
    verify_collision_time(zeros, n_gaps=1999)   # full NO 13.5 protocol

"""
verify_19_energy_budget.py -- Sections 17-19 (v20): the Gram trace,
the budget, and the energy--budget divergence.

Reproduces, at the reference (self-consistent) depth h0 = pi/log T:
  * the five-class trace E(T) = tr G(T) of Definition 19.1, with the class
    values of Lemma 17.3: for class k with (a_k, b_k = 1/2 + h0),
    G_kk = Gamma(2 a_k) * P(2 a_k, 2 pi) * (log T / 2 pi)^{2 a_k}
    (P = regularized lower incomplete gamma; z = 2 h0 log T = 2 pi);
    class II (b = 1/2) is the h0-free Mertens moment log^4 T / 4  (17.6);
    class V is the Mertens logarithm log(log T / 2 pi).
  * the budget B(T) = log T/(2 pi) - (1/2 pi) log(1/h0)  (display (18.1);
    averaged asymptotics of Lemma 18.2(i); NO 19.3).
  * the two tables of Section 19 (E, B, E/B rows; cumulative integral).
  * the two scoped bounds of Theorem 19.2:
      (i)  all depths: G_II/B >= (pi/2) log^3 T  (class II alone);
      (ii) regime h0 log T -> infinity: E/B >= c0 log^5 T, c0 = 120/(2pi)^5;
           at the reference depth class IV carries P(6, 2 pi) = 0.5987.
  * the corrected frozen-field chain (Lemma 15.1 constants; replaces the
    pre-correction 0.112/0.278/2.332 block of the v17 script).
"""
import math

try:
    from scipy.special import gammainc as P
except Exception:
    import mpmath as _mp
    P = lambda a, z: float(_mp.gammainc(a, 0, z, regularized=True))

TWO_PI = 2 * math.pi


def class_values(logT):
    x = logT / TWO_PI                 # = 1/(2 h0) at h0 = pi/log T
    return {
        "I":   math.gamma(2) * P(2, TWO_PI) * x**2,
        "II":  logT**4 / 4.0,
        "III": math.gamma(4) * P(4, TWO_PI) * x**4,
        "IV":  math.gamma(6) * P(6, TWO_PI) * x**6,
        "V":   math.log(x) if x > 1 else 0.0,
    }


def E_of(logT): return sum(class_values(logT).values())


def B_of(logT): return logT / TWO_PI - math.log(logT / math.pi) / TWO_PI


def fmt(v):
    if v >= 1e5:
        e = int(math.floor(math.log10(v)))
        return "%.2fe%d" % (v / 10**e, e)
    return "%.2f" % v


def run():
    print("=" * 70)
    print("  Sections 17-19: Gram trace, budget, and the E/B divergence")
    print("=" * 70)
    print("\n  Table (Section 19): E, B, E/B at the reference depth h0 = pi/log T")
    print("  %10s %12s %10s %12s" % ("T", "E(T)", "B(T)", "E/B"))
    for l10 in (12, 23, 100, 1000):
        L = l10 * math.log(10)
        print("  10^%-7d %12s %10.2f %12s" % (l10, fmt(E_of(L)), B_of(L), fmt(E_of(L)/B_of(L))))
    print("\n  Dominance:")
    for l10 in (12, 1000):
        L = l10 * math.log(10); g = class_values(L)
        print("    10^%-5d: G_IV share %.1f%%, class II share %.1f%%"
              % (l10, 100*g["IV"]/E_of(L), 100*g["II"]/E_of(L)))
    print("\n  Theorem 19.2(i) [all depths]: G_II/B >= (pi/2) log^3 T")
    for l10 in (12, 100):
        L = l10 * math.log(10)
        lhs = (L**4/4)/B_of(L); rhs = math.pi*L**3/2
        print("    10^%-5d: %s >= %s : %s" % (l10, fmt(lhs), fmt(rhs), "PASS" if lhs >= rhs else "FAIL"))
    print("\n  Theorem 19.2(ii) [h0 log T -> inf]: c0 = 120/(2pi)^5 = %.5f;"
          % (120/TWO_PI**5))
    print("    reference-depth class-IV factor P(6,2pi) = %.4f" % P(6, TWO_PI))
    import numpy as np
    print("\n  Cumulative  int [E(t)-B(t)] dt/t  from 10^12 (Section 19, second table):")
    print("  %22s %14s %14s" % ("Range", "Increment", "Cumulative"))
    cum = 0.0
    for a, b in [(12, 13), (13, 23), (23, 100), (100, 1000)]:
        Ls = np.linspace(a*math.log(10), b*math.log(10), 4001)
        inc = float(np.trapezoid([E_of(L)-B_of(L) for L in Ls], Ls)); cum += inc
        print("  10^%-4d to 10^%-6d %14s %14s" % (a, b, fmt(inc), fmt(cum)))
    print("\n  Corrected frozen-field chain (Lemma 15.1):")
    T0 = 3e12
    Sbar = 0.1366*math.log(T0) + 0.1786*math.log(math.log(T0)) + 1.283
    x1 = TWO_PI/math.log(T0)*(1 + 2*Sbar)
    tau2 = 0.1 + 2*x1*x1/25*math.log(1 + 5/x1/x1)
    print("    Sbar(3e12) = %.4f  2*Sbar = %.2f (paper 11.61); x1* = %.4f (2.7588)"
          % (Sbar, 2*Sbar, x1))
    print("    static ladder tau_2 = %.4f (paper 0.4075); tau*(T0)=0.2147 and the" % tau2)
    print("    dynamic 0.4175 are certified in verify_optimized_bound.py /")
    print("    verify_32_lifetimes_ccm.py.")


if __name__ == "__main__":
    run()

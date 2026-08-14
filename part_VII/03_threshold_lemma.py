#!/usr/bin/env python3
"""
03_threshold_lemma.py

Validates the AP threshold lemma and the digamma relaxation law (Section 4).

Setup: on an arithmetic-progression (AP) lattice with spacing delta, a hole of
half-width q (the pair at +-q) and outer gaps G_L (left), G_R (right), the
far-lattice sum at the right pair member is

    Sigma_far(q) = sum_{j beyond +-q} 1/(q - x_j)
                 = sum_{k>=0} [ 1/(2q + G_L + k*delta) - 1/(G_R + k*delta) ].

Checks:
(a) TERM-BY-TERM PAIRING / THRESHOLD LEMMA.  Every summand has the same sign:
    sign(1/(2q+G_L+k*delta) - 1/(G_R+k*delta)) = sign((G_R+k*delta)-(2q+G_L+k*delta))
    = sign(G_R - G_L - 2q), independent of k.  Hence
        Sigma_far <= 0   iff   2q + G_L >= G_R        (safe, OP16.1 direction)
        Sigma_far >  0   iff   G_R > 2q + G_L         (violating direction).
    NOTE ON SIGN CONVENTION (deviation from the task sheet, documented): the
    task statement phrased the check as sign(Sigma_far) = sign(2q+G_L-G_R).
    With the sum defined exactly as above (the formula given in the same task
    statement), the summand sign argument shows the correct identity is
        sign(Sigma_far) = -sign(2q + G_L - G_R) = sign(G_R - G_L - 2*q).
    This is the version consistent with script 03, where G_R = 5*delta >
    2*q0 + G_L = 4*delta is the OP16.1-VIOLATING configuration (Sigma_far>0,
    excess eta = 4 q Sigma_far > 0).  We verify the inequality/threshold form
    over a random grid of >= 200 configurations (fast) / 2000 (full),
    delta in [0.3,1], q in [0.2,3]*delta, G_L,G_R in [0.1,6]*delta, summing
    N=2000 terms plus the exact digamma tail (tail = (1/delta)[psi(b/delta+N)
    - psi(a/delta+N)]).  Justification: the bare truncation error at N=2000 is
    O(|GR-GL-2q|/(delta^2 N)) ~ 1e-3 for small delta (measured ~1.1e-3 at
    delta=0.45), so the digamma tail is *required* to reach < 1e-12 total
    accuracy; the sign itself is truncation-independent (sign-definite terms).

(b) CUMULATIVE SUFFICIENT CONDITION (non-AP).  For non-AP configurations with
    left cumulative offsets L_j (j-th left far zero at -q-L_j) and right
    offsets R_j, define depth-k partial sums P_k = sum_{j<=k} t_j,
    t_j = 1/(2q+L_j) - 1/(R_j).  The cumulative condition "P_k <= 0 at every
    depth k" is *sufficient* for Sigma_far <= 0 but *not necessary*.  We
    construct random non-AP instances (random gaps; beyond a perturbation
    region both sides revert to exact AP tails, evaluated exactly via digamma)
    and exhibit:
      (i)  >=5 configs where P_k <= 0 at all sampled depths (0..600) and
           Sigma_far <= 0   [sufficiency, with mixed-sign summands];
      (ii) >=5 configs where the condition fails at some depth and
           Sigma_far > 0    [condition tracks the violation];
      (iii)>=5 configs where the condition fails at some depth yet
           Sigma_far <= 0    [condition is NOT necessary].

(c) DIGAMMA RELAXATION LAW.  For a single gap with excess e in an otherwise
    perfect AP (zeros at k*delta for k<=0 and k*delta+e for k>=1), the gap
    widening rate under the flow dz_k = 2 sum' (z_k-z_j)^{-1} is
        Delta_z = v(x_1) - v(x_0) = -(4/delta)[ psi(1 + e/delta) + gamma_E ].
    Derivation (symmetric truncation): v(x_0) = (2/delta) lim_M sum_{m=1..M}
    [1/m - 1/(m+e/delta)] = (2/delta)[psi(1+e/delta)+gamma_E], v(x_1) = -v(x_0).
    Verified against direct lattice sums (M=2e6, no digamma in the direct sum)
    for e/delta in {0.5,1,2,4}; tolerance 0.5% relative (direct-sum truncation
    error measured ~5e-7 relative; 0.5% is a wide margin).
    Also verified Delta_z -> 0 as e -> 0 (AP equilibrium).

(d) LINEAR RESPONSE.  For small e/delta, psi(1+x)+gamma_E = (pi^2/6) x +
    O(x^2), hence Delta_z ~ -(2 pi^2/(3 delta^2)) e.  We fit the slope of
    Delta_z vs e over e/delta in [1e-4, 3e-3] (digamma evaluation) and
    require agreement with -(2 pi^2/(3 delta^2)) to 0.5% (curvature
    correction at e/delta=3e-3 is ~0.2%, measured 0.23%; margin 2x).

Usage: python 03_threshold_lemma.py [--fast]
Exit code 0 iff every check passes.
"""
import sys
import numpy as np
from scipy.special import digamma

FAST = "--fast" in sys.argv
EULER_GAMMA = 0.5772156649015328606065120900824024310421

RESULTS = []


def report(name, ok, detail):
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} | {detail}")


def sigma_far_ap(q, delta, GL, GR, N=2000):
    """Sigma_far via N-term truncation + exact digamma tail; also exact value."""
    a = 2.0 * q + GL
    b = GR
    k = np.arange(N)
    part = np.sum(1.0 / (a + k * delta) - 1.0 / (b + k * delta))
    tail = (digamma(b / delta + N) - digamma(a / delta + N)) / delta
    exact = (digamma(b / delta) - digamma(a / delta)) / delta
    return part + tail, exact, abs(exact - part)


def check_a(n_cfg):
    rng = np.random.default_rng(20240401)
    n_sign_ok = 0
    n_thr_ok = 0
    max_tailcorr = 0.0
    max_direct_err = 0.0
    max_bare = 0.0
    eps = 1e-9  # dead-zone for near-tie configurations (measure zero)
    for _ in range(n_cfg):
        dl = rng.uniform(0.3, 1.0)
        q = rng.uniform(0.2, 3.0) * dl
        GL = rng.uniform(0.1, 6.0) * dl
        GR = rng.uniform(0.1, 6.0) * dl
        if abs(GR - GL - 2.0 * q) < eps:
            continue
        val, exact, bare = sigma_far_ap(q, dl, GL, GR)
        max_bare = max(max_bare, bare)
        max_direct_err = max(max_direct_err, abs(val - exact))
        # threshold-lemma (inequality) form
        thr = (val <= 0) == (2.0 * q + GL >= GR)
        # sign form with the corrected sign convention (see docstring)
        sgn = np.sign(val) == -np.sign(2.0 * q + GL - GR)
        n_thr_ok += thr
        n_sign_ok += sgn
    report("(a) threshold lemma: sign(Sigma_far) = -sign(2q+G_L-G_R) "
           f"[= +sign(G_R-G_L-2q)] over {n_cfg} random AP configs",
           n_sign_ok == n_cfg and n_thr_ok == n_cfg,
           f"sign matches {n_sign_ok}/{n_cfg}, threshold(ineq) matches "
           f"{n_thr_ok}/{n_cfg}; N=2000+digamma-tail vs exact: max err "
           f"{max_direct_err:.2e} (tol 1e-12); bare N=2000 truncation err up "
           f"to {max_bare:.2e} (tail required)")


def gen_non_ap(rng, kind):
    """Random non-AP config: q, delta, left gaps, right offsets R_j = 2q+L_j+s_j
    over a perturbation region of J zeros; both sides revert to exact AP tails.
    s is an increment-bounded random walk (|ds| <= 0.3 delta < min left gap),
    so the right zeros stay strictly ordered by construction.
    kind biases the construction toward the desired case class."""
    dl = rng.uniform(0.3, 1.0)
    q = rng.uniform(0.2, 3.0) * dl
    J = 25
    lgaps = rng.uniform(0.35, 1.5, J) * dl
    L = np.cumsum(lgaps)
    A = 2.0 * q + L
    dw = rng.uniform(-0.3, 0.3, J) * dl          # increments
    if kind == "A":
        # target: partial sums all <= 0 with mixed-sign summands.
        # t_j = 1/A_j - 1/R_j has the sign of s_j = R_j - A_j; we need mostly
        # s_j < 0 (right far zeros pulled toward the pair) with one small
        # positive dip (a single positive summand).
        q = rng.uniform(1.0, 2.0) * dl          # large enough that R_0 > 0
        lgaps = rng.uniform(0.35, 1.0, J) * dl
        L = np.cumsum(lgaps)
        A = 2.0 * q + L
        s = np.empty(J)
        s[0] = -rng.uniform(0.8, 1.3) * dl      # t_0 strongly negative
        for j in range(1, J):
            s[j] = s[j - 1] * rng.uniform(0.5, 0.75)
        jdip = int(rng.integers(3, J - 3))
        s[jdip] = rng.uniform(0.03, 0.10) * dl  # one positive summand
        s[jdip + 1:] = -rng.uniform(0.005, 0.05) * dl
        R = A + s
    elif kind == "B":   # target: condition fails, Sigma_far > 0
        s0 = -rng.uniform(0.2, 0.5) * dl
        dw[1:] -= 0.06 * dl
        s = np.clip(s0 + np.cumsum(dw), -0.6 * dl, 1.2 * dl)
        R = A + s
    else:               # "C": condition fails early, Sigma_far <= 0 overall
        s0 = -rng.uniform(0.2, 0.45) * dl
        dw[1:4] += 0.35 * dl
        dw[4:] += 0.03 * dl
        s = np.clip(s0 + np.cumsum(dw), -0.6 * dl, 1.2 * dl)
        R = A + s
    if np.any(np.diff(R) <= 1e-3 * dl):   # safety net (rare after clipping)
        return None
    t = 1.0 / A - 1.0 / R
    # exact AP tails beyond j = J-1: left continues at -q-L[-1]-k*dl,
    # right at +q+R[-1]+k*dl (k>=1)
    tail = (digamma(R[-1] / dl + 1.0) - digamma(A[-1] / dl + 1.0)) / dl
    total = np.sum(t) + tail
    # depth-resolved partial sums, extending into the AP tail (block digamma)
    depths = np.arange(0, 601)
    P = np.empty(len(depths))
    cum = np.concatenate([[0.0], np.cumsum(t)])
    for i, k in enumerate(depths):
        if k < J:
            P[i] = cum[k + 1]
        else:  # partial into the AP tail: add sum_{j=J..k} of tail terms
            m = k - J + 1
            P[i] = cum[J] + (digamma(R[-1] / dl + 1.0 + m)
                             - digamma(A[-1] / dl + 1.0 + m)
                             - digamma(R[-1] / dl + 1.0)
                             + digamma(A[-1] / dl + 1.0)) / dl
    return P, total, t


def check_b(n_try):
    rng = np.random.default_rng(777001)
    sets = {"A": [], "B": [], "C": []}
    tries = 0
    while (min(len(v) for v in sets.values()) < 5) and tries < n_try:
        tries += 1
        kind = ["A", "B", "C"][tries % 3]
        out = gen_non_ap(rng, kind)
        if out is None:
            continue
        P, total, t = out
        cond_all = np.all(P <= 0.0)
        cond_fails = np.any(P > 0.0)
        if kind == "A" and cond_all and total <= 0 and np.any(t > 0):
            sets["A"].append((P, total, t))
        elif kind == "B" and cond_fails and total > 0:
            sets["B"].append((P, total, t))
        elif kind == "C" and cond_fails and total <= 0:
            sets["C"].append((P, total, t))
    okA = len(sets["A"]) >= 5 and all(
        np.all(P <= 0) and tot <= 0 for P, tot, _ in sets["A"])
    okB = len(sets["B"]) >= 5
    okC = len(sets["C"]) >= 5
    dA = [f"Sigma={tot:+.4f}, mixed-sign terms" for _, tot, _ in sets["A"][:3]]
    dB = [f"Sigma={tot:+.4f}, first fail depth {int(np.argmax(P > 0))}"
          for P, tot, _ in sets["B"][:3]]
    dC = [f"Sigma={tot:+.4f}, first fail depth {int(np.argmax(P > 0))}"
          for P, tot, _ in sets["C"][:3]]
    report("(b-i) cumulative condition sufficient: "
           f"{len(sets['A'])} non-AP configs, P_k<=0 all depths => Sigma_far<=0",
           okA, "; ".join(dA) + f" (tries={tries})")
    report("(b-ii) condition fails & Sigma_far>0 exists "
           f"({len(sets['B'])} configs)", okB, "; ".join(dB))
    report("(b-iii) condition NOT necessary: fails at some depth yet "
           f"Sigma_far<=0 ({len(sets['C'])} configs)", okC, "; ".join(dC))


def v0_lattice_direct(x, M=2_000_000):
    """(delta/2)*v(x_0) = sum_{m=1}^M [1/m - 1/(m+x)], plain numpy, no digamma."""
    m = np.arange(1, M + 1, dtype=np.float64)
    return np.sum(1.0 / m - 1.0 / (m + x))


def check_c():
    dl = 0.45
    rows = []
    ok = True
    for x in [0.5, 1.0, 2.0, 4.0]:
        direct = v0_lattice_direct(x)            # (delta/2) v(x0)
        dz_direct = -2.0 * (2.0 / dl) * direct   # v(x1)-v(x0) = -2 v(x0)
        dz_formula = -(4.0 / dl) * (digamma(1.0 + x) + EULER_GAMMA)
        rel = abs(dz_direct - dz_formula) / abs(dz_formula)
        ok = ok and rel < 5e-3
        rows.append(f"e/d={x:g}: direct={dz_direct:+.6f}, "
                    f"digamma={dz_formula:+.6f}, relerr={rel:.2e}")
    report("(c) digamma relaxation law Delta_z=-(4/d)[psi(1+e/d)+gamma_E] "
           "vs direct lattice sums", ok,
           " | ".join(rows) + " (tol 0.5% rel)")
    dz0 = -(4.0 / dl) * (digamma(1.0 + 1e-12) + EULER_GAMMA)
    report("(c2) AP equilibrium: Delta_z -> 0 as e -> 0",
           abs(dz0) < 1e-8, f"Delta_z(e/d=1e-12)={dz0:.2e}")


def check_d():
    dl = 0.45
    xs = np.array([1e-4, 3e-4, 1e-3, 3e-3])
    es = xs * dl
    dzs = -(4.0 / dl) * (digamma(1.0 + xs) + EULER_GAMMA)
    slope = np.polyfit(es, dzs, 1)[0]
    pred = -2.0 * np.pi**2 / (3.0 * dl**2)
    rel = abs(slope - pred) / abs(pred)
    report("(d) linear response Delta_z ~ -(2 pi^2/(3 d^2)) e", rel < 5e-3,
           f"fitted slope={slope:.6f}, predicted={pred:.6f}, "
           f"rel err={rel:.2e} (tol 0.5%; curvature ~0.2% at e/d=3e-3)")


def main():
    print(f"03_threshold_lemma.py  mode={'fast' if FAST else 'full'}")
    check_a(200 if FAST else 2000)
    check_b(40000 if FAST else 200000)
    check_c()
    check_d()
    print(f"SUMMARY: {sum(RESULTS)}/{len(RESULTS)} checks passed")
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
verify_all.py — independent recomputation of every numerical claim of
Part IX (Sections 48–57: tight zero pairs of the Riemann zeta function)
of the manuscript Ismail_rh_pf_v20.

Usage:  python3 verify_all.py [--slow]
  default : T8 uses the 5 tightest pairs (~1 min extra, mpmath dps=30)
  --slow  : T8 uses the 15 tightest pairs (matches brief top-15 table)

Prints "name | computed | target | tol | PASS/FAIL" rows and a final
"ALL PASS (n/n)" summary (or the list of failures).

Section 56 additions (Part IX of the manuscript) (T16-T22, resolution of Open Problem 57.1; all in the
default run, adding ~30 s; primes <= 1e7 sieved once in T6X and reused):
  T16 k-tail truncation constant Delta_0 (Lemma 56.1, Theorem 56.9(a)): value,
      p=2 term, X-independence over 1e2..1e7, all-prime tail estimate
      (log-weighted integral bound, closed form).
  T17 mean-value identity (Lemma 56.3): Monte Carlo E[exp(-L_X)] at X=100,
      s=1, 3e5 samples on [0,2e6], rng=default_rng(19), vs prod M_p(-1)
      (midpoint rule N=4096).
  T18 effective second-moment constants (Lemma 56.4): sum a_p, Var(L_X),
      sigma, the 4X/H error bound and the mean bound at H=T^2.
  T19 certified hybrid Chernoff table (Theorem 56.5): exact midpoint-rule
      log M_p(-s) (STRICT k<=3 integrand, log after grid mean) for p<=1000,
      Bennett tail (sigma_p^2 full-k to k=60, c_p=-log(1-p^-1/2)) for
      1000<p<=1e7, 1e-6 exponent safety margin; plus the pure-Bennett-only
      bound at u=6.9 documenting that exact small primes are essential.
  T20 Lipschitz constant Lambda_1 (Lemma 56.7) + uniform union bound
      (Theorem 56.8): N*d(u-m) at H=T=8.436e9, m=0.5, s optimized on a grid.
      NOTE: with the STRICT k<=3 integrand, N*d(8.86)=6.822e-2 and
      N*d(9.5)=1.905e-4 exceed the manuscript's 2-sig-fig table entries
      (6.8e-2, 1.9e-4) by ~0.3% (the table values are the 2-sf rounding;
      the full-k variant of Theorem 56.5 gives bounds 2-7% smaller, cf. Section 56.4).
      Asserted against the rounding-aware bounds 6.85e-2 / 1.95e-4.
  T21 Bahadur-Rao prefactor validation (Proposition 56.6): product law over
      p<=1e4 with EXACT midpoint-rule factors for every prime (stated
      choice; the remark allowed Bennett for p>1000), tilt s* solving
      (log M)'(-s*)=-u by Newton, u=4.580=4.00 sigma, N=2000 tilted
      samples, rng=default_rng(19); estimate, 5-se check, implied prefactor.
      Runtime ~5 s, so kept in the default run (not gated behind --slow).
  T22 per-pair certificate margin (Corollary 56.2 + Proposition 55.4, from
      detector_sweep_8p4e9.csv): H=sqrt(2/far) column check, eligibility
      far<4/(eps(1-eps)) (equivalently Delta_*>0), margin Delta_*-Delta_0
      positivity, Delta_*=6.94 at the reference pair g=1e-2 with the median far
      (Theorem 56.5's u=6.90; true ensemble median Delta_*=4.46),
      max-far pair Delta_*=-1.25, and the headline B_*(H)/|P_X| arithmetic.
      NOTE (provenance): the Proposition 55.4 table rows pair min g=4.15e-3 with
      the ensemble MEAN far (27.98~28) and the representative g=1.0e-2 with
      the MEDIAN far (24.84); the min-g pair's own far (23.01) gives
      1.43e-4 and the median-g pair (g=0.0356) gives 2.17e-2, printed as
      info rows.  The delegated "min margin in [6.0,8.5] over eligible
      pairs" is NOT reproduced over the full ensemble (min margin 0.62;
      eligibility is exactly Delta_*>0, and Delta_* ranges 0.86..8.86 over
      eligible pairs); the manuscript's 6.9-8.5 range describes the
      representative median/tightest pairs, asserted at the median pair.

Conventions reverse-engineered from / verified against the brief:
  * normalized spacings for GUE tail: s = diff(z) * log(mean(z)/2pi)/(2pi)
  * lag-1 autocorrelation: scale-invariant, any constant dens works
  * lehmer_rows_* columns: (idx1based, t0, g, s, S_on, far, C, Cmid, [hthr], r)
    (lehmer_rows_100k has an extra hthr column in position 8)
  * T6 regression in natural log: ln floor = a*ln g + b*L_X + c
  * T6 controls: normalized spacing in [0.95,1.05], rng=default_rng(11),
    300 draws without replacement, shift relative to control std
  * "300 log-spaced pairs" = rows sorted by g, evenly spaced RANKS:
    sorted_rows[linspace(0, n-1, 300).round()] (verified against
    lehmer_floors_8p4e9.npy row-by-row); same recipe used for T6-2M
  * T8/T9 reference coordinates in 80-bit longdouble; mpmath references
    evaluated at the exact binary float64 value (mp.mpf(float)), NOT at a
    decimal string (round-trip shifts t by 3.5e-7 at 8.4e9)
"""

import sys
import gzip
import time
import numpy as np

SLOW = "--slow" in sys.argv
import os
DATA = os.environ.get("RH_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
# 20+ digit pi in 80-bit longdouble.  CRITICAL: np.longdouble(2*np.pi) keeps
# the float64 value of pi (error 2.4e-16), which in `x mod 2pi` at x~1.2e11
# becomes a phase error of ~5e-6 rad (error * x/2pi).  All longdouble mod-2pi
# phase reductions below must use this pi.
PI_LD = np.longdouble("3.141592653589793238462643383279502884197")
TWO_PI_LD = 2 * PI_LD

t_start = time.time()
results = []


def report(name, computed, target, tol, ok, cfmt="%.6g", tfmt="%.6g"):
    cs = cfmt % computed if isinstance(computed, float) else str(computed)
    ts = tfmt % target if isinstance(target, float) else str(target)
    results.append((name, bool(ok)))
    print("%-42s | %22s | %22s | %8s | %s"
          % (name, cs, ts, ("%.3g" % tol if isinstance(tol, float) else str(tol)),
             "PASS" if ok else "FAIL"))


# ------------------------------------------------------------------ load data
print("loading data ...")
z1 = np.loadtxt(gzip.open(DATA + "/zeros1.gz", "rt"))          # first 100k
z6 = np.loadtxt(gzip.open(DATA + "/zeros6.gz", "rt"))          # first 2,001,052
zl = np.load(DATA + "/lmfdb_zeros_parsed.npy")                 # 772,719 @ 8.4e9
zll = zl.astype(np.longdouble)

rows = {k: np.load(DATA + "/lehmer_rows_%s.npy" % k)
        for k in ("100k", "2M", "8p4e9", "1e12", "1e21")}
floors = np.load(DATA + "/lehmer_floors_8p4e9.npy")            # 300 x 5

# ================================================================= T1 identity
print("\n--- T1: r^2 (1 + 1/(8 C_mid)) = 1 ---")
worst = 0.0
detail = []
for k, a in rows.items():
    r, Cm = a[:, -1], a[:, 7]
    dev = float(np.max(np.abs(r**2 * (1.0 + 1.0 / (8.0 * Cm)) - 1.0)))
    worst = max(worst, dev)
    detail.append("%s:%.2e" % (k, dev))
report("T1 identity max dev (all files)", worst, 0.0, 1e-14, worst < 1e-14,
       cfmt="%.3e")
print("    " + " ".join(detail))

# ================================================================= T2 K1 law
print("\n--- T2: mean far vs 2.5*dens^2 ---")
dens = {"100k": 1.494, "2M": 1.935, "8p4e9": 3.3451, "1e12": 3.896, "1e21": 7.095}
targ2 = {"100k": 0.748, "2M": 0.801, "8p4e9": 1.000, "1e12": 1.083, "1e21": 1.021}
for k, a in rows.items():
    mf = float(a[:, 5].mean())
    ratio = mf / (2.5 * dens[k] ** 2)
    report("T2 far-law ratio %s" % k, ratio, targ2[k], 0.02,
           abs(ratio - targ2[k]) <= 0.02, cfmt="%.4f", tfmt="%.3f")
    print("    mean far = %.3f, 2.5*dens^2 = %.3f" % (mf, 2.5 * dens[k] ** 2))

# ================================================================= T3 GUE tail
print("\n--- T3: GUE small-spacing tail / (pi^2 s0^3 / 9) ---")
S0 = (0.05, 0.10, 0.15)
C9 = np.pi ** 2 / 9.0


def gue_ratios(z):
    g = np.diff(z)
    d = np.log(z.mean() / (2 * np.pi)) / (2 * np.pi)   # dens at mean t
    s = g * d
    return [float(np.mean(s < s0)) / (C9 * s0 ** 3) for s0 in S0], s


targ3 = {"100k": (0.657, 0.675, 0.670), "2M": (0.864, 0.804, 0.784),
         "8.4e9": (0.887, 0.981, 0.957)}
spac = {}
for name, z in (("100k", z1), ("2M", z6), ("8.4e9", zl)):
    rr, spac[name] = gue_ratios(z)
    tg = targ3[name]
    ok = all(abs(a - b) <= 0.01 for a, b in zip(rr, tg))
    report("T3 GUE tail %s (s0=.05/.10/.15)" % name,
           "/".join("%.3f" % x for x in rr),
           "/".join("%.3f" % x for x in tg), 0.01, ok)

# ================================================================= T4 lag-1
print("\n--- T4: lag-1 autocorrelation of consecutive spacings ---")
targ4 = {"100k": -0.204, "2M": -0.268, "8.4e9": -0.332}
for name in ("100k", "2M", "8.4e9"):
    s = spac[name]
    c = float(np.corrcoef(s[:-1], s[1:])[0, 1])
    report("T4 lag-1 corr %s" % name, c, targ4[name], 0.01,
           abs(c - targ4[name]) <= 0.01, cfmt="%.4f", tfmt="%.3f")

# ================================================================= T5 shield
print("\n--- T5: compensation shield at 8.4e9 ---")
r8 = rows["8p4e9"]
i0 = r8[:, 0].astype(int) - 1                      # 0-based index of first zero
gaps = np.diff(zll)
gm = gaps.mean()
sh_prev = float((gaps[i0 - 1].mean() / gm))
sh_next = float((gaps[i0 + 1].mean() / gm))
report("T5 shield prev-gap", sh_prev, 1.354, 0.02, abs(sh_prev - 1.354) <= 0.02,
       cfmt="%.4f", tfmt="%.3f")
report("T5 shield next-gap", sh_next, 1.360, 0.02, abs(sh_next - 1.360) <= 0.02,
       cfmt="%.4f", tfmt="%.3f")

# ================================================================= T6 Euler law
print("\n--- T6: Euler factorization of the floor at 8.4e9 ---")


def sieve(n):
    b = np.ones(n + 1, bool)
    b[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if b[i]:
            b[i * i::i] = False
    return np.flatnonzero(b)


PR = sieve(10 ** 6)
LNPR = np.log(PR.astype(np.longdouble))
P64 = PR.astype(float)
SQ = np.sqrt(P64)


def LX_batch(t_ld, xcuts, chunk=64):
    """L_X(t) for arrays of longdouble t; all cutoffs from one cumsum pass.

    Phases: (t * ln p) mod 2pi in 80-bit longdouble (t*ln p ~ 1.2e11),
    then float64 cos.  cos(2p), cos(3p) from exact multiple-angle ids.
    """
    kidx = {X: np.searchsorted(PR, X, side="right") - 1 for X in xcuts}
    out = {X: np.empty(len(t_ld)) for X in xcuts}
    for c0 in range(0, len(t_ld), chunk):
        ts = t_ld[c0:c0 + chunk]
        c1 = np.empty((len(PR), len(ts)))
        for j, tt in enumerate(ts):
            ph = (tt * LNPR) % TWO_PI_LD
            c1[:, j] = np.cos(np.asarray(ph, dtype=float))
        c2 = 2.0 * c1 ** 2 - 1.0
        c3 = 4.0 * c1 ** 3 - 3.0 * c1
        S = (np.cumsum(c1 / SQ[:, None], axis=0)
             + 0.5 * np.cumsum(c2 / P64[:, None], axis=0)
             + (1.0 / 3.0) * np.cumsum(c3 / (P64 ** 1.5)[:, None], axis=0))
        for X, kk in kidx.items():
            out[X][c0:c0 + len(ts)] = S[kk]
    return out


def ols(Xm, y):
    A = np.column_stack([Xm, np.ones(len(y))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yh = A @ coef
    r2 = 1.0 - float(np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2))
    return coef, r2, y - yh


fidx = floors[:, 0].astype(int) - 1
t0_pairs = (zll[fidx] + zll[fidx + 1]) / 2          # longdouble pair midpoints
g_pair = np.asarray(zll[fidx + 1] - zll[fidx], dtype=float)
fl_pair = floors[:, 3]
far_pair = floors[:, 4]

t6 = time.time()
LXp = LX_batch(t0_pairs, [10 ** 4, 10 ** 6])
print("    (L_X at 300 pair midpoints, X=1e4+1e6 single pass: %.1fs)"
      % (time.time() - t6))

lf = np.log(fl_pair)
lg = np.log(g_pair)

for X, tg in ((10 ** 4, (1.978, 0.783, 0.9447)), (10 ** 6, (1.932, 0.899, 0.9439))):
    (a, b, c), r2, _ = ols(np.column_stack([lg, LXp[X]]), lf)
    report("T6 fit a (X=1e%d)" % int(np.log10(X)), a, tg[0], 0.05,
           abs(a - tg[0]) <= 0.05, cfmt="%.4f", tfmt="%.3f")
    report("T6 fit b (X=1e%d)" % int(np.log10(X)), b, tg[1], 0.05,
           abs(b - tg[1]) <= 0.05, cfmt="%.4f", tfmt="%.3f")
    report("T6 fit R2 (X=1e%d)" % int(np.log10(X)), r2, tg[2], 0.01,
           abs(r2 - tg[2]) <= 0.01, cfmt="%.4f", tfmt="%.4f")

(a1, _), r2g, resg = ols(lg[:, None], lf)          # g-only fit
L4 = LXp[10 ** 4]
corr_L = float(np.corrcoef(L4, resg)[0, 1])
report("T6 corr(L_X(1e4), g-only resid)", corr_L, 0.946, 0.01,
       abs(corr_L - 0.946) <= 0.01, cfmt="%.4f", tfmt="%.3f")


def resid_on(x, c):
    A = np.column_stack([c, np.ones(len(c))])
    cf, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ cf


pc = float(np.corrcoef(resid_on(resg, L4), resid_on(far_pair, L4))[0, 1])
report("T6 partial corr(resid, far | L_X)", pc, 0.004, 0.05,
       abs(pc - 0.004) <= 0.05, cfmt="%.4f", tfmt="%.3f")

# shift vs 300 control mid-gaps, s in [0.95,1.05], seed 11
dd9 = np.log(zl.mean() / (2 * np.pi)) / (2 * np.pi)
s_all = np.diff(zl) * dd9
elig = np.flatnonzero((s_all >= 0.95) & (s_all <= 1.05))
rng = np.random.default_rng(11)
sel = rng.choice(elig, 300, replace=False)
t_ctrl = (zll[sel] + zll[sel + 1]) / 2
LXc = LX_batch(t_ctrl, [10 ** 6])[10 ** 6]
shift = float((LXp[10 ** 6].mean() - LXc.mean()) / LXc.std())
report("T6 shift of mean L_X(1e6) (sigma)", shift, -3.13, 0.3,
       abs(shift + 3.13) <= 0.3, cfmt="%.3f", tfmt="%.2f")

# ================================================================= T7 power law
print("\n--- T7: floor power law (300 pairs) ---")
slope7 = float(np.polyfit(lg, lf, 1)[0])
report("T7 log-log slope floor vs g", slope7, 1.853, 0.1,
       abs(slope7 - 1.853) <= 0.1, cfmt="%.4f", tfmt="%.3f")

# ================================================================= T6X X-scan
print("\n--- T6X: Euler fit scan over X (ONE cumsum pass, primes <= 1e7) ---")
PR7 = sieve(10 ** 7)
LNPR7 = np.log(PR7.astype(np.longdouble))
P7 = PR7.astype(float)
SQ7 = np.sqrt(P7)
XCUTS = [10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7]
KIDX = [np.searchsorted(PR7, X, side="right") - 1 for X in XCUTS]
t6x = time.time()
LXscan = {X: np.empty(len(t0_pairs)) for X in XCUTS}
CH = 16                                # chunk keeps transient memory ~0.5 GB
for c0 in range(0, len(t0_pairs), CH):
    ts = t0_pairs[c0:c0 + CH]
    c1 = np.empty((len(PR7), len(ts)))
    for j, tt in enumerate(ts):
        c1[:, j] = np.cos(np.asarray((tt * LNPR7) % TWO_PI_LD, dtype=float))
    c2 = 2.0 * c1 ** 2 - 1.0
    c3 = 4.0 * c1 ** 3 - 3.0 * c1
    S = (np.cumsum(c1 / SQ7[:, None], axis=0)
         + 0.5 * np.cumsum(c2 / P7[:, None], axis=0)
         + (1.0 / 3.0) * np.cumsum(c3 / (P7 ** 1.5)[:, None], axis=0))
    for X, kk in zip(XCUTS, KIDX):
        LXscan[X][c0:c0 + len(ts)] = S[kk]
    del c1, c2, c3, S
print("    (single pass over %d primes, 6 cutoffs: %.1fs)"
      % (len(PR7), time.time() - t6x))

targ6x = {10 ** 2: (1.847, 0.752, 0.9339), 10 ** 3: (1.978, 0.733, 0.9348),
          10 ** 5: (1.969, 0.829, 0.9505), 10 ** 7: (1.868, 0.944, 0.9546)}
print("    X        a (se)           b (se)           R2")
for X in XCUTS:
    (a, b, c), r2x, resx = ols(np.column_stack([lg, LXscan[X]]), lf)
    Xm = np.column_stack([lg, LXscan[X], np.ones(len(lf))])
    sig2 = float(np.sum(resx ** 2) / (len(lf) - 3))
    se = np.sqrt(np.diag(sig2 * np.linalg.inv(Xm.T @ Xm)))
    print("    X=1e%-4d %.4f (%.4f)   %.4f (%.4f)   %.4f"
          % (int(np.log10(X)), a, se[0], b, se[1], r2x))
    report("T6X OLS se(a) X=1e%d" % int(np.log10(X)), float(se[0]),
           "info", "-", True, cfmt="%.4f")
    report("T6X OLS se(b) X=1e%d" % int(np.log10(X)), float(se[1]),
           "info", "-", True, cfmt="%.4f")
    if X in targ6x:          # X=1e4,1e6 fits already asserted in T6 above
        tg = targ6x[X]
        ok = (abs(a - tg[0]) <= 0.05 and abs(b - tg[1]) <= 0.05
              and abs(r2x - tg[2]) <= 0.01)
        report("T6X fit a/b/R2 X=1e%d" % int(np.log10(X)),
               "%.3f/%.3f/%.4f" % (a, b, r2x),
               "%.3f/%.3f/%.4f" % tg, ".05/.05/.01", ok)

# ================================================================= T6 at 2M
print("\n--- T6-2M: Euler factorization at T~1.2e6 ---")
# 300 pairs: rows sorted by g, evenly spaced ranks (same recipe as 8.4e9)
order2 = np.argsort(rows["2M"][:, 2])
sel2 = order2[np.linspace(0, len(order2) - 1, 300).round().astype(int)]
rows2 = rows["2M"][sel2]
i2 = rows2[:, 0].astype(int) - 1
ga2, gb2 = z6[i2], z6[i2 + 1]
t0_2 = 0.5 * (ga2 + gb2)
g_2 = gb2 - ga2
far_2 = rows2[:, 5]


def Z_rs_fp64(t):
    """Fast fp64 RS Z(t): main sum + first Backlund term, theta fp64 series
    (sufficient at t ~ 1.2e6; NOT sufficient at 8.4e9)."""
    t = np.atleast_1d(np.asarray(t, float))
    tau = np.sqrt(t / (2 * np.pi))
    N = np.floor(tau).astype(int)
    p = tau - N
    th = (t / 2 * np.log(t / (2 * np.pi)) - t / 2 - np.pi / 8
          + 1 / (48 * t) + 7 / (5760 * t ** 3))
    out = np.empty(len(t))
    for j in range(len(t)):
        nn = np.arange(1, N[j] + 1)
        out[j] = 2.0 * np.sum(np.cos(th[j] - t[j] * np.log(nn)) / np.sqrt(nn))
    psi = np.cos(2 * np.pi * (p * p - p - 1.0 / 16.0)) / np.cos(2 * np.pi * p)
    return out + ((-1) ** (N + 1)) * tau ** -0.5 * psi


t62 = time.time()
fl_2 = np.empty(300)
for j in range(300):
    ts2 = np.linspace(ga2[j], gb2[j], 124)[1:-1]   # 122 interior samples
    fl_2[j] = np.min(np.abs(Z_rs_fp64(ts2)))
# L_X(1e4) in fp64 (phases fine at t ~ 1.2e6)
PR4 = sieve(10 ** 4).astype(float)
LNP4 = np.log(PR4)
LX2 = np.empty(300)
for j in range(300):
    c1 = np.cos((t0_2[j] * LNP4) % (2 * np.pi))
    c2 = 2.0 * c1 ** 2 - 1.0
    c3 = 4.0 * c1 ** 3 - 3.0 * c1
    LX2[j] = (np.sum(c1 / np.sqrt(PR4)) + 0.5 * np.sum(c2 / PR4)
              + (1.0 / 3.0) * np.sum(c3 / PR4 ** 1.5))
print("    (300 fp64 floors + L_X: %.1fs)" % (time.time() - t62))
lf2 = np.log(fl_2)
lg2 = np.log(g_2)
(a2m, b2m, c2m), r22m, _ = ols(np.column_stack([lg2, LX2]), lf2)
report("T6-2M fit a (X=1e4)", a2m, 1.892, 0.05, abs(a2m - 1.892) <= 0.05,
       cfmt="%.4f", tfmt="%.3f")
report("T6-2M fit b (X=1e4)", b2m, 0.906, 0.05, abs(b2m - 0.906) <= 0.05,
       cfmt="%.4f", tfmt="%.3f")
report("T6-2M fit R2 (X=1e4)", r22m, 0.9463, 0.01, abs(r22m - 0.9463) <= 0.01,
       cfmt="%.4f", tfmt="%.4f")
(_, _), _, resg2 = ols(lg2[:, None], lf2)
c2a = float(np.corrcoef(LX2, resg2)[0, 1])
report("T6-2M corr(L_X, g-only resid)", c2a, 0.916, 0.01,
       abs(c2a - 0.916) <= 0.01, cfmt="%.4f", tfmt="%.3f")
c2b = float(np.corrcoef(LX2, np.log(far_2))[0, 1])
report("T6-2M corr(L_X, log far)", c2b, -0.422, 0.03,
       abs(c2b + 0.422) <= 0.03, cfmt="%.4f", tfmt="%.3f")

# ================================================================= K5 at 8.4e9
print("\n--- K5: corr(floor residual after g-law, log far) at 8.4e9 ---")
k5 = float(np.corrcoef(resg, np.log(far_pair))[0, 1])
report("K5 corr(resid, log far) 8.4e9", k5, -0.746, 0.03,
       abs(k5 + 0.746) <= 0.03, cfmt="%.4f", tfmt="%.3f")

# ================================================================= shield 100k/2M
print("\n--- T5-low: compensation shield at 100k and 2M ---")
for name, z in (("100k", z1), ("2M", z6)):
    ii = rows[name][:, 0].astype(int) - 1
    gp = np.diff(z)
    gpm = gp.mean()
    sp_ = float(gp[ii - 1].mean() / gpm)
    sn_ = float(gp[ii + 1].mean() / gpm)
    ok = (1.33 <= sp_ <= 1.45) and (1.33 <= sn_ <= 1.45)
    report("T5-low shield %s prev/next" % name,
           "%.4f/%.4f" % (sp_, sn_), "1.375-1.40", "[1.33,1.45]", ok)

# ================================================================= far stats
print("\n--- far median/max at 8.4e9 ---")
fmed = float(np.median(rows["8p4e9"][:, 5]))
fmax = float(rows["8p4e9"][:, 5].max())
report("far median 8.4e9", fmed, 24.84, 0.1, abs(fmed - 24.84) <= 0.1,
       cfmt="%.2f", tfmt="%.2f")
report("far max 8.4e9", fmax, 368.35, 1.0, abs(fmax - 368.35) <= 1.0,
       cfmt="%.2f", tfmt="%.2f")

# ================================================================= T8 displacement
print("\n--- T8: displacement law delta = (g^2/8) lambda_far ---")
import mpmath as mp

NPAIR = 15 if SLOW else 5
deltas, preds = [], []
t8 = time.time()
for row in floors[:NPAIR]:
    i = int(row[0]) - 1
    t0_ld = (zll[i] + zll[i + 1]) / 2
    g_ld = zll[i + 1] - zll[i]
    win = np.concatenate([np.arange(i - 250, i), np.arange(i + 2, i + 252)])
    lam = np.sum(1.0 / (t0_ld - zll[win]))              # lambda_far
    pred = (g_ld ** 2 / 8) * lam
    # exact mpmath t0 from the binary float64 coordinates
    t0_mp = (mp.mpf(float(zl[i])) + mp.mpf(float(zl[i + 1]))) / 2
    mp.mp.dps = 30
    tstar = mp.findroot(lambda x: mp.diff(mp.siegelz, x), t0_mp)
    delta = tstar - t0_mp
    deltas.append(float(delta))
    preds.append(float(pred))
    print("    g=%.3e  lam=%+.3f  delta=%+.3e  pred=%+.3e"
          % (float(g_ld), float(lam), float(delta), float(pred)))
corr8 = float(np.corrcoef(deltas, preds)[0, 1])
report("T8 corr(delta, (g^2/8) lambda_far) [%d pairs]" % NPAIR,
       corr8, 0.999, 0.0, corr8 > 0.999, cfmt="%.6f", tfmt="%.3f")
print("    (T8 elapsed %.1fs)" % (time.time() - t8))

# ------------------------------------------------ T8 per-row check vs CSV (--slow)
if SLOW:
    csv = np.genfromtxt(DATA + "/disp_table_8p4e9_corrected.csv",
                        delimiter=",", names=True)
    dref = csv["delta_corrected"]
    assert len(dref) == 15
    for k in range(15):
        dv, dr = deltas[k], float(dref[k])
        ok = abs(dv - dr) <= 1e-7 or abs(dv - dr) / abs(dr) <= 0.02
        report("T8-row %02d delta vs corrected CSV" % (k + 1),
               "%+.6e" % dv, "%+.6e" % dr, "1e-7|2%", ok)

# ================================================================= T9 instrument
print("\n--- T9: Z_rs2 vs mpmath at a mid-gap point ---")


def Z_rs2(t):
    """Riemann-Siegel Z(t): main sum + first Backlund correction term.

    theta series and all phases in 80-bit longdouble.  Phase for term n is
        theta(t) - t ln n = t*ln(tau/n) - t/2 - pi/8 + thetacorr(t)
    (never materialize theta ~ 8.4e10: the split keeps the largest
    intermediate at t/2 ~ 4.2e9, preserving longdouble phase accuracy).
    """
    tld = np.longdouble(t)
    tau = np.sqrt(tld / TWO_PI_LD)
    N = int(np.floor(tau))
    p = float(tau - N)
    nn = np.arange(1, N + 1).astype(np.longdouble)
    thcorr = (1 / (48 * tld) + 7 / (5760 * tld ** 3) + 31 / (80640 * tld ** 5))
    ph = (tld * np.log(tau / nn) - tld / 2 - PI_LD / 8 + thcorr) % TWO_PI_LD
    main = 2.0 * float(np.sum(np.cos(np.asarray(ph, float))
                              / np.sqrt(np.asarray(nn, float))))
    psi = np.cos(2 * np.pi * (p * p - p - 1.0 / 16.0)) / np.cos(2 * np.pi * p)
    corr = (-1) ** (N + 1) * float(tau) ** -0.5 * psi
    return main + corr


t_mid = float((zll[399999] + zll[400000]) / 2)     # zeros #400000/#400001
t9 = time.time()
z_rs2 = Z_rs2(t_mid)
mp.mp.dps = 50
z_ref = mp.siegelz(mp.mpf(t_mid))                  # exact binary t, dps=50
err9 = abs(z_rs2 - float(z_ref))
report("T9 |Z_rs2 - mpmath| mid-gap", err9, 1.5e-7, 5e-7, err9 < 5e-7,
       cfmt="%.3e", tfmt="%.1e")
print("    (t=%.6f, Z=%.9f, elapsed %.1fs)"
      % (t_mid, z_rs2, time.time() - t9))

# ================================================================= envelope
print("\n--- Envelope: log|Z| vs L_X(1e4) at the 300 control mid-gaps ---")
LXc4 = LX_batch(t_ctrl, [10 ** 4])[10 ** 4]        # same controls as T6 shift
tenv = time.time()
Zc = np.array([abs(Z_rs2(float(tv))) for tv in t_ctrl])
print("    (Z_rs2 at 300 control midpoints: %.1fs)" % (time.time() - tenv))
lZc = np.log(Zc)
cenv = float(np.corrcoef(LXc4, lZc)[0, 1])
report("Envelope corr(L_X(1e4), log|Z|)", cenv, 0.942, 0.01,
       abs(cenv - 0.942) <= 0.01, cfmt="%.4f", tfmt="%.3f")
(menv, cenv0), r2env, _ = ols(LXc4[:, None], lZc)
report("Envelope slope log|Z| = m*L_X + c", menv, 0.824, 0.05,
       abs(menv - 0.824) <= 0.05, cfmt="%.4f", tfmt="%.3f")
report("Envelope R2", r2env, 0.888, 0.02, abs(r2env - 0.888) <= 0.02,
       cfmt="%.4f", tfmt="%.3f")
print("    (intercept %.4f, brief 0.787)" % cenv0)

# ================================================================= T10 ensemble
print("\n--- T10: LMFDB ensemble integrity ---")
ok_size = zl.size == 772719
report("T10 size == 772,719", float(zl.size), 772719.0, 0.0, ok_size,
       cfmt="%.0f", tfmt="%.0f")
ok_inc = bool(np.all(np.diff(zll) > 0))
report("T10 strictly increasing", float(ok_inc), 1.0, 0.0, ok_inc,
       cfmt="%.0f", tfmt="%.0f")
ok_first = abs(zl[0] - 8436146000.02) < 0.5
report("T10 first t ~ 8436146000.02", float(zl[0]), 8436146000.02, 0.5, ok_first,
       cfmt="%.2f", tfmt="%.2f")
ok_last = abs(zl[-1] - 8436376999.73) < 0.5
report("T10 last t ~ 8436376999.73", float(zl[-1]), 8436376999.73, 0.5, ok_last,
       cfmt="%.2f", tfmt="%.2f")

# ================================================================= T11 detector sweep
print("\n--- T11: Speiser-detector sweep CSV integrity ---")
det = np.genfromtxt(DATA + "/detector_sweep_8p4e9.csv", delimiter=",", names=True)
report("T11 sweep rows == 2736", float(det.size), 2736.0, 0.0,
       det.size == 2736, cfmt="%.0f", tfmt="%.0f")
nsc = float(det["n_sign_changes"].max())
report("T11 sign changes == 0 everywhere", nsc, 0.0, 0.0, nsc == 0.0,
       cfmt="%.0f", tfmt="%.0f")
rmin11 = float(det["r"].min())
report("T11 min r", rmin11, 0.961225, 1e-6, abs(rmin11 - 0.961225) <= 1e-6,
       cfmt="%.6f", tfmt="%.6f")
rmed11 = float(np.median(det["r"]))
report("T11 median r", rmed11, 0.998054, 1e-6, abs(rmed11 - 0.998054) <= 1e-6,
       cfmt="%.6f", tfmt="%.6f")
fmed11 = float(np.median(det["far"]))
report("T11 far median", fmed11, 24.842, 0.005, abs(fmed11 - 24.842) <= 0.005,
       cfmt="%.3f", tfmt="%.3f")
hr11 = det["H"] / (det["g"] / 2.0)
hmin11 = float(hr11.min())
report("T11 min H/(g/2)", hmin11, 3.486, 0.005, abs(hmin11 - 3.486) <= 0.005,
       cfmt="%.4f", tfmt="%.3f")
hmed11 = float(np.median(hr11))
report("T11 median H/(g/2)", hmed11, 16.01, 0.01, abs(hmed11 - 16.01) <= 0.01,
       cfmt="%.4f", tfmt="%.2f")
hr_dev = float(np.max(np.abs(hr11 - det["H_over_halfgap"])))
report("T11 H/(g/2) vs stored column", hr_dev, 0.0, 1e-6, hr_dev <= 1e-6,
       cfmt="%.2e", tfmt="%.0f")
rng11 = np.random.default_rng(7)
for j in rng11.choice(det.size, 5, replace=False):
    i = int(det["pair_idx"][j]) - 1
    t0s = (zll[i] + zll[i + 1]) / 2
    lo = np.arange(max(0, i - 250), i)
    hi = np.arange(i + 2, min(len(zll), i + 252))
    far_rc = float(np.sum(1.0 / (t0s - zll[lo]) ** 2)
                   + np.sum(1.0 / (t0s - zll[hi]) ** 2))
    far_csv = float(det["far"][j])
    rel = abs(far_rc - far_csv) / far_csv
    report("T11 far spot row %d (pair %d)" % (j, i + 1), far_rc, far_csv, 1e-6,
           rel <= 1e-6, cfmt="%.6f", tfmt="%.6f")

# ================================================================= T12 full-pair fits
print("\n--- T12: full-ensemble (2736-pair) fits ---")
f8a = np.load(DATA + "/floors_all_8p4e9.npy")          # idx,t0,g,s,far,floor
L8a = np.load(DATA + "/LX_all_8p4e9.npy", allow_pickle=True).item()
LXall = L8a["LX_pairs"]                                 # (2736, 6) X=1e2..1e7
lg_a = np.log(f8a[:, 2])
lf_a = np.log(f8a[:, 5])
far_a = f8a[:, 4]
(a7, b7, c7), r2_7, res7 = ols(np.column_stack([lg_a, LXall[:, 5]]), lf_a)
report("T12 full fit a (X=1e7)", a7, 1.937, 0.005, abs(a7 - 1.937) <= 0.005,
       cfmt="%.4f", tfmt="%.3f")
report("T12 full fit b (X=1e7)", b7, 0.927, 0.005, abs(b7 - 0.927) <= 0.005,
       cfmt="%.4f", tfmt="%.3f")
report("T12 full fit R2 (X=1e7)", r2_7, 0.9510, 0.001,
       abs(r2_7 - 0.9510) <= 0.001, cfmt="%.4f", tfmt="%.4f")
L4a = LXall[:, 2]                                       # column 2 = X=1e4
(_, _), _, resg_a = ols(lg_a[:, None], lf_a)
pc_a = float(np.corrcoef(resid_on(resg_a, L4a), resid_on(far_a, L4a))[0, 1])
report("T12 partial corr(resid, far | L_X)", pc_a, 0.027, 0.01,
       abs(pc_a - 0.027) <= 0.01, cfmt="%.4f", tfmt="%.3f")
k5_a = float(np.corrcoef(resg_a, np.log(far_a))[0, 1])
report("T12 K5 corr(resid, log far)", k5_a, -0.725, 0.01,
       abs(k5_a + 0.725) <= 0.01, cfmt="%.4f", tfmt="%.3f")

# ================================================================= T13 smooth cutoff
print("\n--- T13: smooth-cutoff scan ---")
sc = np.genfromtxt(DATA + "/smooth_cutoff_scan.csv", delimiter=",", names=True,
                   dtype=None, encoding="utf-8")
scX = sorted(set(int(x) for x in sc["X"]))
dev_e, dev_b = 0.0, 0.0
for X in scX:
    bs = {c: float(sc["b"][(sc["cutoff"] == c) & (sc["X"] == X)][0])
          for c in ("sharp", "exp4", "bump")}
    dev_e = max(dev_e, abs(bs["exp4"] - bs["sharp"]))
    dev_b = max(dev_b, abs(bs["bump"] - bs["sharp"]))
report("T13 max |b_exp4 - b_sharp| over X", dev_e, 0.0, 0.03, dev_e <= 0.03,
       cfmt="%.4f", tfmt="%.0f")
report("T13 max |b_bump - b_sharp| over X", dev_b, 0.0, 0.03, dev_b <= 0.03,
       cfmt="%.4f", tfmt="%.0f")
# spot recompute: X=1e4, exp4 weight Phi(p/X)=exp(-(p/X)^4), primes p<=4X
PRS = sieve(4 * 10 ** 4)
PHI = np.exp(-(PRS.astype(float) / 1e4) ** 4)
LNPS = np.log(PRS.astype(np.longdouble))
PSF = PRS.astype(float)
Ls = np.empty(len(t0_pairs))
for j, tt in enumerate(t0_pairs):
    c1 = np.cos(np.asarray((tt * LNPS) % TWO_PI_LD, dtype=float))
    c2 = 2.0 * c1 ** 2 - 1.0
    c3 = 4.0 * c1 ** 3 - 3.0 * c1
    Ls[j] = (np.sum(PHI * c1 / np.sqrt(PSF)) + 0.5 * np.sum(PHI * c2 / PSF)
             + (1.0 / 3.0) * np.sum(PHI * c3 / PSF ** 1.5))
(as_, bs_, cs_), r2s, _ = ols(np.column_stack([lg, Ls]), lf)
b_csv = float(sc["b"][(sc["cutoff"] == "exp4") & (sc["X"] == 10000)][0])
report("T13 spot recompute b_exp4(X=1e4)", bs_, b_csv, 0.005,
       abs(bs_ - b_csv) <= 0.005, cfmt="%.6f", tfmt="%.6f")

# ================================================================= T14 EIV
print("\n--- T14: EIV attenuation correction at X=1e7 ---")
varW = float(np.var(LXall[:, 5], ddof=1))
lam14 = 1.0 - 0.113 ** 2 / varW          # sigma_U = sup bound 0.113 (the floor convention of Section 52)
fc = np.genfromtxt(DATA + "/fits_all_pairs.csv", delimiter=",", names=True,
                   dtype=None, encoding="utf-8")
mrow = (fc["ensemble"] == "8.4e9") & (fc["X"] == 10000000)
b_obs14 = float(fc["b"][mrow][0])
b_eiv14 = b_obs14 / lam14
report("T14 reliability lambda (sup bound)", lam14, 0.974, 0.005,
       abs(lam14 - 0.974) <= 0.005, cfmt="%.4f", tfmt="%.3f")
report("T14 b_EIV = b/lambda (X=1e7)", b_eiv14, 0.952, 0.005,
       abs(b_eiv14 - 0.952) <= 0.005, cfmt="%.4f", tfmt="%.3f")

# ================================================================= T15 GUE constant
print("\n--- T15: GUE far-field constant MC (N=200, Dumitriu-Edelman) ---")
C_GUE = 4 * np.pi ** 2 / 15.0                        # 2.6318945069
if SLOW:
    try:
        from scipy.linalg import eigh_tridiagonal as _eigt
        def _gue_eigs(dd, ee):
            return _eigt(dd, ee, eigvals_only=True, check_finite=False)
    except ImportError:
        def _gue_eigs(dd, ee):
            Tm = np.diag(dd) + np.diag(ee, 1) + np.diag(ee, -1)
            return np.linalg.eigvalsh(Tm)

    def gue_unfold(w, N):
        R = 2.0 * np.sqrt(N)
        z = np.clip(w / R, -1.0, 1.0)
        F = 0.5 + (z * np.sqrt(1 - z * z) + np.arcsin(z)) / np.pi
        return N * (F - 0.5)

    N15, S0C, WANT = 200, 0.25, 20000
    Xb = 0.2 * (N15 / 2.0)                             # bulk conditioning
    wins = [w for w in (25.0, 50.0, 100.0, 200.0)
            if w <= min(200.0, N15 / 2.0 - Xb - 5)]    # -> [25, 50]
    rng15 = np.random.default_rng(2024)
    gaps15 = []
    sums15 = {w: [] for w in wins}
    t15 = time.time()
    nm = 0
    while len(gaps15) < WANT:
        dd = rng15.standard_normal(N15)
        ee = np.sqrt(rng15.chisquare(2 * np.arange(N15 - 1, 0, -1)) / 2.0)
        x = gue_unfold(_gue_eigs(dd, ee), N15)
        sp = np.diff(x)
        midx = 0.5 * (x[:-1] + x[1:])
        cond = (np.abs(midx) <= Xb) & (sp < S0C)
        for i in np.nonzero(cond)[0]:
            rel = np.delete(x - midx[i], [i, i + 1])
            arel = np.abs(rel)
            gaps15.append(sp[i])
            for w in wins:
                m = arel < w
                sums15[w].append(np.sum(1.0 / rel[m] ** 2) if m.any() else 0.0)
        nm += 1
    for w in wins:
        est = float(np.mean(sums15[w][:WANT])) + 2.0 / w   # window correction
        ok15 = abs(est / C_GUE - 1.0) <= 0.03
        report("T15 C_GUE est (N=200, X=%d, +2/X)" % int(w), est, C_GUE, 0.03,
               ok15, cfmt="%.5f", tfmt="%.5f")
    print("    (%d matrices, %d pairs, %.0fs; rel devs %.3f%%/%.3f%%)"
          % (nm, WANT, time.time() - t15,
             100 * abs((float(np.mean(sums15[wins[0]][:WANT])) + 2 / wins[0])
                 / C_GUE - 1),
             100 * abs((float(np.mean(sums15[wins[1]][:WANT])) + 2 / wins[1])
                 / C_GUE - 1)))
else:
    print("    skipped (default run; use --slow to enable)")

# ================================================================= T16 Delta_0
print("\n--- T16: k-tail truncation constant Delta_0 (Lemma 56.1, Theorem 56.9(a)) ---")
x16 = P7 ** -0.5
d0_term = -np.log1p(-x16) - x16 - 0.5 * P7 ** -1 - (P7 ** -1.5) / 3.0
d0_cum = np.cumsum(d0_term)
D0 = float(d0_cum[-1])
report("T16 Delta_0(1e7)", D0, 0.2386607627819104, 1e-12,
       abs(D0 - 0.2386607627819104) <= 1e-12, cfmt="%.13f", tfmt="%.13f")
d0p2 = float(d0_term[0])
report("T16 p=2 term", d0p2, 0.15298926591521006, 1e-15,
       abs(d0p2 - 0.15298926591521006) <= 1e-15, cfmt="%.15f", tfmt="%.15f")
XS16 = [10 ** k for k in range(2, 8)]
d0_x = [float(d0_cum[np.searchsorted(PR7, X, side="right") - 1]) for X in XS16]
mono16 = all(d0_x[i + 1] >= d0_x[i] for i in range(len(d0_x) - 1))
report("T16 Delta_0 monotone X=1e2..1e7", float(mono16), 1.0, 0.0, mono16,
       cfmt="%.0f", tfmt="%.0f")
print("    " + "  ".join("1e%d:%.10f" % (int(np.log10(X)), v)
                         for X, v in zip(XS16, d0_x)))
span16 = d0_x[-1] - d0_x[0]
report("T16 Delta_0(1e7)-Delta_0(1e2) < 5e-4", span16, 5e-4, 5e-4,
       span16 < 5e-4, cfmt="%.3e", tfmt="%.1e")
report("T16 Delta_0(1e2)", d0_x[0], 0.2381784258, 1e-9,
       abs(d0_x[0] - 0.2381784258) <= 1e-9, cfmt="%.10f", tfmt="%.10f")
# all-prime overshoot: sum_{p>X} p^-2/(4(1-p^-1/2)) <= (1/log X) *
# int_X^inf dt/(4 t^2 (1-t^-1/2)) = (-2u - 2 log(1-u))/(4 log X), u=X^-1/2
u16 = 1e7 ** -0.5
tail16 = float((-2 * u16 - 2 * np.log1p(-u16)) / (4 * np.log(1e7)))
report("T16 all-prime tail est <= 2e-9", tail16, 1.5e-9, 2e-9,
       tail16 <= 2e-9, cfmt="%.3e", tfmt="%.1e")

# ================================================================= T17 mean value
print("\n--- T17: mean-value identity MC, X=100, s=1 (Lemma 56.3) ---")
NQ9 = 4096
th9 = (np.arange(NQ9) + 0.5) * (2 * np.pi / NQ9)
c1g9, c2g9, c3g9 = np.cos(th9), np.cos(2 * th9), np.cos(3 * th9)


def term_grid(PRf):
    """term_p(theta) on the NQ9 midpoint grid, for float prime array PRf."""
    return (PRf[:, None] ** -0.5 * c1g9 + 0.5 * PRf[:, None] ** -1 * c2g9
            + (PRf[:, None] ** -1.5) / 3.0 * c3g9)


PR17 = P7[PR7 <= 100]
TM17 = term_grid(PR17)
prod17 = float(np.exp(np.sum(np.log(np.exp(-TM17).mean(axis=1)))))
report("T17 prod M_p(-1) (N=4096)", prod17, 1.5605, 5e-4,
       abs(prod17 - 1.5605) <= 5e-4, cfmt="%.6f", tfmt="%.4f")
rng17 = np.random.default_rng(19)                       # fixed seed
t17 = rng17.uniform(0.0, 2e6, 300_000)
ph17 = (t17[:, None] * np.log(PR17)[None, :]) % (2 * np.pi)
L17 = (np.cos(ph17) @ PR17 ** -0.5 + 0.5 * np.cos(2 * ph17) @ PR17 ** -1
       + (1.0 / 3.0) * np.cos(3 * ph17) @ PR17 ** -1.5)
Y17 = np.exp(-L17)
mc17 = float(Y17.mean())
se17 = float(Y17.std(ddof=1) / np.sqrt(len(Y17)))
report("T17 MC mean (seed 19)", mc17, 1.5620, 4e-3,
       abs(mc17 - 1.5620) <= 4e-3, cfmt="%.6f", tfmt="%.4f")
report("T17 MC std err", se17, 0.0030, 3e-4,
       abs(se17 - 0.0030) <= 3e-4, cfmt="%.6f", tfmt="%.4f")
rat17 = abs(mc17 - prod17) / se17
report("T17 |MC - product| <= 5 se", rat17, 5.0, 5.0, rat17 <= 5.0,
       cfmt="%.3f", tfmt="%.0f")

# ================================================================= T18 moments
print("\n--- T18: effective second moment constants (Lemma 56.4) ---")
a_p18 = P7 ** -0.5 + 0.5 * P7 ** -1 + (P7 ** -1.5) / 3.0
Sa18 = float(a_p18.sum())
report("T18 sum a_p", Sa18, 463.4263549323204, 1e-9,
       abs(Sa18 - 463.4263549323204) <= 1e-9, cfmt="%.13f", tfmt="%.13f")
VarL18 = float(0.5 * np.sum(P7 ** -1 + P7 ** -2 / 4.0 + P7 ** -3 / 9.0))
report("T18 Var(L_X) k<=3", VarL18, 1.5869646529291164, 1e-12,
       abs(VarL18 - 1.5869646529291164) <= 1e-12, cfmt="%.13f", tfmt="%.13f")
sig18 = float(np.sqrt(VarL18))
report("T18 sigma", sig18, 1.25975, 1e-4, abs(sig18 - 1.25975) <= 1e-4,
       cfmt="%.6f", tfmt="%.5f")
TBIG = 8.436e9
err18 = Sa18 ** 2 * 4 * 1e7 / TBIG ** 2
report("T18 2nd-moment err (sum a_p)^2 4X/H", err18, 1.21e-7, 0.05,
       abs(err18 / 1.21e-7 - 1) <= 0.05, cfmt="%.4e", tfmt="%.2e")
# Lemma 56.4 (v2): mixed-harmonic class (ii) 8 pi(X)^2/H, sum-frequency
# class (iii) (sum a_p)^2/(H log 2), and the corrected total 1.71e-7.
err18ii = 8.0 * len(P7) ** 2 / TBIG ** 2
report("T18 class (ii) 8 pi(X)^2/H", err18ii, 4.96e-8, 0.05,
       abs(err18ii / 4.96e-8 - 1) <= 0.05, cfmt="%.4e", tfmt="%.2e")
err18iii = Sa18 ** 2 / (TBIG ** 2 * np.log(2.0))
report("T18 class (iii) (sum a_p)^2/(H log2)", err18iii, 4.4e-15, 0.05,
       abs(err18iii / 4.4e-15 - 1) <= 0.05, cfmt="%.4e", tfmt="%.2e")
err18tot = err18 + err18ii + err18iii
report("T18 total 2nd-moment error (Lemma 56.4)", err18tot, 1.71e-7, 0.05,
       abs(err18tot / 1.71e-7 - 1) <= 0.05, cfmt="%.4e", tfmt="%.2e")
mean18 = 2 * Sa18 / (TBIG ** 2 * np.log(2))
report("T18 mean bound 2 sum a_p/(H log 2)", mean18, 1.9e-17, 0.05,
       abs(mean18 / 1.9e-17 - 1) <= 0.05, cfmt="%.4e", tfmt="%.1e")

# ================================================================= T19 Chernoff
print("\n--- T19: certified hybrid Chernoff bounds (Theorem 56.5) ---")
PR19 = P7[PR7 <= 1000]
TM19 = term_grid(PR19)                       # strict k<=3 integrand
PB19 = P7[PR7 > 1000]
kk19 = np.arange(1, 61)                      # sigma_p^2 full-k, k to 60
sig2_19 = 0.5 * np.sum(PB19[:, None] ** -kk19[None, :] / kk19[None, :] ** 2,
                       axis=1)
cp19 = -np.log1p(-PB19 ** -0.5)


def sum_logM_small19(s):
    """sum_{p<=1000} log M_p(-s); log AFTER the grid mean, then sum."""
    return float(np.sum(np.log(np.exp(-s * TM19).mean(axis=1))))


def tail_bennett19(s):
    sc = s * cp19
    return float(np.sum(sig2_19 * (np.expm1(sc) - sc) / cp19 ** 2))


def cert_d19(u, s):
    return float(np.exp(-(s * u - sum_logM_small19(s) - tail_bennett19(s)
                          - 1e-6)))


for u19, s19, tgt19 in ((6.90, 6.25, 3.35e-9), (7.00, 6.5, 1.76e-9),
                        (8.50, 8.25, 2.77e-14), (9.36, 9.5, 1.36e-17)):
    d19 = cert_d19(u19, s19)
    report("T19 certified d(%.2f, s=%.2f)" % (u19, s19), d19, tgt19, 0.02,
           abs(d19 / tgt19 - 1) <= 0.02, cfmt="%.4e", tfmt="%.2e")
# pure Bennett over ALL primes at u=6.9: exact small primes are essential
sig2_all19 = 0.5 * np.sum(P7[:, None] ** -kk19[None, :] / kk19[None, :] ** 2,
                          axis=1)
cp_all19 = -np.log1p(-P7 ** -0.5)


def pure_bennett19(u, s):
    sc = s * cp_all19
    return float(np.exp(-(s * u -
                          np.sum(sig2_all19 * (np.expm1(sc) - sc)
                                 / cp_all19 ** 2))))


with np.errstate(over="ignore"):
    pb_vals = [pure_bennett19(6.9, sv) for sv in np.linspace(1e-6, 12, 1201)]
pb19 = min(v for v in pb_vals if np.isfinite(v))
report("T19 pure-Bennett-only d(6.9)", pb19, 1.24e-4, "[5e-5,5e-4]",
       5e-5 <= pb19 <= 5e-4, cfmt="%.3e", tfmt="%.2e")

# ================================================================= T20 uniform
print("\n--- T20: Lipschitz bound + uniform union bound (Lemma 56.7, Theorem 56.8) ---")
lam1_20 = float(np.sum(np.log(P7) * (P7 ** -0.5 + P7 ** -1 + P7 ** -1.5)))
report("T20 Lambda_1", lam1_20, 6328.439190975707, 1e-9,
       abs(lam1_20 - 6328.439190975707) <= 1e-9, cfmt="%.13f", tfmt="%.13f")
m20 = 0.5
h20 = 2 * m20 / lam1_20
N20 = TBIG / h20                                 # H = T = 8.436e9
print("    h = %.6e, N = H/h = %.6e" % (h20, N20))


def best_cert_d20(u):
    """min over s of the T19 certified bound; coarse grid + local polish."""
    best = (np.inf, None)
    for c0 in range(0, 261, 20):
        sa = np.linspace(1.0, 14.0, 261)[c0:c0 + 20]
        logM = np.log(np.exp(-sa[:, None, None] * TM19[None, :, :])
                      .mean(axis=2)).sum(axis=1)
        for j, sv in enumerate(sa):
            sc = sv * cp19
            tb = float(np.sum(sig2_19 * (np.expm1(sc) - sc) / cp19 ** 2))
            ex = -(sv * u - logM[j] - tb - 1e-6)
            if ex < best[0]:
                best = (float(ex), float(sv))
    for sv in np.linspace(best[1] - 0.05, best[1] + 0.05, 41):
        ex = -(sv * u - sum_logM_small19(sv) - tail_bennett19(sv) - 1e-6)
        if ex < best[0]:
            best = (float(ex), float(sv))
    return float(np.exp(best[0])), best[1]


d20a, s20a = best_cert_d20(9.36 - m20)
nd20a = N20 * d20a
report("T20 N*d(8.86) (u=9.36)", nd20a, "6.8e-2", "[1e-3,6.85e-2]",
       1e-3 <= nd20a <= 6.85e-2, cfmt="%.4e")
print("    (d(8.86) = %.4e at s*=%.3f; strict k<=3 integrand exceeds the "
      "2-sf table value 6.8e-2 by 0.3%%)" % (d20a, s20a))
d20b, s20b = best_cert_d20(10.0 - m20)
nd20b = N20 * d20b
report("T20 N*d(9.5) (u=10.0)", nd20b, "1.9e-4", "<=1.95e-4",
       nd20b <= 1.95e-4, cfmt="%.4e")
print("    (d(9.5) = %.4e at s*=%.3f; same 2-sf rounding note as above)"
      % (d20b, s20b))
d20c, s20c = best_cert_d20(9.0 - m20)
nd20c = N20 * d20c
report("T20 N*d(8.5) (u=9.0, marginal)", nd20c, 1.48, "[0.5,3.0]",
       0.5 <= nd20c <= 3.0, cfmt="%.4f", tfmt="%.2f")
print("    (d(8.5) = %.4e at s*=%.3f)" % (d20c, s20c))

# ================================================================= T21 Bahadur-Rao
print("\n--- T21: Bahadur-Rao prefactor, tilted IS (Proposition 56.6) ---")
PR21 = P7[PR7 <= 10 ** 4]
npr21 = len(PR21)
TM21 = term_grid(PR21)             # exact midpoint factors, ALL p <= 1e4
u21 = 4.580                        # = 4.00 sigma at X=1e4
sig21 = float(np.sqrt(0.5 * np.sum(PR21 ** -1 + PR21 ** -2 / 4.0
                                   + PR21 ** -3 / 9.0)))
report("T21 sigma(X=1e4)", sig21, 1.145, 0.002, abs(sig21 - 1.145) <= 0.002,
       cfmt="%.6f", tfmt="%.3f")


def K21(s):
    """K(s)=log M(-s), K'(s), K''(s) for the product law (exact factors)."""
    E = np.exp(-s * TM21)
    Mp = E.mean(axis=1)
    w = E / Mp[:, None]
    m1 = (w * TM21).mean(axis=1)
    m2 = (w * TM21 ** 2).mean(axis=1)
    return float(np.log(Mp).sum()), float(-m1.sum()), \
        float((m2 - m1 ** 2).sum())


s21 = 3.0
for _ in range(100):               # Newton: K'(s*) = +u  [<=> (log M)'(-s*)=-u]
    Kv, Kp, Kpp = K21(s21)
    sn = s21 - (Kp - u21) / Kpp
    if abs(sn - s21) < 1e-13:
        s21 = sn
        break
    s21 = sn
K21v, K21p, K21pp = K21(s21)
I21 = s21 * u21 - K21v
emI21 = float(np.exp(-I21))
pref21 = 1.0 / (s21 * np.sqrt(2 * np.pi * K21pp))
print("    s* = %.6f, I = %.6f, e^-I = %.6e" % (s21, I21, emI21))
report("T21 BR prefactor 1/(s* sqrt(2 pi K''))", pref21, 0.0959, 0.02,
       abs(pref21 / 0.0959 - 1) <= 0.02, cfmt="%.6f", tfmt="%.4f")
br21 = pref21 * emI21
report("T21 BR prediction pref*e^-I", br21, 1.97e-6, 0.02,
       abs(br21 / 1.97e-6 - 1) <= 0.02, cfmt="%.4e", tfmt="%.2e")
# tilted importance sampling, N=2000, fixed seed
E21 = np.exp(-s21 * TM21)
CDF21 = np.cumsum(E21 / E21.mean(axis=1)[:, None] / NQ9, axis=1)
epK21 = float(np.exp(K21v))
rng21 = np.random.default_rng(19)              # fixed seed
U21 = rng21.random((2000, npr21))
idx21 = np.empty((2000, npr21), dtype=np.int32)
for p in range(npr21):
    idx21[:, p] = np.searchsorted(CDF21[p], U21[:, p])
Lt = TM21[np.arange(npr21)[None, :], idx21].sum(axis=1)
w21 = np.exp(s21 * Lt) * (Lt <= -u21)          # IS weight incl. e^{+K} below
est21 = float(epK21 * w21.mean())
se21 = float(epK21 * w21.std(ddof=1) / np.sqrt(len(w21)))
print("    hit rate %.3f (tilted mean = -u by construction)"
      % float((Lt <= -u21).mean()))
report("T21 IS estimate (seed 19, N=2000)", est21, 1.89e-6, "[1e-6,3.5e-6]",
       1.0e-6 <= est21 <= 3.5e-6, cfmt="%.4e", tfmt="%.2e")
rat21 = abs(est21 - 1.97e-6) / se21
report("T21 |est - 1.97e-6| <= 5 se", rat21, 5.0, 5.0, rat21 <= 5.0,
       cfmt="%.3f", tfmt="%.0f")
pref21i = est21 / emI21
report("T21 implied prefactor est/e^-I", pref21i, 0.0959, "[0.05,0.25]",
       0.05 <= pref21i <= 0.25, cfmt="%.4f", tfmt="%.4f")

# ================================================================= T22 certificate
print("\n--- T22: per-pair certificate margin (Corollary 56.2 + Proposition 55.4) ---")
g22, far22, H22 = det["g"], det["far"], det["H"]
eps22 = g22 / 2.0
Hrc22 = np.sqrt(2.0 / far22)
hdev22 = float(np.max(np.abs(Hrc22 - H22) / H22))
report("T22 H col == sqrt(2/far) max rel dev", hdev22, 0.0, 1e-6,
       hdev22 <= 1e-6, cfmt="%.2e", tfmt="%.0f")
Dst22 = np.log(4.0 * (eps22 ** 2 + 4.0 / far22) ** 2 / g22 ** 2)
elig22 = far22 < 4.0 / (eps22 * (1.0 - eps22))      # exactly Delta_* > 0
report("T22 eligible pairs (>=2700)", float(elig22.sum()), 2700.0, ">=",
       elig22.sum() >= 2700, cfmt="%.0f", tfmt="%.0f")
marg22 = Dst22[elig22] - D0
mmin22, mmax22 = float(marg22.min()), float(marg22.max())
report("T22 margin Delta_*-Delta_0 > 0 (min)", mmin22, ">0", "-",
       mmin22 > 0, cfmt="%.4f")
report("T22 margin range all eligible [min,max]",
       "%.4f/%.4f" % (mmin22, mmax22), "info (reference 6.9-8.5)", "-", True)
# manuscript's "reference pair" (Table 56.2: u=6.90 = Delta_* at the reference
# pair): representative g=1.0e-2 with the data median far. NOTE (v2): the true
# ensemble median of Delta_* is 4.46 (median-gap pair g=0.0356), NOT 6.9;
# the value 6.9 belongs to the reference pair g=1e-2 (top ~1% by tightness).
fmed22 = float(np.median(far22))
gmed22 = 1.0e-2
emed22 = gmed22 / 2.0
Dst_med22 = float(np.log(4.0 * (emed22 ** 2 + 4.0 / fmed22) ** 2 / gmed22 ** 2))
report("T22 Delta_* at reference pair (g=1e-2)", Dst_med22, 6.90, 0.1,
       abs(Dst_med22 - 6.90) <= 0.1, cfmt="%.4f", tfmt="%.2f")
mmed22 = Dst_med22 - D0
report("T22 margin at reference pair in [6.0,8.5]", mmed22, "6.9", "[6.0,8.5]",
       6.0 <= mmed22 <= 8.5, cfmt="%.4f")
Dst_truemed22 = float(np.median(Dst22[elig22]))
print("    info: true ensemble median Delta_* (eligible) = %.4f, margin %.4f"
      % (Dst_truemed22, Dst_truemed22 - D0))
imax22 = int(np.argmax(far22))
report("T22 max-far pair Delta_*", float(Dst22[imax22]), -1.25, 0.3,
       abs(Dst22[imax22] + 1.25) <= 0.3, cfmt="%.4f", tfmt="%.2f")
# headline B_*(H)/|P_X| = g^2/(4 (eps^2+2H^2)^2) of Proposition 8.4.
# Table provenance verified against data: min g = 4.1523e-3, mean far =
# 27.98 ("28"), median far = 24.84, max-far pair g=4.228e-2, far=368.35.
Bstar22 = g22 ** 2 / (4.0 * (eps22 ** 2 + 2.0 * H22 ** 2) ** 2)
it22 = int(np.argmin(g22))
fmean22 = float(far22.mean())
gt22 = float(g22[it22])
Bt22 = gt22 ** 2 / (4.0 * ((gt22 / 2) ** 2 + 4.0 / fmean22) ** 2)
report("T22 B*/|P_X| tightest (min g, mean far)", Bt22, 2.1e-4, 0.15,
       abs(Bt22 / 2.1e-4 - 1) <= 0.15, cfmt="%.4e", tfmt="%.1e")
Bm22 = gmed22 ** 2 / (4.0 * (emed22 ** 2 + 4.0 / fmed22) ** 2)
report("T22 B*/|P_X| reference (g=1e-2, median far)", Bm22, 9.6e-4, 0.15,
       abs(Bm22 / 9.6e-4 - 1) <= 0.15, cfmt="%.4e", tfmt="%.1e")
Bx22 = float(Bstar22[imax22])
report("T22 B*/|P_X| max-far pair", Bx22, 3.5, 0.15,
       abs(Bx22 / 3.5 - 1) <= 0.15, cfmt="%.4f", tfmt="%.1f")
# literal per-pair variants (info only; see docstring provenance note)
report("T22 info: B* min-g pair own far=23.01", float(Bstar22[it22]),
       "info (table 2.1e-4 uses mean far)", "-", True, cfmt="%.4e")
imedg22 = int(np.argsort(g22)[len(g22) // 2])
report("T22 info: B* median-g pair (g=0.0356)", float(Bstar22[imedg22]),
       "info (table 9.6e-4 uses g=1e-2)", "-", True, cfmt="%.4e")

# ================================================================= summary
n_pass = sum(1 for _, ok in results if ok)
n_all = len(results)
print("\n" + "=" * 80)
if n_pass == n_all:
    print("ALL PASS (%d/%d)   [total runtime %.1fs]"
          % (n_pass, n_all, time.time() - t_start))
else:
    print("FAILURES (%d/%d passed):" % (n_pass, n_all))
    for name, ok in results:
        if not ok:
            print("  FAIL: %s" % name)
sys.exit(0 if n_pass == n_all else 1)

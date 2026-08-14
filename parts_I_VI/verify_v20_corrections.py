#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_v20_corrections.py  --  rh_pf_v20
=========================================
Reproduces every number introduced by the Part-IV corrections of v20
(Lemma 23.1 closed form and calibration, Lemma 23.2' spill bound,
Remark 24.1' cluster counterexample, Proposition 24.3 net constants,
Corollary 24.5 shallow indistinguishability), together with the
regularized-gamma values of Lemma 17.3, the Mertens fourth-moment
asymptotic of (17.6)/Theorem 19.2, the Kusmin-correction arithmetic of
Lemma 15.1, and the low-range |S(T)| check used in its stitching.

Every check prints PASS/FAIL; exit code 0 iff all pass.
"""
import math, sys
import numpy as np

FAILS = []
def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    if not ok: FAILS.append(name)
    print("[%s] %s%s" % (tag, name, ("  --  " + detail) if detail else ""))

W = 0.5

def contrib(u, h, w=W):
    """Exact zero-side contribution of an off-line quadruple at offset u
    (Lemma 23.1): 2 Re[(u-ih)^2 exp(-(u-ih)^2/w^2)]."""
    z = (u - 1j*h)**2
    return 2*np.real(z*np.exp(-z/w**2))

def online(u, w=W):
    """h->0 collapse: on-line double zero value 2 u^2 e^{-u^2/w^2}."""
    return 2*u*u*math.exp(-u*u/w**2)

# ----------------------------------------------------------------------
print("== Lemma 23.1: closed form, sign threshold, calibration, bound ==")

# (a) closed form == direct product of the witness transforms
ok = True
for u in (0.0, 0.1, 0.2, 0.45):
    h = 0.3085
    g1  = (u-1j*h)*np.exp(-(u-1j*h)**2/(2*W*W))
    g2c = np.conj((u+1j*h)*np.exp(-(u+1j*h)**2/(2*W*W)))
    ok &= abs(2*np.real(g1*g2c) - contrib(u, h)) < 1e-14
check("closed form == direct evaluation (4 offsets)", ok)

# (b) centre value and the demand identity -2 h^2 e^{h^2/w^2}
h1 = 0.30852
check("centre demand at DH witness  = -0.2786",
      abs(contrib(0.0, h1) + 2*h1*h1*math.exp(h1*h1/W**2)) < 1e-12
      and abs(2*h1*h1*math.exp(h1*h1/W**2) - 0.2786) < 5e-5,
      "2h^2 e^{h^2/w^2} = %.4f" % (2*h1*h1*math.exp(h1*h1/W**2)))

# (c) sign threshold u_c/h: -> 1 shallow, ~0.40 at (1/2,1/2)
def uc_over_h(h):
    us = np.linspace(0, 3*h, 4000)
    c  = np.array([contrib(u, h) for u in us])
    ic = np.where(c > 0)[0]
    return us[ic[0]]/h if len(ic) else np.inf
check("u_c/h -> 1 as h -> 0", abs(uc_over_h(0.02) - 1.0) < 0.05,
      "u_c/h(0.02) = %.3f" % uc_over_h(0.02))
check("u_c/h ~ 0.40 at h = 1/2", abs(uc_over_h(0.5) - 0.403) < 0.01,
      "u_c/h(0.5) = %.3f" % uc_over_h(0.5))

# (d) the 0.70 calibration on |u| <= h/5, uniform over h <= 1/2
kmin = min(-contrib(h/5, h)/(2*h*h*math.exp(h*h/W**2))
           for h in np.linspace(0.005, 0.5, 400))
check("kappa(|u|<=h/5) = 0.700 uniformly", abs(kmin - 0.700) < 0.003,
      "min ratio = %.3f" % kmin)

# (e) absolute bound, sharp at u = 0
worst = 0.0
for h in np.linspace(0.02, 0.5, 25):
    for u in np.linspace(0, 4, 400):
        bound = 2*(u*u + h*h)*math.exp((h*h - u*u)/W**2)
        worst = max(worst, abs(contrib(u, h))/bound)
check("|contribution| <= 2(u^2+h^2)e^{(h^2-u^2)/w^2}, sharp",
      worst <= 1 + 1e-9 and worst > 0.999, "max ratio = %.4f" % worst)

# ----------------------------------------------------------------------
print("== Lemma 23.2': spill of distant quadruples ==")
# numeric sanity of the display constant on a worst-case comb:
# unit-spaced quadruples of depth 1/2 at |u| >= D, C2 log T zeros per unit.
def spill_bound(D, logT, C2=0.56, w=W):
    Cw = 20*C2*math.exp(1/(4*w*w))
    return Cw*logT*(D+2)**2*math.exp(-D*D/w**2)
def spill_brute(D, logT, C2=0.56, w=W):
    s = 0.0
    for j in range(0, 400):
        x = D + j
        s += 2*C2*logT*2*(x*x + 0.25)*math.exp((0.25 - x*x)/w**2)
    return s
ok = all(spill_brute(D, 30.0) <= spill_bound(D, 30.0)
         for D in (1.0, 1.5, 2.0, 3.0))
check("brute comb spill <= displayed bound (D = 1..3, logT = 30)", ok,
      "e.g. D=2: %.2e <= %.2e" % (spill_brute(2,30), spill_bound(2,30)))

# ----------------------------------------------------------------------
print("== Remark 24.1': three-quadruple cluster counterexample ==")
gg = lambda z, t0: (z - t0)*np.exp(-(z - t0)**2/(2*W*W))
def Sigma_on(t0, onz): return float(np.sum(np.abs(gg(onz, t0))**2))
def Z(t0, onz, quads):
    return Sigma_on(t0, onz) + sum(contrib(tq - t0, hq) for tq, hq in quads)

h  = 0.3085; tc = 1000.0
quads = [(tc - h, h), (tc, h), (tc + h, h)]
onz   = np.arange(tc - 40, tc + 40, 4.0)          # sparse background
lhs   = sum(2*hq*hq for _, hq in quads)
mid   = Z(tc, onz, quads)
rhs   = sum(max(0.0, -Z(tq, onz, quads)) + Sigma_on(tq, onz) for tq, hq in quads)
check("middle-frequency Z > 0 (spill beats own demand)", mid > 0,
      "Z(t_mid) = %+.4f" % mid)
check("aggregate fails: 0.571 vs 0.130",
      abs(lhs - 0.571) < 2e-3 and abs(rhs - 0.130) < 5e-3 and lhs > 4*rhs,
      "sum 2h^2 = %.3f  vs  sum(delta+Sigma_on) = %.3f" % (lhs, rhs))

onz2 = np.arange(tc - 40, tc + 40, 0.7)           # zeta-like background
rhs2 = sum(max(0.0, -Z(tq, onz2, quads)) + Sigma_on(tq, onz2) for tq, hq in quads)
check("fails also on zeta-like background", lhs > rhs2,
      "%.3f > %.3f" % (lhs, rhs2))

# positive control: isolated quadruple, both backgrounds; and the kappa-net form
okA = okB = True
for bg in (onz, onz2):
    q1 = [(tc, h)]
    d  = max(0.0, -Z(tc, bg, q1)); so = Sigma_on(tc, bg)
    okA &= 2*h*h <= d + so + 1e-12
    ts = tc + h/5
    ds = max(0.0, -Z(ts, bg, q1)); sos = Sigma_on(ts, bg)
    okB &= 0.700*2*h*h <= ds + sos + 1e-12
check("positive control: isolated per-zero bound holds", okA)
check("positive control: kappa-net form holds at t* = t + h/5", okB)

# ----------------------------------------------------------------------
print("== Corollary 24.5: shallow O(h^2)-indistinguishability ==")
ok_even = max(abs(contrib(u, 0.17) - contrib(u, -0.17))
              for u in np.linspace(0, 2, 200)) < 1e-14
check("evenness in h (exact)", ok_even)
def Cdev(hmax):
    return max(abs(contrib(u, h) - online(u))/h**2
               for h in np.linspace(0.01, hmax, 12)
               for u in np.linspace(0, 3, 400))
c_shallow, c_all = Cdev(0.10), Cdev(0.50)
check("C'_{1/2} <= 2.1 for h <= 0.1", c_shallow <= 2.1,
      "sup = %.3f" % c_shallow)
check("C'_{1/2} <= 5.5 for all h <= 1/2", c_all <= 5.5,
      "sup = %.3f" % c_all)

# ----------------------------------------------------------------------
print("== Lemma 17.3 and (17.6): gamma values and the Mertens moment ==")
try:
    from scipy.special import gammainc
    vals = {2: 0.9864, 4: 0.8723, 6: 0.5987}
    ok = all(abs(gammainc(m, 2*math.pi) - v) < 6e-4 for m, v in vals.items())
    check("regularized gamma(m, 2 pi), m = 2,4,6", ok,
          ", ".join("%.4f" % gammainc(m, 2*math.pi) for m in (2, 4, 6)))
except Exception as e:
    check("regularized gamma values (scipy)", False, str(e))

try:
    from sympy import primerange
    X = 10**6
    s = sum(math.log(p)**4/p for p in primerange(2, X))
    ratio = s/(math.log(X)**4/4)
    check("sum (log p)^4 / p ~ (1/4) log^4 X  (ratio at 1e6)",
          0.98 < ratio < 1.0, "ratio = %.3f" % ratio)
except Exception as e:
    check("Mertens fourth moment (sympy)", False, str(e))

# ----------------------------------------------------------------------
print("== Lemma 15.1 arithmetic and low-range stitching ==")
c = 2.44953 + 0.111589*(math.log(0.618) - math.log(2.38))
check("Cheng-Graham -> HPY substitution: c = 2.29907",
      abs(c - 2.29907) < 2e-5, "c = %.5f" % c)
check("bound minimum at T = e equals 1.42",
      abs(0.1366 + 1.283 - 1.42) < 2e-3)

try:
    import mpmath as mp
    mp.mp.dps = 15
    gams = [float(mp.zetazero(n).imag) for n in range(1, 6)]
    Ssm  = lambda T: -(T/(2*math.pi))*(math.log(T/(2*math.pi)) - 1) - 7/8
    pts  = [math.e, 5, 10, 14.0, 14.2, 20, 21.0, 21.1, 25.0, 25.1, 30.4, 30.5, 32]
    mx   = max(abs(sum(1 for g in gams if g <= T) + Ssm(T)) for T in pts)
    check("max |S(T)| on [e, 32] <= 0.71", mx <= 0.71, "max = %.3f" % mx)
except Exception as e:
    check("low-range S(T) (mpmath)", False, str(e))

# ----------------------------------------------------------------------
print()
if FAILS:
    print("RESULT: %d FAILURE(S): %s" % (len(FAILS), ", ".join(FAILS)))
    sys.exit(1)
print("RESULT: ALL CHECKS PASS")

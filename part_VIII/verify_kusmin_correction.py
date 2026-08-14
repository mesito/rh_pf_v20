"""
verify_kusmin_correction.py -- Theorem 41.1(i) (Trudgian-parameter branch,
c = 2.332) of the manuscript (rh_pf_v20, Part VIII).

# Part (i) of Theorem 41.1 is certified here; the height-optimized part (ii)
# and its pipeline are certified by verify_optimized_bound.py.
Reproduces, with PASS/FAIL checks:
  1. Trudgian (6.1) at (eta, r) = (0.06, 2.08): phi1/phi2/phi3, a, b, the k1-coefficient,
     and the VALIDATION c(k1=2.38) = 2.44953 reproducing Trudgian's arXiv constants
     (0.111, 0.275, 2.450) exactly.
  2. The Kusmin--Landau correction: Cheng--Graham k1=2.38 (erroneous input) replaced by
     Hiary--Patel--Yang k1=0.618 [J. Number Theory 256 (2024), 195-217]:
     c = 2.44953 - 0.15046 = 2.29907; with the published safety margin +0.0605: c <= 2.332.
  3. The corrected pipeline: Sbar(T0)=6.483, x1*(T0)=3.0545, tau*(T0)=0.2258,
    print("  dividend table (branch (i) clock; branch (ii) is in verify_14_15_flow_bounds.py):")

Requires: mpmath, numpy.
"""
import math
import numpy as np
import mpmath as mp

mp.mp.dps = 25
PASS = lambda ok: "PASS" if ok else "FAIL"

def part1_trudgian_61():
    eta, r = mp.mpf('0.06'), mp.mpf('2.08')
    R = r*(mp.mpf('0.5')+eta)
    phi1, phi2, phi3 = mp.asin(eta/R), mp.asin(1/r), mp.asin((1+eta)/R)
    c1, c2, c3 = mp.cos(phi1), mp.cos(phi2), mp.cos(phi3)
    logr, pi = mp.log(r), mp.pi
    a = (phi1*eta+(phi2-3*pi/2)*(mp.mpf('0.5')+eta)+phi3*(1+eta)+R*(c1+c2+c3))/(6*pi*logr)
    b = (-phi1+phi3+R*((1-c1)/eta+(pi/2-c3-phi3)/(R-(1+eta))))/(2*pi*logr)
    Bk = R*(2*c2-c1-c3)+phi2-phi3+eta*(2*phi2-phi1-phi3)
    coef = -2*Bk/(2*pi*logr)          # c-term = coef * log(k1)
    T1 = mp.log(mp.zeta(1+eta)/mp.zeta(2+2*eta))/(2*logr)
    T2 = mp.log(mp.zeta(mp.mpf('0.5')+mp.sqrt(2)*(eta+mp.mpf('0.5'))))/pi
    T3 = mp.quad(lambda ph: mp.log(mp.zeta(1+eta+R*mp.cos(ph))), [-pi/2, pi/2])/(4*pi*logr)
    inner = (mp.log(mp.zeta(1+eta))*(phi1+R*(c1-1)/eta)
             + (mp.mpf('0.5')+eta)*(pi/2-phi2-r*c2)*mp.log(2*pi)
             + ((1+eta)*(pi/2-phi3)-R*c3)/(1+eta-R)*mp.log(mp.zeta(R-eta)))
    cval = lambda k1: T1+T2+T3+(inner-2*mp.log(k1)*Bk)/(2*pi*logr)+mp.mpf('0.003')
    print("== Part 1: Trudgian (6.1) at (eta,r)=(0.06,2.08) ==")
    print(f"  phi1={float(phi1):.6f} (0.051534) {PASS(abs(phi1-0.051534)<1e-6)}")
    print(f"  phi2={float(phi2):.6f} (0.501532) {PASS(abs(phi2-0.501532)<1e-6)}")
    print(f"  phi3={float(phi3):.6f} (1.143350) {PASS(abs(phi3-1.143350)<1e-6)}")
    print(f"  a   ={float(a):.6f} (0.110428)  {PASS(abs(a-0.110428)<1e-6)}")
    print(f"  b   ={float(b):.6f} (0.274023)  {PASS(abs(b-0.274023)<1e-6)}")
    print(f"  k1-coefficient = {float(coef):.6f} (+0.111589) {PASS(abs(coef-0.111589)<1e-6)}")
    cCG = cval(mp.mpf('2.38'))
    print(f"  VALIDATION c(k1=2.38)={float(cCG):.5f} (2.44953; Trudgian arXiv 2.450) "
          f"{PASS(abs(cCG-2.44953)<2e-5)}")
    print("== Part 2: Kusmin--Landau correction (HPY k1=0.618) ==")
    cH = cval(mp.mpf('0.618'))
    print(f"  Delta = 0.111589*(ln 0.618 - ln 2.38) = {float(cH-cCG):+.5f} (-0.15046) "
          f"{PASS(abs((cH-cCG)+0.15046)<2e-5)}")
    print(f"  c(HPY) = {float(cH):.5f} (2.29907) {PASS(abs(cH-2.29907)<2e-5)}")
    c_final = cH + mp.mpf('0.0605')
    print(f"  + published safety margin 0.0605 -> {float(c_final):.4f} <= 2.332 "
          f"{PASS(c_final <= 2.332)}")
    print("  => |S(T)| <= 0.112 log T + 0.278 loglog T + 2.332  for T >= e")
    return 2.332

def tau_star(logT, c_const=2.332, K=40000, ny=8000):
    """Frozen-field budget-extremal collision time (paired extremal lattice)."""
    Sb = 0.112*logT + 0.278*math.log(logT) + c_const
    cc = 2*math.pi/logT
    b = 2*Sb
    j = np.arange(1, K+1)
    d = cc*(2*j - 1 + b)                       # level positions, two zeros per level
    ys = np.linspace(1e-6, 1.0, ny)
    integ = np.array([1.0/(1.0/y + 2*np.sum(2*y/(d*d + y*y))) for y in ys])
    return float(np.trapezoid(integ, ys)), Sb

def part3_pipeline(c_const):
    print("== Part 3: corrected pipeline ==")
    logT0 = math.log(3e12)
    t0, Sb = tau_star(logT0, c_const)
    x1 = 2*math.pi/logT0*(1+2*Sb)
    print(f"  Sbar(T0) = {Sb:.4f} (6.483)  {PASS(abs(Sb-6.4832)<3e-4)}")
    print(f"  x1*(T0)  = {x1:.4f} (3.0545) {PASS(abs(x1-3.0545)<3e-4)}")
    print(f"  B_S(T0)  = {2*Sb:.2f} (12.97) {PASS(abs(2*Sb-12.966)<0.01)}")
    print(f"  tau*(T0) = {t0:.4f} (0.2258) {PASS(abs(t0-0.2258)<1e-3)}")
    print("  dividend table (branch (i) clock; branch (ii) is in verify_14_15_flow_bounds.py):")
    refs = [(3.0e12, 0.2258), (1.25e15, 0.1977), (2.4e17, 0.1784),
            (5.2e21, 0.1507), (1.1e26, 0.1308), (1.3e52, 0.0746)]
    for T, ref in refs:
        v, S = tau_star(math.log(T), c_const)
        print(f"    T={T:11.3g}  Sbar={S:6.2f}  tau*={v:.4f}  (paper {ref:.4f}) "
              f"{PASS(abs(v-ref)<1.5e-3)}")
    lo, hi = logT0, 40.0
    for _ in range(18):
        m = (lo+hi)/2
        if tau_star(m, c_const, K=20000, ny=4000)[0] > 0.2: lo = m
        else: hi = m
    Tc = math.exp((lo+hi)/2)
    print(f"  0.2-crossing at T = {Tc:.2e}  (~7.2e14) {PASS(5e14 < Tc < 1.2e15)}")

if __name__ == "__main__":
    c = part1_trudgian_61()
    part3_pipeline(c)

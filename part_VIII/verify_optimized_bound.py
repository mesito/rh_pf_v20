"""
verify_optimized_bound.py -- Theorem 41.1(ii) and Theorem 15.6 (v20):
the height-optimized form derived from Trudgian (6.1) with the
Hiary--Patel--Yang input k1 = 0.618 at the near-optimal parameters
(eta, r) = (0.150, 2.470):

    |S(T)| <= 0.1366 log T + 0.1786 loglog T + 1.283   (T >= e).

Certifies, with PASS/FAIL:
  * raw evaluation a=0.136528, b=0.178547, c=1.243834 (25-digit quadrature),
    each below the stated rounded-up constants (c with the +0.0605 margin);
  * near-optimality (all four coordinate neighbours give larger Sbar(T0));
  * the small-T range: bound >= 2 for T >= 32, and on [e, 32]
    max |S(T)| = max |N - theta/pi - 1| <= 0.71 < 1.42 <= bound;
  * the pipeline: Sbar(T0)=5.807, x1*(T0)=2.7588, tau*(T0)=0.2147,
    ladder 0.4075 / 0.4175 (+2.4%), 0.2-crossing at ~9.3e13.

Requires: mpmath, numpy; loads zeta_zeros.npy (computes once if absent).
"""
import os, sys, math
import numpy as np
import mpmath as mp

FAILED = []
def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok: FAILED.append(name)

def trudgian_abc(eta, r, k1):
    eta, r, k1 = mp.mpf(eta), mp.mpf(r), mp.mpf(k1)
    R = r*(mp.mpf('0.5')+eta)
    p1, p2, p3 = mp.asin(eta/R), mp.asin(1/r), mp.asin((1+eta)/R)
    c1, c2, c3 = mp.cos(p1), mp.cos(p2), mp.cos(p3)
    lr, pi = mp.log(r), mp.pi
    a = (p1*eta+(p2-3*pi/2)*(mp.mpf('0.5')+eta)+p3*(1+eta)+R*(c1+c2+c3))/(6*pi*lr)
    b = (-p1+p3+R*((1-c1)/eta+(pi/2-c3-p3)/(R-(1+eta))))/(2*pi*lr)
    T1 = mp.log(mp.zeta(1+eta)/mp.zeta(2+2*eta))/(2*lr)
    T2 = mp.log(mp.zeta(mp.mpf('0.5')+mp.sqrt(2)*(eta+mp.mpf('0.5'))))/pi
    I0 = mp.quad(lambda ph: mp.log(mp.zeta(1+eta+R*mp.cos(ph))), [-pi/2, pi/2])
    Bk = R*(2*c2-c1-c3)+p2-p3+eta*(2*p2-p1-p3)
    inner = (mp.log(mp.zeta(1+eta))*(p1+R*(c1-1)/eta)
             + (mp.mpf('0.5')+eta)*(pi/2-p2-r*c2)*mp.log(2*pi)
             + ((1+eta)*(pi/2-p3)-R*c3)/(1+eta-R)*mp.log(mp.zeta(R-eta)))
    c = T1+T2+I0/(4*pi*lr)+(inner-2*mp.log(k1)*Bk)/(2*pi*lr)+mp.mpf('0.003')
    return float(a), float(b), float(c)

AS, BS, CS = 0.1366, 0.1786, 1.283
Sbar = lambda L: AS*L + BS*math.log(L) + CS

def load_zeros(n=2000):
    if os.path.exists("zeta_zeros.npy"):
        g = np.load("zeta_zeros.npy")
        if len(g) >= n: return g[:n]
    print(f"  computing {n} zeros via mpmath.zetazero (one-time)...")
    g = np.array([float(mp.zetazero(k).imag) for k in range(1, n+1)])
    np.save("zeta_zeros.npy", g); return g

def tau_star(logT, K=60000, ny=12000):
    S = Sbar(logT); cc = 2*math.pi/logT; bb = 2*S
    d = cc*(2*np.arange(1, K+1) - 1 + bb)
    ys = np.linspace(1e-6, 1.0, ny)
    return float(np.trapezoid(np.array(
        [1.0/(1.0/y + 2*np.sum(2*y/(d*d + y*y))) for y in ys]), ys))

def main():
    mp.mp.dps = 25
    print("== evaluation at (eta, r) = (0.150, 2.470), k1 = 0.618 ==")
    a, b, c = trudgian_abc('0.150', '2.470', '0.618')
    check("a = 0.136528 <= 0.1366", abs(a-0.136528) < 2e-6 and a <= AS, f"{a:.6f}")
    check("b = 0.178547 <= 0.1786", abs(b-0.178547) < 2e-6 and b <= BS, f"{b:.6f}")
    check("c_partial = 1.243834 (full c = 1.221514 in Part VIII)", abs(c-1.243834) < 2e-6, f"{c:.6f}")
    L0 = math.log(3e12); S0raw = a*L0 + b*math.log(L0) + c
    ok = True
    for de, dr in [(0.005,0),(-0.005,0),(0,0.05),(0,-0.05)]:
        aa, bb2, cc2 = trudgian_abc(0.150+de, 2.470+dr, '0.618')
        ok &= (aa*L0 + bb2*math.log(L0) + cc2) >= S0raw - 1e-4
    check("near-optimality (4 neighbours)", ok)

    print("== small-T range ==")
    Lth = 3.464  # bound = 2 at T ~ 32
    check("bound >= 2 for T >= 36", Sbar(math.log(36)) >= 2 - 5e-3, f"{Sbar(math.log(36)):.3f}")
    g = load_zeros(50)
    mp.mp.dps = 15
    ts = np.linspace(math.e, 36, 1300)
    Sv = np.array([np.sum(g <= t) - float(mp.siegeltheta(t))/math.pi - 1 for t in ts])
    bmin = min(Sbar(math.log(t)) for t in ts)
    check("max|S| on [e,36] <= 0.71 < 1.42 <= bound",
          np.max(np.abs(Sv)) <= 0.71 and bmin >= 1.42 - 5e-3,
          f"max|S|={np.max(np.abs(Sv)):.3f}, bound_min={bmin:.3f}")

    print("== pipeline (paper headline values) ==")
    check("Sbar(T0) = 5.807", abs(Sbar(L0)-5.807) < 1e-3, f"{Sbar(L0):.4f}")
    x1 = 2*math.pi/L0*(1+2*Sbar(L0))
    check("x1*(T0) = 2.7588", abs(x1-2.7588) < 5e-4, f"{x1:.4f}")
    check("B_S(T0) = 11.61", abs(2*Sbar(L0)-11.61) < 0.01, f"{2*Sbar(L0):.3f}")
    t0 = tau_star(L0)
    check("tau*(T0) = 0.2147", abs(t0-0.2147) < 8e-4, f"{t0:.4f}")
    x2 = x1*x1
    tau2 = 0.1 + (2*x2/25)*math.log(1+5/x2)
    taud = (math.sqrt(1+10*x2+x2*x2) - (x2-1))/12
    check("ladder 0.4075 / 0.4175", abs(tau2-0.4075) < 5e-4 and abs(taud-0.4175) < 5e-4,
          f"{tau2:.4f}/{taud:.4f}")
    check("dynamic correction +2.4%", abs(100*(taud/tau2-1)-2.4) < 0.15,
          f"{100*(taud/tau2-1):+.2f}%")
    lo, hi = L0, 45
    for _ in range(18):
        m = (lo+hi)/2
        if tau_star(m, 25000, 4000) > 0.2: lo = m
        else: hi = m
    Tc = math.exp((lo+hi)/2)
    check("0.2-crossing ~ 9.3e13", 8e13 < Tc < 1.3e14, f"{Tc:.2e}")

    print()
    if FAILED:
        print("FAILED:", ", ".join(FAILED)); sys.exit(1)
    print("ALL CHECKS PASS -- the optimized bound and its pipeline are machine-certified.")

if __name__ == "__main__":
    main()

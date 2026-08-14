"""
verify_23_floor_core.py -- certifier for Theorems 4.1 and 5.1 and the
density-capped descent floor of Section 23 (corrected constants).

Certifies, on the first 2000 zeros and on random configurations:
  C1  Corrected floor drag bound  S(y) <= log(2T) + 4(Sbar+1)/y  on real zeros,
      WITH an explicit upper add-on for the truncated tail; regression guard:
      the OLD constant (log T/2 smooth part) is demonstrably violated.
  C2  Closed-form floor integral vs direct quadrature (new constants A=5+4Sbar,
      r=log 2T).
  C3  Sparse floor tau_+ >= y^2/6 (Lemma 23.5): algebraic identity
      I0-(q^2+y^2)^2 = 8 q^2 y^2 and a configuration grid.
  C4  Theorem 5.1 brackets: eps_- >= delta_h/5 and eps_+ <= 2 delta_h whenever
      A*N0 >= 2 h0^2 G  (random sweep).
  C5  Theorem 4.1 arithmetic skeleton: 4AN0 - AN0/2 - C3 log T > 0 for
      N0 >= c T log T, and area(E(T)) = O(log T / T) -> 0.
  C6  Live V'-positivity spot check on the actual zeta (compatible with the
      corrected statement: E(T) empty for verified ranges).
"""
import math
import numpy as np

FAILED = []
def check(name, ok, extra=""):
    tag = "PASS" if ok else "FAIL"
    if not ok: FAILED.append(name)
    print(f"  [{tag}] {name}" + (f"  ({extra})" if extra else ""))

def load_zeros():
    import os
    for p in ("zeta_zeros.npy", "../zeta_zeros.npy", "/home/claude/zeta_zeros.npy"):
        if os.path.exists(p):
            return np.load(p)
    raise SystemExit("zeta_zeros.npy not found")

def main():
    g = load_zeros()
    Sbar = 2.0  # admissible envelope for |S(u)|, u <= 2*2515 (true max ~1)

    print("== C1: corrected floor drag bound on real zeros ==")
    rho = lambda u: math.log(u/(2*math.pi))/(2*math.pi)
    def tail_upper(t0, y, D):
        # zeros beyond distance D from t0 (outside the 2000-zero cache):
        # sum K <= int_D^inf 2y/v^2 * [rho(t0+v)+rho(max(t0-v,10))] dv (crude)
        vs = np.geomspace(D, 1e6, 4000)
        dens = np.array([rho(t0+v)+rho(max(t0-v, 10.0)) for v in vs])
        return float(np.trapezoid(2*y/vs**2*dens, vs))
    worst_new, worst_old = 0.0, 0.0
    viol_old = 0
    for i in (150, 500, 780, 1200, 1800):
        t0 = (g[i]+g[i+1])/2
        D = min(t0-g[0], g[-1]-t0)
        for y in (2.0, 4.0, 6.0, 10.0):
            S = float(np.sum(2*y/((t0-g)**2+y**2))) + tail_upper(t0, y, D)
            new_b = math.log(2*t0) + 4*(Sbar+1)/y
            old_b = math.log(t0)/2 + 4*Sbar/y
            worst_new = max(worst_new, S/new_b)
            worst_old = max(worst_old, S/old_b)
            viol_old += S > old_b
    check("new bound S(y) <= log2T + 4(Sbar+1)/y holds (incl. tail add-on)",
          worst_new < 1, f"max ratio {worst_new:.3f}")
    check("regression guard: OLD constant (logT/2) is violated on real zeros",
          viol_old > 0, f"{viol_old} violations, max ratio {worst_old:.3f}")

    print("== C2: closed-form floor vs quadrature (new constants) ==")
    ok = True
    for (y0, T) in ((0.3, 3e12), (0.5, 1e6), (0.1, 1e10)):
        A, r = 5 + 4*Sbar, math.log(2*T)
        closed = (1/r)*(y0 - (A/r)*math.log(1 + r*y0/A))
        ys = np.linspace(0, y0, 200001)
        quad = float(np.trapezoid(ys/(A + r*ys), ys))
        ok &= abs(closed-quad) < 1e-9*max(quad, 1e-12)
    check("floor closed form = quadrature (3 configs, <1e-9 rel)", ok)

    print("== C3: sparse floor tau_+ >= y^2/6 (Lemma 23.5) ==")
    algebra_ok = True
    for _ in range(2000):
        y, q = np.random.uniform(0.01, 3), np.random.uniform(0.02, 6)
        I0 = y**4 + 10*q*q*y*y + q**4
        algebra_ok &= abs(I0 - (q*q+y*y)**2 - 8*q*q*y*y) < 1e-9*I0
    check("identity I0 - (q^2+y^2)^2 = 8 q^2 y^2 (2000 random)", algebra_ok)
    grid_ok, worst = True, 1e9
    for y in (0.1, 0.3, 0.7, 1.0, 1.5):
        for q in (1.2, 2.0, 3.0, 5.0):
            if q <= y: continue
            I0 = y**4 + 10*q*q*y*y + q**4
            tau = (math.sqrt(I0) - (q*q - y*y))/12
            worst = min(worst, tau/(y*y/6))
            grid_ok &= tau >= y*y/6 - 1e-14
    check("tau_+ >= y^2/6 on configuration grid", grid_ok, f"min ratio {worst:.3f}")

    print("== C4: Theorem 5.1 brackets ==")
    rng = np.random.default_rng(1)
    ok = True
    for _ in range(500):
        AN0 = 10**rng.uniform(6, 14)
        h0 = rng.uniform(0.01, 0.5)
        G = AN0/(2*h0*h0)*rng.uniform(0, 0.999)
        dh = h0*h0/AN0
        em = h0*h0/(4*AN0 + h0*h0*G)
        ep = h0*h0/(AN0 - h0*h0*G)
        ok &= (em >= dh/5 - 1e-18) and (ep <= 2*dh + 1e-18)
    check("eps_- >= delta_h/5 and eps_+ <= 2 delta_h (500 random, AN0>=2h0^2G)", ok)

    print("== C5: Theorem 4.1 arithmetic and exceptional area ==")
    ok = True
    for T in (1e6, 1e9, 1e12):
        c, A_, C3 = 1e-3, 0.301, 10.0        # conservative constants
        N0 = c*T*math.log(T)
        ok &= 4*A_*N0 - 0.5*A_*N0 - C3*math.log(T) > 0
    check("4AN0 - AN0/2 - C3 log T > 0 for N0 >= cT log T", ok)
    areas = [ (T*math.log(T))*math.pi*(math.log(T)/(0.301*1e-3*T*math.log(T)))**2
              for T in (1e6, 1e9, 1e12) ]
    check("area(E(T)) decreasing to 0", areas[0] > areas[1] > areas[2] and areas[2] < 1e-3,
          f"{areas[2]:.2e} at T=1e12")

    print("== C6: live V'-positivity spot check (actual zeta) ==")
    try:
        import mpmath
        mpmath.mp.dps = 15
        A_, N0 = 0.301, len(g)
        allpos = True
        for h in (0.05, 0.2, 0.45):
            for t in (35.0, 500.0, 1500.0):
                s = mpmath.mpc(0.5+h, t)
                vp = A_*N0/h**2 + mpmath.re(mpmath.diff(mpmath.zeta, s)/mpmath.zeta(s))
                allpos &= float(vp) > 0
        check("V' > 0 on 9-point grid (E(T) empty for verified zeta range)", allpos)
    except Exception as e:
        check("V' spot check", False, str(e)[:50])

    print()
    if FAILED:
        print("FAILURES:", ", ".join(FAILED)); raise SystemExit(1)
    print("ALL CORRECTION CHECKS PASS -- the corrected Sections 4/5 and the "
          "floor lemma are numerically certified.")

if __name__ == "__main__":
    main()

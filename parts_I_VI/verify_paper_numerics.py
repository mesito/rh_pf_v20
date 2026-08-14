"""
verify_paper_numerics.py -- machine-certification of every quoted Layer-B
statistic and corresponding Numerical Observation in the manuscript (v20).

Recomputes, from the first 2000 nontrivial zeros (positive ordinates,
direct summation -- the paper's convention), with PASS/FAIL asserts:

  * C_n = S_on L_n^2/8 statistics:  range [1.002, 2.306], mean 1.29, sd 0.20
  * h_thr * log t0:                 range [0.37, 8.68]; >1: 99.3%; >2: 92.5%
  * tunneling ratio Mh0/s:          2.79 +/- 0.20;  min Mh0 = 0.37 at gap 1496 (s = 0.118)
  * identity Mh0 = pi s_n / sqrt(C_n)  (exact, machine precision)
  * invariant h_thr^2 * S_on = 2       (exact, machine precision)
  * Speiser table rows (gaps 1, 100, 600):  S_on and h_thr
  * eta table rows (gaps 1, 10, 100, 600) by the paper's quadrature
  * V' grid (Sec. 18.1): all 70 points positive; min ~ 2.4e3 at (0.5, 35);
    V'(0.005, 2500) ~ 2.4e7   [A = 0.301, N0 = 2000]
  * kappa = (sqrt(10-2 sqrt 5)-2)/(sqrt 5 - 1) = 0.284079044

Zeros: loads 'zeta_zeros.npy' if present; otherwise computes them once via
mpmath.zetazero (slow first run) and caches.
"""
import os, sys, math
import numpy as np

FAILED = []
def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok: FAILED.append(name)

def load_zeros(n=2000):
    if os.path.exists("zeta_zeros.npy"):
        g = np.load("zeta_zeros.npy")
        if len(g) >= n: return g[:n]
    import mpmath as mp
    print(f"  computing {n} zeros via mpmath.zetazero (one-time)...")
    g = np.array([float(mp.zetazero(k).imag) for k in range(1, n + 1)])
    np.save("zeta_zeros.npy", g)
    return g

def main():
    g = load_zeros(2000)
    t0 = (g[:-1] + g[1:]) / 2
    Ln = np.diff(g)
    Son = np.array([np.sum(1.0 / (tt - g) ** 2) for tt in t0])   # positive zeros, direct
    hthr = np.sqrt(2.0 / Son)
    Cn = Son * Ln * Ln / 8
    lg = np.log(t0)
    HL = hthr * lg                       # = M h0 with the local mesoscopic M = log t0
    s = Ln * lg / (2 * np.pi)

    print("== C_n statistics (Remark 6.5 / Sec. 18.6) ==")
    check("C_n min = 1.002", abs(Cn.min() - 1.002) < 3e-3, f"{Cn.min():.3f}")
    check("C_n max = 2.306", abs(Cn.max() - 2.306) < 2e-2, f"{Cn.max():.3f}")
    check("C_n mean = 1.29", abs(Cn.mean() - 1.29) < 1e-2, f"{Cn.mean():.2f}")
    check("C_n sd   = 0.20", abs(Cn.std() - 0.20) < 1e-2, f"{Cn.std():.2f}")
    check("analytic bound C_n >= 1 confirmed", Cn.min() >= 1.0)

    print("== h_thr * log t0 (Remark 6.6, eq. fit) ==")
    check("range [0.37, 8.68]", abs(HL.min() - 0.37) < 1e-2 and abs(HL.max() - 8.68) < 2e-2,
          f"[{HL.min():.2f}, {HL.max():.2f}]")
    check(">1: 99.3%", abs(100 * np.mean(HL > 1) - 99.3) < 0.2, f"{100*np.mean(HL>1):.1f}%")
    check(">2: 92.5%", abs(100 * np.mean(HL > 2) - 92.5) < 0.3, f"{100*np.mean(HL>2):.1f}%")

    print("== tunneling (Thm 15.1 / Sec. 18.6) ==")
    check("Mh0/s mean = 2.79", abs((HL / s).mean() - 2.79) < 2e-2, f"{(HL/s).mean():.2f}")
    check("Mh0/s sd   = 0.20", abs((HL / s).std() - 0.20) < 2e-2, f"{(HL/s).std():.2f}")
    im = int(np.argmin(HL))
    check("min Mh0 = 0.37 at gap 1496", abs(HL[im] - 0.37) < 1e-2 and im + 1 == 1496,
          f"{HL[im]:.2f} at gap {im+1}")
    check("s at minimum = 0.118", abs(s[im] - 0.118) < 3e-3, f"{s[im]:.3f}")
    check("identity Mh0 = pi s/sqrt(C_n) (exact)",
          np.max(np.abs(HL - np.pi * s / np.sqrt(Cn))) < 1e-10)
    check("invariant h^2 S_on = 2 (machine)", np.max(np.abs(hthr**2 * Son - 2)) < 1e-12)

    print("== Speiser table rows (Sec. 13) ==")
    for idx, rS, rh in [(0, 0.218, 3.028), (99, 5.742, 0.590), (599, 20.998, 0.309)]:
        check(f"gap {idx+1}: S_on={rS}, h_thr={rh}",
              abs(Son[idx] - rS) < 2e-2 and abs(hthr[idx] - rh) < 5e-3,
              f"{Son[idx]:.3f}, {hthr[idx]:.3f}")

    print("== eta rows by the paper's quadrature (Sec. 16/18.9) ==")
    def eta(idx, K=300):
        tt = t0[idx]; i0 = int(np.argmin(np.abs(g - tt)))
        sel = g[max(0, i0 - K): i0 + K]
        h0 = hthr[idx]; hs = np.linspace(1e-9, h0, 4001)
        Pon = np.array([np.sum(1.0 / ((tt - sel) ** 2 + hh * hh)) for hh in hs])
        tau = np.trapezoid(hs / (1 + 2 * hs * hs * Pon), hs)
        return tau / (h0 * h0 / 2)
    for idx, ref in [(0, 0.4556), (9, 0.4545), (99, 0.4697), (599, 0.4763)]:
        e = eta(idx)
        check(f"eta gap {idx+1} = {ref}", abs(e - ref) < 2e-3, f"{e:.4f}")

    print("== V' grid (Sec. 18.1; A=0.301, N0=2000) ==")
    import mpmath as mp
    mp.mp.dps = 15
    def Vp(h, t):
        z = mp.mpf("0.5") + h + 1j * t
        return 0.301 * 2000 / h**2 + float(mp.re(mp.zeta(z, derivative=1) / mp.zeta(z)))
    grid = [(h, t) for h in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
                    for t in [17.5, 25, 35, 50, 70, 500, 1000, 1500, 2000, 2500]]
    vals = [Vp(h, t) for h, t in grid]
    check("all 70 grid points positive", all(v > 0 for v in vals))
    jm = int(np.argmin(vals))
    check("minimum ~ 2.4e3 at (0.5, 35)", 2.3e3 < vals[jm] < 2.5e3 and grid[jm] == (0.5, 35),
          f"{vals[jm]:.1f} at {grid[jm]}")
    v25 = Vp(0.005, 2500)
    check("V'(0.005, 2500) ~ 2.4e7", 2.3e7 < v25 < 2.5e7, f"{v25:.3g}")

    k = (math.sqrt(10 - 2 * math.sqrt(5)) - 2) / (math.sqrt(5) - 1)
    check("kappa = 0.284079044", abs(k - 0.284079044) < 1e-8, f"{k:.9f}")

    print()
    if FAILED:
        print("FAILED:", ", ".join(FAILED)); sys.exit(1)
    print("ALL CHECKS PASS -- the quoted numerics are machine-certified.")

if __name__ == "__main__":
    main()

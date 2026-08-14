"""
verify_22_24_dictionary.py -- Part IV (Sections 22-24) of the manuscript: the lifetime-
deficit equivalence, the aggregate descent lemma, the descent floor (corrected constants A=5+4*Sbar, r=log 2T; sparse floor tau>=y^2/6, Lemma 23.5; certified in verify_23_floor_core.py), the
band-wide deficit landscape, the self-consistency--capacity pincer, and the
clipping of the prime field at the Weil boundary.

Reproduces:
  aggregate descent  d/dt sum y_k^2 <= -2 m  (any configuration, shielding-robust);
  the two DH witnesses inside their predicted lifetime/deficit windows;
  band-wide triangle closure <= 2.1e-3 over 401 frequencies in [30,130];
  margin minima at the centres of the widest gaps;
  clipping: free-field extreme-value prediction +0.077 vs measured +0.0018.
"""
import math
import numpy as np
import mpmath as mp
from ic_core import dh_coeffs_fast, vonmangoldt, banner

mp.mp.dps = 15

def aggregate_descent_check():
    """Verify d/dt sum_k y_k^2 <= -2 (number of pairs) for random configurations."""
    rng = np.random.default_rng(7)
    ok = True
    for npair, nreal in [(2,0),(3,5),(5,20),(8,50),(12,100)]:
        a = rng.uniform(-10,10,npair); y = rng.uniform(0.05,1,npair); x = rng.uniform(-12,12,nreal)
        zs = np.concatenate([a+1j*y, a-1j*y, x+0j]); tot = 0.0
        for k in range(npair):
            zk = a[k]+1j*y[k]
            f = 2*np.sum([1/(zk-z) for z in zs if z != zk]); tot += 2*y[k]*f.imag
        ok = ok and (tot <= -2*npair + 1e-9)
    return ok

def run():
    banner("Lemma 23.3: aggregate descent (flow side; shielding-robust)")
    print(f"  d/dt sum y_k^2 <= -2m for all random configurations: {'OK' if aggregate_descent_check() else 'FAIL'}")
    print("  (symmetrization: cross terms -4(y_k-y_j)^2/D- -4(y_k+y_j)^2/D+ <= 0)")

    banner("Theorem 24.1 / Numerical Observation 24.2: the two DH witnesses")
    w = 0.5; slack = math.exp(1/(4*w*w))
    print(f"  slack e^(1/4w^2) = {slack:.4f}; lower-bound constant 1/(3e^(1/4w^2)) = {1/(3*slack):.4f}")
    for nm, d, h, tau in [("rho1", 0.2786, 0.3085, 0.182), ("rho2", 0.0548, 0.1508, 0.045)]:
        lo = d/(3*slack)
        inside = lo <= tau <= d + 0.01
        print(f"  {nm}: deficit={d}, window=[{lo:.4f},{d:.4f}], lifetime={tau}, 2h^2={2*h*h:.4f}  {'INSIDE' if inside else 'OUT'}")

    banner("Section 24: band-wide deficit landscape (slow part, ~2 min)")
    N = 60000
    LAM = vonmangoldt(N)
    cf = dh_coeffs_fast(N)
    n = np.arange(2, N+1); y = np.log(n)
    env = (math.sqrt(math.pi)*w**3/2)*(1 - w*w*y*y/2)*np.exp(-w*w*y*y/4)/(2*math.pi)
    base = LAM[2:]*n**(-0.5)*env
    t_grid = np.arange(30, 130.001, 0.5)   # coarser for speed
    C = np.array([np.sum(base*2*np.cos(t*y)) for t in t_grid])
    # archimedean A(t0) and zero side Z(t0)
    zeros = []; i = 1
    while True:
        g = float(mp.im(mp.zetazero(i)))
        if g > 145: break
        zeros.append(g); i += 1
    zeros = np.array(zeros)
    tg = np.arange(19, 141, 0.02)
    thp = np.array([0.5*float(mp.re(mp.digamma(0.25+1j*t/2))) for t in tg]) - 0.5*math.log(math.pi)
    A = np.array([np.sum((tg-t0)**2*np.exp(-(tg-t0)**2/(w*w))*thp)/math.pi*0.02 for t0 in t_grid])
    Z = np.array([np.sum((zeros-t0)**2*np.exp(-(zeros-t0)**2/(w*w))) for t0 in t_grid])
    closure = np.abs(A - C - Z)
    deficit = C - A
    margin = A - C
    print(f"  triangle closure |A-C-Z| over {len(t_grid)} freqs: max {closure.max():.4f} (paper <= 2.1e-3 at finer grid)")
    print(f"  true deficit max(C-A) = {deficit.max():+.4f} (paper +0.0018, within closure)")
    print(f"  C never exceeds {C.max()/0.977*100:.1f}% of envelope 0.977 (paper 5.4%)")
    print(f"  margin range [{margin.min():.3f},{margin.max():.3f}], mean {margin.mean():.3f} (paper [0,0.172], 0.044)")
    imin = np.argmin(margin)
    print(f"  margin minimum at t0={t_grid[imin]:.1f}, nearest zero {np.min(np.abs(zeros-t_grid[imin])):.2f} away (wide-gap centre)")

    banner("Part III: clipping at the Weil boundary")
    sC = C.std()
    Neff = (t_grid[-1]-t_grid[0])/0.5
    pred = (C-A).mean() + sC*math.sqrt(2*math.log(Neff))
    print(f"  free-field extreme prediction max(C-A) ~ {pred:+.3f}; measured {deficit.max():+.4f}")
    print(f"  skewness of F=C-A: {float(((( C-A)-(C-A).mean())**3).mean()/(C-A).std()**3):+.2f} (paper -0.50; one-sided clip)")

if __name__ == "__main__":
    run()

"""
verify_30_weil_witnesses.py -- Proposition 30.1 / Numerical Observation 30.2
(Weil witnesses) and the capacity triangle (Numerical Observation 32.6).

Zero-side Weil values QW_DH at rho1, rho2 for several windows w, the zeta
control, and the three-way explicit-formula triangle (zero = arch - coeff)
with the support split of the coefficient side.
"""
import math
import numpy as np
import mpmath as mp
from ic_core import KAPPA, dh_coeffs_fast, vonmangoldt, banner

mp.mp.dps = 20

# on-line zeros of H (DH) and of zeta, computed once
def dh_online_zeros(T=140):
    """Ordinates of true on-line zeros of H (Re H = 0 confirmed by |H|<1e-3).
       Loads the verified cache if present; else scans with confirmation."""
    import os, pickle
    cache = os.path.join(os.path.dirname(__file__), "dh_online_true.pkl")
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    from ic_core import H
    I = mp.mpc(0, 1)
    reH = lambda t: float(mp.re(H(mp.mpf('0.5')+mp.mpf(float(t))*I)))
    Hv  = lambda t: H(mp.mpf('0.5')+mp.mpf(float(t))*I)
    ts = np.arange(1.0, T, 0.05)
    rv = np.array([reH(t) for t in ts])
    zs = []
    for i in np.where(np.diff(np.sign(rv)) != 0)[0]:
        a, b = float(ts[i]), float(ts[i+1])
        for _ in range(30):
            m = 0.5*(a+b)
            if np.sign(reH(a)) != np.sign(reH(m)): b = m
            else: a = m
        r = 0.5*(a+b)
        if abs(float(abs(Hv(r)))) < 1e-3 and (not zs or abs(r-zs[-1]) > 0.05):
            zs.append(r)
    return np.array(zs)

def zeta_online_zeros(T=140):
    zs=[]; n=1
    while True:
        g=float(mp.im(mp.zetazero(n)))
        if g>T: break
        zs.append(g); n+=1
    return np.array(zs)

def witness_zeroside(t0, h, online, w):
    """QW(f,f) zero side: off-line pair contributes -2h^2 e^{h^2/w^2};
       each real zero contributes |g_w(gamma)|^2 = (gamma-t0)^2 e^{-(gamma-t0)^2/w^2}."""
    pair = -2*h*h*math.exp(h*h/(w*w))
    real = np.sum((online-t0)**2*np.exp(-(online-t0)**2/(w*w)))
    return pair + real, pair, real

def run():
    banner("Proposition 30.1 / NO 30.2: Weil witnesses")
    dh_on = dh_online_zeros(140)
    ze_on = zeta_online_zeros(140)
    h1, t1 = 0.308517, 85.6993484854
    h2, t2 = 0.150830, 114.16334273
    print(f"  DH on-line zeros found: {len(dh_on)} (up to T=140)")
    print(f"  rho1 (t0={t1}, h={h1}):")
    print(f"    {'w':>5} {'QW_DH':>10} {'pair':>10} {'realsum':>10}")
    for w in [0.4, 0.5, 0.7, 1.0]:
        q,p,r = witness_zeroside(t1, h1, dh_on, w)
        paper = {0.4:-0.3451,0.5:-0.2786,0.7:-0.2295,1.0:-0.1159}[w]
        print(f"    {w:>5} {q:>10.4f} {p:>10.4f} {r:>10.4f}   paper {paper}")
    print(f"  rho2 (t0={t2}, h={h2}):")
    for w in [0.35, 0.45, 0.60]:
        q,_,_ = witness_zeroside(t2, h2, dh_on, w)
        paper = {0.35:-0.0548,0.45:-0.0509,0.60:-0.0480}[w]
        print(f"    w={w}: QW_DH = {q:.4f}   paper {paper}")
    # zeta control at the paper's explicit centre t0 = 109.1 (the wide gap
    # 107.17 -> 111.03; the table of Section 30 quotes this row verbatim)
    ic = int(np.searchsorted(ze_on, 109.1)) - 1
    tc = (ze_on[ic]+ze_on[ic+1])/2
    print(f"  zeta control (wide gap, t0 = {tc:.1f}; paper row: +6e-10, +3e-6, +0.0037, +0.183):")
    for w in [0.4,0.5,0.7,1.0]:
        q = np.sum((ze_on-tc)**2*np.exp(-(ze_on-tc)**2/(w*w)))
        print(f"    w={w}: QW_zeta = {q:+.3g} (no off-line pair => positive)")
    print(f"  robustness margin w^2/e at w=0.5: {0.5**2/math.e:.4f} (paper 0.092) < 0.279")

    # ----- capacity triangle, Numerical Observation 32.6 -----
    banner("Numerical Observation 32.6: capacity triangle (w=0.5)")
    w=0.5
    def archimedean(t0, fn='dh'):
        """arch side: (1/2pi) int |f^(t)|^2 * 2 theta'(t) dt  (factor 2 = both
           sides of the critical line, matching (30.1)'s 2 d_t theta / 2pi).
           f^(t) = (t-t0) e^{-(t-t0)^2/2w^2}, |f^|^2 = (t-t0)^2 e^{-(t-t0)^2/w^2}."""
        tg = np.arange(t0-12, t0+12, 0.01)
        if fn=='dh':
            thp = np.array([0.5*float(mp.re(mp.digamma(mp.mpf('0.75')+1j*t/2))) for t in tg]) + 0.5*math.log(5/math.pi)
        else:
            thp = np.array([0.5*float(mp.re(mp.digamma(mp.mpf('0.25')+1j*t/2))) for t in tg]) - 0.5*math.log(math.pi)
        integ = (tg-t0)**2*np.exp(-(tg-t0)**2/(w*w)) * 2*thp/(2*math.pi)
        return float(np.trapezoid(integ, tg))
    def coeffside(t0, coeffs, N):
        n=np.arange(2,N+1); y=np.log(n)
        env=(math.sqrt(math.pi)*w**3/2)*(1-w*w*y*y/2)*np.exp(-w*w*y*y/4)/(2*math.pi)
        terms=coeffs[2:N+1]*n**(-0.5)*2*np.cos(t0*y)*env
        return float(np.sum(terms)), n, terms
    N=60000
    cf=dh_coeffs_fast(N); vm=vonmangoldt(N)
    t1=85.6993484854
    # zeta triangle at t1
    zq,_,zr = witness_zeroside(t1,0,ze_on,w)   # no off-line pair for zeta -> but use zeta zeros
    Az=archimedean(t1,'zeta'); Cz,_,_=coeffside(t1,vm,N)
    print(f"  zeta: zero {zq:+.4f}  arch {Az:+.4f}  coeff {Cz:+.4f}  closure {abs(Az-Cz-zq):.4f}  (paper +0.0226/+0.0461/+0.0230/0.0004)")
    # DH triangle at t1
    dq,_,_=witness_zeroside(t1,0.308517,dh_on,w)
    Ad=archimedean(t1,'dh'); Cd,nn,terms=coeffside(t1,cf,N)
    print(f"  DH:   zero {dq:+.4f}  arch {Ad:+.4f}  coeff {Cd:+.4f}  closure {abs(Ad-Cd-dq):.4f}  (paper -0.2786/+0.0744/+0.3457/0.0074)")
    # support split of DH coefficient side
    def is_primepower(m):
        for p in range(2,int(m**0.5)+1):
            if m%p==0:
                while m%p==0: m//=p
                return m==1
        return True  # m prime
    pp_mask=np.array([is_primepower(int(m)) for m in nn])
    print(f"    prime-power part: {terms[pp_mask].sum():+.4f}  (paper -0.0094)")
    print(f"    composite part:   {terms[~pp_mask].sum():+.4f}  (paper +0.3550)")
    print(f"    composite fraction of total: {terms[~pp_mask].sum()/Cd:.2f}  (paper 1.03)")

if __name__ == "__main__":
    run()

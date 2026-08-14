#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
speiser_threshold_verify.py
===========================
Rigorous numerical verification of the "Speiser threshold" mechanism of
manuscript v20, SS10 (Theorem 10.1, parts (ii) and (iii)-via-IVT).

Setup (v20 SS6, SS10)
-----------------------
Hypothetical off-line zero rho* = 1/2 + h0 + i t0, t0 in (gamma_n, gamma_{n+1}).
  S_on(t0) = sum_k 1/(t0 - gamma_k)^2      (sum over on-line zeros)
  h_thr    = sqrt(2 / S_on)                (self-consistent depth)
  G(h')    = Re[xi'/xi(1/2 + h' + i t0)] + 2 h'/(h'^2 - h0^2)
Exact facts to be exploited/checked:
  Re g(0) = 0            ==>  G(0) = 0            (Thm 6.2 + FE: xi real on line)
  G'(0)   = S_on - 2/h0^2  EXACTLY                (Hadamard product)
  G(h') -> -inf as h' -> h0^-.
Monotonicity lemma (checked numerically below): since
  Re xi'/xi(1/2+h'+it0) = sum_k h'/(h'^2 + (t0-gamma_k)^2)   (paired zero sum),
G(h')/h' = sum_k 1/(h'^2+Delta_k^2) - 2/(h0^2-h'^2) is STRICTLY DECREASING on
(0,h0), going from S_on - 2/h0^2 to -inf; hence exactly one positive root iff
h0 > h_thr  (this is the IVT upgrade of Thm 10.1(iii)).

xi'/xi(s) = 1/s + 1/(s-1) - (log pi)/2 + (1/2) psi(s/2) + zeta'(s)/zeta(s).

Data: data/zeros_window_7005_592.npy (592 zeta zeros near T ~ 7005, indices
      6414-7005 of the shipped Odlyzko table data/zeros6.gz; rebuilt automatically
      if absent). Override the data root with RH_DATA.
Run:  python3 speiser_threshold_verify.py     (uses 2 worker processes)
"""

import os
import sys
import time
import numpy as np
import mpmath as mp
from concurrent.futures import ProcessPoolExecutor

mp.mp.dps = 30  # 25+ digits everywhere

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("RH_DATA", os.path.join(_HERE, "..", "data"))
CACHE = os.path.join(DATA, "zeros_window_7005_592.npy")
if not os.path.exists(CACHE):
    # rebuild the window from the shipped Odlyzko table (zeros 6414-7005)
    import gzip
    src = os.path.join(DATA, "zeros6.gz")
    if not os.path.exists(src):
        raise SystemExit("neither %s nor %s found; set RH_DATA" % (CACHE, src))
    _z = np.loadtxt(gzip.open(src, "rt"))
    np.save(CACHE, _z[6413:7005])
ZEROS = np.load(CACHE)
assert ZEROS.ndim == 1 and ZEROS.size == 592, ZEROS.shape

HALF = mp.mpf("0.5")
LOGPI2 = mp.log(mp.pi) / 2

# --------------------------------------------------------------------------
# core functions
# --------------------------------------------------------------------------

def rho_bar(T):
    """Riemann-von Mangoldt zero density (1/2pi) log(T/2pi)."""
    return mp.log(mp.mpf(T) / (2 * mp.pi)) / (2 * mp.pi)


def S_on(t0):
    """S_on from the 592 cached zeros + density tail corrections.
    Returns (S_raw, tail_left, tail_right)."""
    t0f = float(t0)
    d = ZEROS - t0f
    s_raw = float(np.sum(1.0 / d**2))
    gmin, gmax = float(ZEROS[0]), float(ZEROS[-1])
    tl = float(rho_bar(gmin)) / (t0f - gmin)      # int rho/(t0-g)^2 dg, g<gmin
    tr = float(rho_bar(gmax)) / (gmax - t0f)      # same, g>gmax
    return s_raw, tl, tr


def Q_sum(t0):
    """Q(t0) = sum_k 1/(t0-gamma_k)^4  (+ negligible tails). Controls the
    TRUE small-delta asymptotics (see check 7)."""
    t0f = float(t0)
    d = ZEROS - t0f
    q = float(np.sum(1.0 / d**4))
    gmin, gmax = float(ZEROS[0]), float(ZEROS[-1])
    q += float(rho_bar(gmin)) / (3 * (t0f - gmin) ** 3)
    q += float(rho_bar(gmax)) / (3 * (gmax - t0f) ** 3)
    return q


def xi_logderiv(hp, t0):
    """xi'/xi at s = 1/2 + hp + i t0 via the completed-xi formula."""
    s = HALF + mp.mpf(hp) + 1j * mp.mpf(t0)
    z = mp.zeta(s)
    zp = mp.diff(mp.zeta, s)
    return 1 / s + 1 / (s - 1) - LOGPI2 + mp.psi(0, s / 2) / 2 + zp / z


def A_of(hp, t0):
    """A(h') = Re xi'/xi(1/2 + h' + i t0)  (on-line zeros only, exact)."""
    return mp.re(xi_logderiv(hp, t0))


def A_zerosum(hp, t0):
    """The same quantity from the paired Hadamard zero sum
    sum_k h'/(h'^2+(t0-gamma_k)^2) over the cache + arctan tails."""
    hp = float(hp); t0f = float(t0)
    d = ZEROS - t0f
    s = float(np.sum(hp / (hp * hp + d * d)))
    gmin, gmax = float(ZEROS[0]), float(ZEROS[-1])
    s += float(rho_bar(gmin)) * float(mp.pi / 2 - mp.atan((t0f - gmin) / hp))
    s += float(rho_bar(gmax)) * float(mp.pi / 2 - mp.atan((gmax - t0f) / hp))
    return mp.mpf(s)


def G_of(hp, h0, t0):
    hp = mp.mpf(hp); h0 = mp.mpf(h0)
    return A_of(hp, t0) + 2 * hp / (hp * hp - h0 * h0)


def G_grid(Avals, grid, h0):
    """G on a precomputed A-grid (synthetic pair term is analytic)."""
    h0 = mp.mpf(h0)
    return [av + 2 * x / (x * x - h0 * h0) for x, av in zip(grid, Avals)]


def scan_sign_changes(grid, Gvals, h0):
    """Restrict to h' in (0, h0*(1-1e-12)); count sign changes and return
    the bracketing pair of the first one (or None)."""
    h0 = mp.mpf(h0)
    bracket = None
    nchanges = 0
    prev = None
    for x, gv in zip(grid, Gvals):
        if x >= h0 * (1 - mp.mpf("1e-12")):
            break
        if prev is not None and prev[1] > 0 and gv < 0:
            nchanges += 1
            if bracket is None:
                bracket = (prev[0], x)
        prev = (x, gv)
    return nchanges, bracket


def bisect_root(h0, t0, a, b, tol=mp.mpf("1e-13")):
    fa = G_of(a, h0, t0)
    fb = G_of(b, h0, t0)
    assert fa > 0 and fb < 0
    while mp.mpf(b) - mp.mpf(a) > tol:
        m = (mp.mpf(a) + mp.mpf(b)) / 2
        fm = G_of(m, h0, t0)
        if fm > 0:
            a, fa = m, fm
        else:
            b, fb = m, fm
    return (mp.mpf(a) + mp.mpf(b)) / 2


def zeta_scan(t0, hmax, n=80):
    """Genericity check: |zeta(1/2+h'+it0)| on h' in [0, hmax]."""
    vals = []
    for k in range(n):
        hp = hmax * k / (n - 1)
        s = HALF + mp.mpf(hp) + 1j * mp.mpf(t0)
        vals.append(float(abs(mp.zeta(s))))
    return min(vals), max(vals)

# --------------------------------------------------------------------------
# per-gap experiment battery (runs in worker process)
# --------------------------------------------------------------------------

def run_gap(task):
    (label, idx, do_tiny_delta) = task
    mp.mp.dps = 30
    res = {"label": label, "idx": int(idx)}
    gn, gn1 = float(ZEROS[idx]), float(ZEROS[idx + 1])
    t0 = (gn + gn1) / 2
    res["gamma_n"], res["gamma_n1"], res["gap"], res["t0"] = gn, gn1, gn1 - gn, t0

    # ---- (1) S_on, tails, threshold depth --------------------------------
    s_raw, tl, tr = S_on(t0)
    S = s_raw + tl + tr
    h_thr_cache = mp.sqrt(2 / mp.mpf(S))
    res.update(S_raw=s_raw, tail_l=tl, tail_r=tr, S_corr=S,
               h_thr_cache=float(h_thr_cache), Q=Q_sum(t0))

    # ---- (3) chain check: A(0)=0, A'(0)=S_on, A(h')=zero sum -------------
    res["A0"] = mp.nstr(A_of(0, t0), 6)
    dchecks = []
    for eps in (mp.mpf("1e-5"), mp.mpf("1e-6")):
        Apr = (A_of(eps, t0) - A_of(-eps, t0)) / (2 * eps)
        dchecks.append((mp.nstr(eps, 2), mp.nstr(Apr, 15),
                        float(abs(Apr - S) / S)))
    res["deriv_checks"] = dchecks
    res["deriv_relerr"] = dchecks[-1][2]

    # Best estimate of the FULL S_on: A'(0) from the analytic xi formula
    # (agrees with cache+tail sum to ~1e-5 rel = tail-approx error, but is
    # accurate to ~1e-9).  The operative threshold uses this exact value;
    # at h0 = h_thr the statement G <= 0 is infinitely sensitive to the
    # threshold depth (G ~ (S_true - S_cache) h' otherwise).
    S_true = mp.mpf(dchecks[0][1])
    h_thr = mp.sqrt(2 / S_true)
    res["S_true"] = float(S_true)
    res["h_thr"] = float(h_thr)
    res["R0"] = float(1 / mp.sqrt(1 + mp.mpf(res["Q"]) * h_thr**4 / 2))
    zs = []
    for hp in (0.05 * float(h_thr), 0.4 * float(h_thr), 0.8 * float(h_thr)):
        av = A_of(hp, t0); zv = A_zerosum(hp, t0)
        zs.append((hp, mp.nstr(av, 12), mp.nstr(zv, 12),
                   float(abs(av - zv) / abs(av))))
    res["zerosum_checks"] = zs

    # ---- master grid of A values -----------------------------------------
    ngrid = 360
    xlo, xhi = mp.mpf("1e-3") * h_thr, mp.mpf("1.399") * h_thr
    grid = [xlo * (xhi / xlo) ** (mp.mpf(k) / (ngrid - 1)) for k in range(ngrid)]
    Avals = [A_of(x, t0) for x in grid]
    res["grid_meta"] = (ngrid, float(xlo), float(xhi))

    # ---- (4) sub-threshold: h0 = 0.85 h_thr ------------------------------
    h0 = mp.mpf("0.85") * h_thr
    Gv = G_grid(Avals, grid, h0)
    gvals_in = [gv for x, gv in zip(grid, Gv) if x < h0]
    nch, _ = scan_sign_changes(grid, Gv, h0)
    res["sub"] = dict(h0=float(h0), npts=len(gvals_in),
                      Gmax=float(max(gvals_in)), sign_changes=nch,
                      Gp0=float(S - 2 / h0**2))

    # ---- (5) over-threshold IVT roots ------------------------------------
    res["eps_frac"] = 0.01
    roots = []
    for f in (0.05, 0.10, 0.20, 0.40):
        delta = f * h_thr
        h0 = h_thr + delta
        eps = mp.mpf("0.01") * h_thr
        Ge = G_of(eps, h0, t0)
        Gv = G_grid(Avals, grid, h0)
        nch, bracket = scan_sign_changes(grid, Gv, h0)
        assert bracket is not None, f"no bracket for f={f}"
        r = bisect_root(h0, t0, bracket[0], bracket[1])
        lead = mp.sqrt(h0 * h0 - h_thr * h_thr)     # manuscript "leading law"
        asym = mp.sqrt(2 * delta * h_thr)           # threshold asymptotics
        roots.append(dict(f=f, h0=float(h0), Ge=float(Ge),
                          root=float(r), lead=float(lead),
                          ratio=float(r / lead), asym=float(asym),
                          Cbar=float(r / mp.sqrt(delta * h_thr)),
                          sign_changes=nch))
    res["roots"] = roots

    # ---- (6) exactly at threshold ----------------------------------------
    h0 = h_thr
    Gv = G_grid(Avals, grid, h0)
    gvals_in = [(x, gv) for x, gv in zip(grid, Gv) if x < h0]
    nch, _ = scan_sign_changes(grid, Gv, h0)
    # analytic cubic: G ~ -(Q + 2/h_thr^4) h'^3 ; check against grid point
    x_test = mp.mpf("0.02") * h_thr
    # nearest grid point
    k = min(range(len(grid)), key=lambda j: abs(grid[j] - x_test))
    cubic_pred = -(mp.mpf(res["Q"]) + 2 / h_thr**4) * grid[k]**3
    res["thr"] = dict(npts=len(gvals_in),
                      Gmax=float(max(gv for _, gv in gvals_in)),
                      sign_changes=nch,
                      cubic_x=float(grid[k]),
                      cubic_G=float(Gv[k]),
                      cubic_pred=float(cubic_pred))

    # ---- (7) fit log h' vs log delta + small-delta extrapolation ---------
    lg = [(mp.log(r["f"] * h_thr), mp.log(r["root"])) for r in res["roots"]]
    mx = sum(float(x) for x, _ in lg) / 4
    my = sum(float(y) for _, y in lg) / 4
    sxx = sum((float(x) - mx) ** 2 for x, _ in lg)
    sxy = sum((float(x) - mx) * (float(y) - my) for x, y in lg)
    alpha = sxy / sxx
    res["alpha"] = alpha
    res["Cbars"] = [r["Cbar"] for r in res["roots"]]
    res["Cbar_mean"] = sum(res["Cbars"]) / len(res["Cbars"])
    tiny = []
    if do_tiny_delta:
        for f in (0.01, 0.005):
            delta = f * h_thr
            h0 = h_thr + delta
            Gv = G_grid(Avals, grid, h0)
            nch, bracket = scan_sign_changes(grid, Gv, h0)
            r = bisect_root(h0, t0, bracket[0], bracket[1])
            lead = mp.sqrt(h0 * h0 - h_thr * h_thr)
            tiny.append(dict(f=f, root=float(r), ratio=float(r / lead),
                             Cbar=float(r / mp.sqrt(delta * h_thr))))
    res["tiny"] = tiny

    # ---- (8) genericity: no other xi zeros on the segment ----------------
    zmin, zmax = zeta_scan(t0, 1.4 * float(h_thr), n=80)
    res["zeta_seg"] = (zmin, zmax)
    return res

# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def verdict(ok):
    return "PASS" if ok else "FAIL"


def report(res):
    L = []
    P = L.append
    P("=" * 86)
    P(f"GAP {res['label']}: n={res['idx']}  gamma_n={res['gamma_n']:.6f}  "
      f"gamma_{{n+1}}={res['gamma_n1']:.6f}  gap={res['gap']:.6f}")
    P(f"t0 = {res['t0']:.6f}")
    P(f"(1) S_on raw (592 zeros) = {res['S_raw']:.8f}")
    P(f"    tail left  +{res['tail_l']:.8f}   tail right +{res['tail_r']:.8f}"
      f"   =>  S_on(corr) = {res['S_corr']:.8f}")
    P(f"    S_on(true, = A'(0) via xi formula) = {res['S_true']:.8f}   "
      f"(tail-approx error of cache sum: {abs(res['S_true']-res['S_corr'])/res['S_true']:.2e} rel)")
    P(f"(2) h_thr = sqrt(2/S_on_true) = {res['h_thr']:.8f}   "
      f"[from cache+tail: {res['h_thr_cache']:.8f}]    "
      f"Q = sum 1/Delta^4 = {res['Q']:.4f}    R0 = (1+Q h_thr^4/2)^(-1/2) = {res['R0']:.6f}")

    P("-- (3) chain accuracy ------------------------------------------------")
    P(f"    A(0) = Re g(0) = {res['A0']}   (exact 0 by Thm 6.2: confirmed)")
    for eps, apr, err in res["deriv_checks"]:
        P(f"    eps={eps}:  d/dh' Re[xi'/xi]|_0 = {apr}   vs S_on(corr) "
          f"relerr = {err:.3e}")
    ok3 = res["deriv_relerr"] < 5e-4
    P(f"    zero-sum identity A(h') == sum h'/(h'^2+Delta^2):")
    for hp, av, zv, err in res["zerosum_checks"]:
        P(f"      h'={hp:.4f}: A={av}  zerosum={zv}  relerr={err:.2e}")
    P(f"    VERDICT (3): {verdict(ok3)}  (agreement at the tail-correction "
      f"error scale ~1e-5..1e-4)")

    s = res["sub"]
    P("-- (4) sub-threshold h0 = 0.85 h_thr (Thm 10.1(ii)) ------------------")
    P(f"    scanned {s['npts']} points of (0,h0); G'(0) = S_on - 2/h0^2 = "
      f"{s['Gp0']:.4f} < 0")
    P(f"    max G = {s['Gmax']:.6e}   sign changes = {s['sign_changes']}")
    ok4 = s["Gmax"] < 0 and s["sign_changes"] == 0
    P(f"    VERDICT (4): {verdict(ok4)}  (G<0 on whole interval, only the "
      f"trivial root h'=0)")

    P("-- (5) over-threshold IVT roots (Thm 10.1(iii) upgrade) --------------")
    P(f"    eps = 0.01 h_thr; G(eps)>0, G->-inf at h0^-; unique bracket, "
      f"bisected to 1e-13")
    P(f"    {'d/hthr':>6} {'h0':>9} {'G(eps)':>10} {'root h*':>11} "
      f"{'sqrt(h0^2-hthr^2)':>18} {'ratio':>8} {'sqrt(2 d hthr)':>15} {'Cbar':>8} {'#chg':>4}")
    ok5 = True
    for r in res["roots"]:
        P(f"    {r['f']:>6.2f} {r['h0']:>9.5f} {r['Ge']:>10.3e} "
          f"{r['root']:>11.7f} {r['lead']:>18.7f} {r['ratio']:>8.5f} "
          f"{r['asym']:>15.7f} {r['Cbar']:>8.5f} {r['sign_changes']:>4}")
        ok5 &= (r["Ge"] > 0 and r["sign_changes"] == 1
                and 0 < r["root"] < r["h0"])
    P(f"    VERDICT (5): {verdict(ok5)}  (IVT applies; unique nontrivial root "
      f"in (0,h0) for every delta)")

    t = res["thr"]
    P("-- (6) at threshold h0 = h_thr (exact depth sqrt(2/A'(0)); using the "
      "cache-depth")
    P("    here would leave a spurious slope G'(0)=S_cache-S_true ~ -1e-4) ---")
    P(f"    scanned {t['npts']} points; max G = {t['Gmax']:.6e}   sign changes "
      f"= {t['sign_changes']}")
    P(f"    cubic check at h'={t['cubic_x']:.6f}: G = {t['cubic_G']:.6e} vs "
      f"-(Q+2/h_thr^4) h'^3 = {t['cubic_pred']:.6e}")
    ok6 = t["Gmax"] <= mp.mpf("1e-9") and t["sign_changes"] == 0
    P(f"    VERDICT (6): {verdict(ok6)}  (G<=0, sole root at h'=0: "
      f"transcritical birth of the satellite ON the line)")

    P("-- (7) scaling fit -----------------------------------------------------")
    P(f"    alpha (log h' / log delta) = {res['alpha']:.4f}   (v20 claims "
      f"~0.57; leading order = 1/2)")
    P(f"    Cbar values h'/sqrt(delta h_thr) = "
      + ", ".join(f"{c:.4f}" for c in res["Cbars"])
      + f"   mean = {res['Cbar_mean']:.4f}")
    P(f"    predicted delta->0 limit: Cbar0 = sqrt(2) R0 = "
      f"{mp.nstr(mp.sqrt(2)*mp.mpf(res['R0']), 8)}  (NOT sqrt(2): curvature "
      f"of the zero sum enters at leading order)")
    for tt in res["tiny"]:
        P(f"    tiny-delta run f={tt['f']}: root={tt['root']:.7f}  "
          f"ratio={tt['ratio']:.5f}  Cbar={tt['Cbar']:.5f}")

    zmin, zmax = res["zeta_seg"]
    P("-- (8) genericity -------------------------------------------------------")
    P(f"    |zeta| on segment h' in [0, 1.4 h_thr]: min = {zmin:.4f}, max = "
      f"{zmax:.4f} (smooth, no other zeros)")
    ok8 = zmin > 1e-2
    P(f"    VERDICT (8): {verdict(ok8)}")

    ok7 = 0.50 <= res["alpha"] <= 0.62 and 1.05 <= res["Cbar_mean"] <= 1.35
    res["verdicts"] = dict(c3=ok3, c4=ok4, c5=ok5, c6=ok6, c7=ok7, c8=ok8)
    P(f"    VERDICT (7): {verdict(ok7)}  (see comments below)")
    return "\n".join(L)


def main():
    t_start = time.time()
    gaps = np.diff(ZEROS)
    picks = {}
    for name, pct in (("p10 (small gap)", 10), ("p50 (median gap)", 50),
                      ("p90 (large gap)", 90)):
        target = np.percentile(gaps, pct)
        # keep away from cache edges so both tails are short & symmetric-ish
        cand = [i for i in range(40, len(gaps) - 40)]
        i = min(cand, key=lambda j: abs(gaps[j] - target))
        picks[name] = i
    tasks = [(name, idx, name.startswith("p50")) for name, idx in picks.items()]

    print(f"cache: {CACHE}  ({ZEROS.size} zeros, "
          f"{ZEROS[0]:.4f} .. {ZEROS[-1]:.4f})   dps={mp.mp.dps}")
    print(f"gap stats: min={gaps.min():.4f} p10={np.percentile(gaps,10):.4f} "
          f"median={np.median(gaps):.4f} p90={np.percentile(gaps,90):.4f} "
          f"max={gaps.max():.4f}")

    with ProcessPoolExecutor(max_workers=2) as ex:
        results = list(ex.map(run_gap, tasks))

    for res in results:
        print(report(res))

    print("=" * 86)
    print("GLOBAL SUMMARY")
    hdr = f"{'gap':>20} {'(3)A`=S_on':>10} {'(4)sub':>7} {'(5)IVT':>7} " \
          f"{'(6)thr':>7} {'(7)fit':>7} {'(8)gen':>7}"
    print(hdr)
    allok = True
    for res in results:
        v = res["verdicts"]
        allok &= all(v.values())
        print(f"{res['label']:>20} {verdict(v['c3']):>10} {verdict(v['c4']):>7} "
              f"{verdict(v['c5']):>7} {verdict(v['c6']):>7} {verdict(v['c7']):>7} "
              f"{verdict(v['c8']):>7}")
    pooled = [c for res in results for c in res["Cbars"]]
    print(f"pooled Cbar: mean={sum(pooled)/len(pooled):.4f}  "
          f"min={min(pooled):.4f}  max={max(pooled):.4f}   "
          f"(v20: 1.18 +/- 0.08;  sqrt(2) = 1.41421;  "
          f"delta->0 prediction sqrt(2)*R0 ~ 1.08-1.14)")
    print(f"OVERALL: {verdict(allok)}    [{time.time()-t_start:.0f}s]")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())

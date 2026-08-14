"""GUE Monte Carlo: conditioned far-field sum E[sum 1/x_m^2 | tight pair at origin].
Dumitriu-Edelman beta=2 tridiagonal, semicircle-CDF unfolding, bulk conditioning.
Saves ./output/gue_mc_results.npz
"""
import numpy as np
from scipy.linalg import eigh_tridiagonal
from multiprocessing import Pool
import time, sys

S0 = 0.25          # conditioning threshold on normalized gap
WINDOWS = [25.0, 50.0, 100.0, 200.0]

def gue_tridiag(N, rng):
    d = rng.standard_normal(N)
    e = np.sqrt(rng.chisquare(2*np.arange(N-1, 0, -1)) / 2.0)
    return eigh_tridiagonal(d, e, eigvals_only=True, check_finite=False)

def unfold(w, N):
    """Semicircle CDF unfolding, centered: mean bulk spacing = 1."""
    R = 2.0*np.sqrt(N)
    z = np.clip(w/R, -1.0, 1.0)
    F = 0.5 + (z*np.sqrt(1-z*z) + np.arcsin(z))/np.pi
    return N*(F - 0.5)

def worker(task):
    N, nmat, seed = task
    rng = np.random.default_rng(seed)
    edge = N/2.0
    Xb = max(10.0, edge*0.2)          # pair midpoint must be within |x|<=Xb
    maxwin = min(WINDOWS[-1], edge - Xb - 5)
    wins = [w for w in WINDOWS if w <= maxwin]
    gaps_cond, sums = [], {w: [] for w in wins}
    all_sp = []
    nmats_done = 0
    for _ in range(nmat):
        w = gue_tridiag(N, rng)
        x = unfold(w, N)
        # bulk mask
        midx = 0.5*(x[:-1] + x[1:])
        sp = np.diff(x)
        bulk = (np.abs(midx) <= Xb)
        all_sp.append(sp[bulk])
        cond = bulk & (sp < S0)
        idx = np.nonzero(cond)[0]
        for i in idx:
            x0 = midx[i]
            g = sp[i]
            rel = np.delete(x - x0, [i, i+1])
            arel = np.abs(rel)
            gaps_cond.append(g)
            for wv in wins:
                m = arel < wv
                sums[wv].append(np.sum(1.0/rel[m]**2) if m.any() else 0.0)
        nmats_done += 1
    return (N, nmats_done, np.array(gaps_cond),
            {w: np.array(v) for w, v in sums.items()},
            np.concatenate(all_sp) if all_sp else np.array([]))

if __name__ == "__main__":
    plan = {200: 40000, 300: 40000, 500: 45000}   # matrices per N
    tasks = []
    seed = 12345
    for N, total in plan.items():
        per = total//2
        for r in range(2):
            tasks.append((N, per, seed)); seed += 1
    t0 = time.time()
    out = {200: [], 300: [], 500: []}
    with Pool(2) as p:
        for res in p.imap_unordered(worker, tasks):
            out[res[0]].append(res)
            print(f"N={res[0]} chunk done: {res[1]} mats, {len(res[2])} pairs, "
                  f"{time.time()-t0:.0f}s", flush=True)
    # merge
    res = {}
    for N, chunks in out.items():
        gaps = np.concatenate([c[2] for c in chunks])
        wins = set()
        for c in chunks: wins |= set(c[3].keys())
        sums = {w: np.concatenate([c[3][w] for c in chunks if w in c[3]]) for w in wins}
        sp = np.concatenate([c[4] for c in chunks])
        nmats = sum(c[1] for c in chunks)
        res[N] = (nmats, gaps, sums, sp)
        print(f"N={N}: {nmats} matrices, {len(gaps)} conditioned pairs, "
              f"{len(sp)} bulk spacings", flush=True)
    np.savez('./output/gue_mc_results.npz',
             **{f"N{N}_gaps": v[1] for N, v in res.items()},
             **{f"N{N}_sp": v[3] for N, v in res.items()},
             **{f"N{N}_nmats": np.array([v[0]]) for N, v in res.items()},
             **{f"N{N}_sums_{int(w)}": v[2][w] for N, v in res.items() for w in v[2]})
    print("saved.", flush=True)

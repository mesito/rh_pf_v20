#!/usr/bin/env python3
"""smooth_cutoff_scan.py — re-run the Euler-exponent fit b(X) with SMOOTH
prime cutoffs, to test whether the drift b: 0.752->0.944 (X: 1e2->1e7)
is an artifact of the SHARP cutoff p<=X.

Cutoff choices (both standard):
  sharp : Phi(u) = 1{u<=1}, u = p/X                       (baseline re-fit)
  exp4  : Phi(u) = exp(-u^4),            truncated p <= 4X (Phi(4)=e^-256)
  bump  : C^inf bump, Phi(u)=1 for u<=1/2,
          Phi(u)=exp(1 - 1/(1-(2u-1)^2)) for 1/2<u<1, 0 for u>=1.

L_X^Phi(t) = sum_{k=1..3} (1/k) sum_p Phi(p/X) p^{-k/2} cos(k t ln p)
Phases: longdouble t*ln(p) mod 2pi -> float64 cos (same as verify_all.py).
Data: 300 log-spaced Lehmer pairs @ 8.4e9 (lehmer_floors_8p4e9.npy), floors
reused; only L_X recomputed.  Fit: ln floor = a ln g + b L_X^Phi + c.
"""

import time
import os
import numpy as np

DATA = os.environ.get("RH_DATA","../data")
TWO_PI_LD = np.longdouble(2 * np.pi)
XCUTS = [10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7]
PMAX = 4 * 10 ** 7


def sieve(n):
    b = np.ones(n + 1, bool)
    b[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if b[i]:
            b[i * i::i] = False
    return np.flatnonzero(b)


def ols(Xm, y):
    A = np.column_stack([Xm, np.ones(len(y))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yh = A @ coef
    res = y - yh
    r2 = 1.0 - float(np.sum(res ** 2) / np.sum((y - y.mean()) ** 2))
    sig2 = float(np.sum(res ** 2) / (len(y) - A.shape[1]))
    se = np.sqrt(np.diag(sig2 * np.linalg.inv(A.T @ A)))
    return coef, se, r2


print("loading data ...")
zl = np.load(DATA + "/lmfdb_zeros_parsed.npy")
zll = zl.astype(np.longdouble)
floors = np.load(DATA + "/lehmer_floors_8p4e9.npy")   # (idx,t0,g,floor,far)
fidx = floors[:, 0].astype(int) - 1
t0_pairs = (zll[fidx] + zll[fidx + 1]) / 2            # longdouble midpoints
g_pair = floors[:, 2]
fl_pair = floors[:, 3]
lf = np.log(fl_pair)
lg = np.log(g_pair)
N = len(lf)

print("sieving primes to %d ..." % PMAX)
PR = sieve(PMAX)
LNPR = np.log(PR.astype(np.longdouble))
P64 = PR.astype(float)
SQ = np.sqrt(P64)
print("  %d primes" % len(PR))

# ------------------------- weight vectors per (choice, X) -------------------
def phi_exp4(u):
    return np.exp(-(u ** 4))


def phi_bump(u):
    out = np.zeros_like(u)
    lo = u <= 0.5
    out[lo] = 1.0
    mid = (u > 0.5) & (u < 1.0)
    x = 2.0 * u[mid] - 1.0
    out[mid] = np.exp(1.0 - 1.0 / (1.0 - x ** 2))
    return out


SETS = []    # (label, X, idx_hi, w1, w2, w3)
for X in XCUTS:
    for label, phi, umax in (("sharp", None, 1.0),
                             ("exp4", phi_exp4, 4.0),
                             ("bump", phi_bump, 1.0)):
        hi = np.searchsorted(PR, int(umax * X), side="right")
        p = P64[:hi]
        w = np.ones(hi) if phi is None else phi(p / X)
        SETS.append((label, X, hi,
                     w / np.sqrt(p), 0.5 * w / p, (w / 3.0) / p ** 1.5))

LX = {(label, X): np.empty(N) for label, X, _, _, _, _ in SETS}

# ------------------------- main loop over t chunks ---------------------------
CH = 8
t0m = time.time()
for c0 in range(0, N, CH):
    ts = t0_pairs[c0:c0 + CH]
    c1 = np.empty((len(PR), len(ts)))
    for j, tt in enumerate(ts):
        c1[:, j] = np.cos(np.asarray((tt * LNPR) % TWO_PI_LD, dtype=float))
    c2 = 2.0 * c1 ** 2 - 1.0
    c3 = 4.0 * c1 ** 3 - 3.0 * c1
    for label, X, hi, w1, w2, w3 in SETS:
        LX[(label, X)][c0:c0 + len(ts)] = (w1 @ c1[:hi] + w2 @ c2[:hi]
                                           + w3 @ c3[:hi])
    del c1, c2, c3
    print("  chunk %d/%d (%.1fs)" % (c0 + CH, N, time.time() - t0m),
          flush=True)

# ------------------------- fits ----------------------------------------------
rows_out = []
print("\n%-6s %-6s %-16s %-16s %s" % ("cut", "X", "a (se)", "b (se)", "R2"))
for label, X, _, _, _, _ in SETS:
    (a, b, c), se, r2 = ols(np.column_stack([lg, LX[(label, X)]]), lf)
    rows_out.append((label, X, a, se[0], b, se[1], r2))
    print("%-6s 1e%-5d %.4f (%.4f)   %.4f (%.4f)   %.4f"
          % (label, int(np.log10(X)), a, se[0], b, se[1], r2))

import csv
with open(DATA + "/smooth_cutoff_scan.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["cutoff", "X", "a", "se_a", "b", "se_b", "R2"])
    for r in rows_out:
        w.writerow([r[0], r[1], "%.6f" % r[2], "%.6f" % r[3],
                    "%.6f" % r[4], "%.6f" % r[5], "%.6f" % r[6]])
print("\nwrote " + DATA + "/smooth_cutoff_scan.csv")

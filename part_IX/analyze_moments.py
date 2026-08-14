#!/usr/bin/env python3
"""analyze_moments.py -- blocking/averaging mediation test.

Reads data/moment_allgaps.npy (gap_idx, t_mid, g, floor, L_X), already
sorted by t.  For B in {1,16,256,4096}: form blocks of B consecutive gaps,
average log floor, log g, L_X within blocks; OLS
  logfloor_block = a*logg_block + b*LX_block + c
plus partial corr(logfloor, LX | logg).  Writes
  data/moment_mediation_scan.csv   rows: B,n_blocks,a,b,se_b,R2,partial_corr
"""
import os
import numpy as np

DATA = os.environ.get("RH_DATA","../data")
a = np.load(DATA + "/moment_allgaps.npy")
g, fl, LX = a[:, 2], a[:, 3], a[:, 4]
assert np.all(fl > 0) and np.all(g > 0)
lf, lg = np.log(fl), np.log(g)
n = len(lf)
print("n =", n)


def ols_full(Xm, y):
    A = np.column_stack([Xm, np.ones(len(y))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef
    r2 = 1.0 - float(res @ res / np.sum((y - y.mean()) ** 2))
    k = A.shape[1]
    sig2 = float(res @ res / (len(y) - k))
    se = np.sqrt(np.diag(sig2 * np.linalg.inv(A.T @ A)))
    return coef, se, r2, res


def resid_on(x, c):
    A = np.column_stack([c, np.ones(len(c))])
    cf, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ cf


rows = []
for B in (1, 16, 256, 4096):
    nb = n // B
    m = nb * B
    lfb = lf[:m].reshape(nb, B).mean(1)
    lgb = lg[:m].reshape(nb, B).mean(1)
    LXb = LX[:m].reshape(nb, B).mean(1)
    (aa, bb, cc), se, r2, _ = ols_full(np.column_stack([lgb, LXb]), lfb)
    pc = float(np.corrcoef(resid_on(lfb, lgb), resid_on(LXb, lgb))[0, 1])
    rows.append((B, nb, aa, bb, se[1], r2, pc))
    print("B=%-5d n=%-7d a=%.4f b=%.4f se_b=%.4f R2=%.4f pcorr=%.4f"
          % (B, nb, aa, bb, se[1], r2, pc))

hdr = "B,n_blocks,a,b,se_b,R2,partial_corr"
np.savetxt(DATA + "/moment_mediation_scan.csv", np.array(rows),
           delimiter=",", header=hdr, comments="",
           fmt=["%d", "%d", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f"])
print("wrote moment_mediation_scan.csv")

# extra context for the report: g-only R2 per B, sd of LX blocks
for B in (1, 16, 256, 4096):
    nb = n // B
    m = nb * B
    lfb = lf[:m].reshape(nb, B).mean(1)
    lgb = lg[:m].reshape(nb, B).mean(1)
    LXb = LX[:m].reshape(nb, B).mean(1)
    _, _, r2g, _ = ols_full(lgb[:, None], lfb)
    print("B=%-5d R2(g only)=%.4f  sd(LXb)=%.4f  corr(LXb,lgb)=%.4f"
          % (B, r2g, LXb.std(), np.corrcoef(LXb, lgb)[0, 1]))

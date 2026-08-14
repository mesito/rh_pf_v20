#!/usr/bin/env python3
"""compute_lx.py -- L_X(t_mid) at X=1e6 for all retained gap midpoints,
then the blocking/averaging mediation analysis.  Writes
  data/moment_allgaps.npy            (gap_idx, t_mid, g, floor, L_X)
  data/moment_mediation_scan.csv
"""
import time
import os
import numpy as np

DATA = os.environ.get("RH_DATA","../data")
PI_LD = np.longdouble("3.14159265358979323846264338327950288")
TWO_PI_LD = 2 * PI_LD
T0 = 8436146000.0
X = 10 ** 6


def sieve(n):
    b = np.ones(n + 1, bool)
    b[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if b[i]:
            b[i * i::i] = False
    return np.flatnonzero(b)


PR = sieve(X).astype(float)
LNP = np.log(PR)                                   # float64 (d*v fine)
U = np.asarray((np.longdouble(T0) * np.log(PR.astype(np.longdouble)))
               % TWO_PI_LD, dtype=float)
W1 = PR ** -0.5
W2 = PR ** -1.0
W3 = PR ** -1.5

h0 = np.load(DATA + "/moment_floors_half0.npy")
h1 = np.load(DATA + "/moment_floors_half1.npy")
allrows = np.vstack([h0, h1])
allrows = allrows[np.argsort(allrows[:, 0])]
print("combined rows:", len(allrows))
# integrity: gap indices must be strictly increasing, spacing 4
gi = allrows[:, 0].astype(np.int64)
assert np.all(np.diff(gi) == 4)

tmid = allrows[:, 1]
LX = np.empty(len(tmid))
M = 256
t0 = time.time()
for c0 in range(0, len(tmid), M):
    d = tmid[c0:c0 + M] - T0
    c1 = np.cos(U[None, :] + np.outer(d, LNP))
    c2 = 2.0 * c1 ** 2 - 1.0
    c3 = 4.0 * c1 ** 3 - 3.0 * c1
    LX[c0:c0 + len(d)] = (c1 @ W1 + 0.5 * (c2 @ W2) + (1.0 / 3.0) * (c3 @ W3))
    if (c0 // M) % 100 == 0:
        print("L_X %d/%d  %.0fs" % (c0, len(tmid), time.time() - t0),
              flush=True)
print("L_X done in %.0fs" % (time.time() - t0))

out = np.column_stack([allrows, LX])
np.save(DATA + "/moment_allgaps.npy", out)
print("saved moment_allgaps.npy", out.shape)

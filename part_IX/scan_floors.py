#!/usr/bin/env python3
"""scan_floors.py <half> -- max|Z| on interiors of every-4th gap of the
LMFDB 8.4e9 ensemble (offset from fixed seed), using a vectorized Z_rs2
with accurate longdouble pi constants (validated vs mpmath dps=50 to ~5e-9).

Writes checkpoint npz per chunk and a final npy:
  " + os.environ.get("RH_DATA","../data") + "/moment_floors_half{h}.npy  columns: gap_idx, t_mid, g, floor
"""
import sys, os, time
import os
import numpy as np

H = int(sys.argv[1])
DATA = os.environ.get("RH_DATA","../data")
K = 12                      # interior samples per gap
BATCH = 512                 # points per vectorized batch
CHUNK = 4096                # gaps per checkpoint chunk

PI_LD = np.longdouble("3.14159265358979323846264338327950288")
TWO_PI_LD = 2 * PI_LD
LN2PI_LD = np.log(TWO_PI_LD)
T0 = 8436146000.0           # exact integer base (< 2^53) for phase splitting
N = 36642                   # RS main-sum length, constant over the ensemble

nn = np.arange(1, N + 1)
v = np.log(nn)                                   # float64 (d*v error ~2e-10)
w = 1.0 / np.sqrt(nn)
v_ld = np.log(nn.astype(np.longdouble))
u = np.asarray((np.longdouble(T0) * v_ld) % TWO_PI_LD, dtype=float)

zl = np.load(DATA + "/lmfdb_zeros_parsed.npy")
gaps_all = np.diff(zl)
rng = np.random.default_rng(20260808)
off = int(rng.integers(0, 4))
gidx_all = np.arange(off, len(gaps_all), 4)
# split into two halves
gidx = gidx_all[H::2]
print("half %d: offset %d, %d gaps" % (H, off, len(gidx)), flush=True)


def theta_mod(t):
    tld = np.longdouble(t)
    th = ((tld / 2) * (np.log(tld) - LN2PI_LD) - tld / 2 - PI_LD / 8
          + 1 / (48 * tld) + 7 / (5760 * tld ** 3) + 31 / (80640 * tld ** 5))
    return float(th % TWO_PI_LD)


def Z_batch(ts):
    ts = np.asarray(ts, float)
    d = ts - T0
    A = np.array([theta_mod(t) for t in ts])
    ph = A[:, None] - u[None, :] - np.outer(d, v)
    main = 2.0 * np.sum(np.cos(ph) * w[None, :], axis=1)
    tau = np.sqrt(ts / (2 * np.pi))
    p = tau - N
    psi = np.cos(2 * np.pi * (p * p - p - 1.0 / 16.0)) / np.cos(2 * np.pi * p)
    return main + ((-1.0) ** (N + 1)) * tau ** -0.5 * psi


out = np.empty((len(gidx), 4))
ck = DATA + "/moment_floors_half%d_ck.npy" % H
ckd = DATA + "/moment_floors_half%d_ckdat.npy" % H
start_chunk = 0
if os.path.exists(ck) and os.path.exists(ckd):
    start_chunk = int(np.load(ck)[0])
    prev = np.load(ckd)
    out[:len(prev)] = prev
    print("resuming at chunk %d (%d rows)" % (start_chunk, len(prev)),
          flush=True)

t_start = time.time()
for c0 in range(start_chunk * CHUNK, len(gidx), CHUNK):
    gi = gidx[c0:c0 + CHUNK]
    za, zb = zl[gi], zl[gi + 1]
    # K interior points per gap
    fr = (np.arange(K) + 1.0) / (K + 1.0)
    ts = za[:, None] + (zb - za)[:, None] * fr[None, :]
    tsf = ts.ravel()
    Zv = np.empty(len(tsf))
    for b0 in range(0, len(tsf), BATCH):
        Zv[b0:b0 + BATCH] = Z_batch(tsf[b0:b0 + BATCH])
    fl = np.max(np.abs(Zv).reshape(len(gi), K), axis=1)
    out[c0:c0 + len(gi), 0] = gi
    out[c0:c0 + len(gi), 1] = 0.5 * (za + zb)
    out[c0:c0 + len(gi), 2] = zb - za
    out[c0:c0 + len(gi), 3] = fl
    cnum = c0 // CHUNK
    np.save(ckd, out[:c0 + len(gi)])
    np.save(ck, np.array([cnum + 1]))
    if cnum % 4 == 0:
        el = time.time() - t_start
        done = c0 + len(gi) - start_chunk * CHUNK
        tot = len(gidx) - start_chunk * CHUNK
        print("half %d chunk %d/%d  %.0f%%  elapsed %.0fs eta %.0fs"
              % (H, cnum, (len(gidx) + CHUNK - 1) // CHUNK,
                 100.0 * done / tot, el, el * (tot - done) / max(done, 1)),
              flush=True)

np.save(DATA + "/moment_floors_half%d.npy" % H, out)
print("half %d done in %.0fs" % (H, time.time() - t_start), flush=True)

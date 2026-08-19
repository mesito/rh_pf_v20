#!/usr/bin/env python3
"""
Part A compute: full-ensemble floors + L_X for all tight pairs (8.4e9 & 2M).

Conventions (preprint_sec02/the floor convention of Section 52, cross-checked against verify_all.py):
  tight pair: s = g * log(t0/2pi)/(2pi) < 0.15
  floor = max_{u in [-g/2, g/2]} |Z_rs2(t0+u)|, 33-point grid + parabolic refine
  L_X(t) = sum_{k=1..3} (1/k) sum_{p<=X} p^{-k/2} cos(k t ln p),
           phases (t*ln p) mod 2pi in 80-bit longdouble, one cumsum pass
           over primes <= 1e7 yields all six cutoffs.

Resumable: each stage checkpoints every 256 pairs; rerun until "ALL DONE".
"""
import os
import time
import gzip
import numpy as np

D = os.environ.get("RH_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")) + "/"
TWO_PI_LD = np.longdouble(2 * np.pi)

# ------------------------------------------------------------------ data
zl = np.load(D + 'lmfdb_zeros_parsed.npy')
zll = zl.astype(np.longdouble)
z6 = np.loadtxt(gzip.open(D + 'zeros6.gz', 'rt'))


def sieve(n):
    b = np.ones(n + 1, bool)
    b[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if b[i]:
            b[i * i::i] = False
    return np.flatnonzero(b)


# ------------------------------------------------- tight-pair extraction
def tight_pairs(z):
    g = np.diff(z)
    t0 = 0.5 * (z[:-1] + z[1:])
    s = g * np.log(t0 / (2 * np.pi)) / (2 * np.pi)
    return np.flatnonzero(s < 0.15)          # 0-based index of first zero


pairs8 = tight_pairs(zl)
r8 = np.load(D + 'lehmer_rows_8p4e9.npy')
assert len(pairs8) == 2736 == len(r8), (len(pairs8), len(r8))
assert np.array_equal(pairs8 + 1, r8[:, 0].astype(int))
pairs2 = tight_pairs(z6)
r2 = np.load(D + 'lehmer_rows_2M.npy')
assert len(pairs2) == 6088 == len(r2), (len(pairs2), len(r2))
assert np.array_equal(pairs2 + 1, r2[:, 0].astype(int))
print('pairs verified: 8.4e9 %d, 2M %d' % (len(pairs8), len(pairs2)), flush=True)

# ------------------------------------------------- Z_rs2 vectorized floors
LNN = np.log(np.arange(1, 37000).astype(np.longdouble))
SQN = np.sqrt(np.arange(1., 37000.))


def Z_rs2_vec(ts):
    out = np.empty(len(ts))
    for j, t in enumerate(ts):
        tld = np.longdouble(t)
        tau = np.sqrt(tld / TWO_PI_LD)
        N = int(np.floor(tau))
        p = float(tau - N)
        lnn = LNN[:N]
        thcorr = 1 / (48 * tld) + 7 / (5760 * tld ** 3) + 31 / (80640 * tld ** 5)
        ph = (tld * (np.log(tau) - lnn) - tld / 2 - np.longdouble(np.pi / 8)
              + thcorr) % TWO_PI_LD
        main = 2.0 * float(np.sum(np.cos(np.asarray(ph, float)) / SQN[:N]))
        psi = np.cos(2 * np.pi * (p * p - p - 1.0 / 16.0)) / np.cos(2 * np.pi * p)
        out[j] = main + (-1) ** (N + 1) * float(tau) ** -0.5 * psi
    return out


def floor_max(i0):
    ga, gb = zll[i0], zll[i0 + 1]
    g = float(gb - ga)
    t0 = float((ga + gb) / 2)
    us = np.linspace(-g / 2, g / 2, 33)
    zv = np.abs(Z_rs2_vec(t0 + us))
    j = int(np.argmax(zv))
    f = zv[j]
    if 0 < j < len(us) - 1:
        y0, y1, y2 = zv[j - 1], zv[j], zv[j + 1]
        h = us[1] - us[0]
        den = y0 - 2 * y1 + y2
        if den < 0:
            du = 0.5 * h * (y0 - y2) / den
            f = y1 - 0.25 * (y0 - y2) * du / h
    return f


# ------------------------------------------------- L_X scan (longdouble)
PR7 = sieve(10 ** 7)
LNPR7 = np.log(PR7.astype(np.longdouble))
P7 = PR7.astype(float)
SQ7 = np.sqrt(P7)
XCUTS = [10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7]
KIDX = [np.searchsorted(PR7, X, side='right') - 1 for X in XCUTS]


def LX_scan(t_ld, out, start):
    """out: (n, 6) array, filled from row `start`; returns rows completed."""
    n = len(t_ld)
    CH = 16
    done = start
    for c0 in range(start, n, CH):
        ts = t_ld[c0:c0 + CH]
        c1 = np.empty((len(PR7), len(ts)))
        for j, tt in enumerate(ts):
            c1[:, j] = np.cos(np.asarray((tt * LNPR7) % TWO_PI_LD, dtype=float))
        c2 = 2.0 * c1 ** 2 - 1.0
        c3 = 4.0 * c1 ** 3 - 3.0 * c1
        S = (np.cumsum(c1 / SQ7[:, None], axis=0)
             + 0.5 * np.cumsum(c2 / P7[:, None], axis=0)
             + (1.0 / 3.0) * np.cumsum(c3 / (P7 ** 1.5)[:, None], axis=0))
        for q, kk in enumerate(KIDX):
            out[c0:c0 + len(ts), q] = S[kk]
        del c1, c2, c3, S
        done = c0 + len(ts)
        if done % 256 < CH or done == n:
            return done          # checkpoint boundary
    return done


# ================================================== STAGE F8: floors 8.4e9
PF = D + 'floors_all_8p4e9.partial.npy'
FIN = D + 'floors_all_8p4e9.npy'
if not os.path.exists(FIN):
    if os.path.exists(PF):
        tmp = np.load(PF, allow_pickle=True).item()
        arr8, nd8 = tmp['arr'], tmp['nd']
    else:
        arr8 = np.zeros((len(pairs8), 6))
        arr8[:, 0] = pairs8 + 1
        arr8[:, 1] = np.asarray((zll[pairs8] + zll[pairs8 + 1]) / 2, float)
        arr8[:, 2] = np.asarray(zll[pairs8 + 1] - zll[pairs8], float)
        arr8[:, 3] = arr8[:, 2] * np.log(arr8[:, 1] / (2 * np.pi)) / (2 * np.pi)
        arr8[:, 4] = r8[:, 5]
        nd8 = 0
    t0 = time.time()
    while nd8 < len(pairs8):
        arr8[nd8, 5] = floor_max(int(arr8[nd8, 0]) - 1)
        nd8 += 1
        if nd8 % 256 == 0:
            np.save(PF, {'arr': arr8, 'nd': nd8}, allow_pickle=True)
            print('F8 %d/%d (%.0fs)' % (nd8, len(pairs8), time.time() - t0),
                  flush=True)
            if time.time() - t0 > 380:
                print('F8 checkpoint exit', flush=True)
                raise SystemExit(0)
    np.save(FIN, arr8)
    if os.path.exists(PF):
        os.remove(PF)
    print('F8 done (%.0fs)' % (time.time() - t0), flush=True)

# ================================================== STAGE L8: L_X 8.4e9 + controls
PL = D + 'LX_all_8p4e9.partial.npy'
FINL = D + 'LX_all_8p4e9.npy'
if not os.path.exists(FINL):
    arr8 = np.load(FIN)
    t_pairs = (zll[pairs8] + zll[pairs8 + 1]) / 2
    # same 300 controls as verify_all (seed 11, s in [0.95,1.05])
    dd9 = np.log(zl.mean() / (2 * np.pi)) / (2 * np.pi)
    s_all = np.diff(zl) * dd9
    elig = np.flatnonzero((s_all >= 0.95) & (s_all <= 1.05))
    rng = np.random.default_rng(11)
    sel = rng.choice(elig, 300, replace=False)
    t_ctrl = (zll[sel] + zll[sel + 1]) / 2
    t_all = np.concatenate([t_pairs, t_ctrl])
    if os.path.exists(PL):
        tmp = np.load(PL, allow_pickle=True).item()
        LX, ndl = tmp['arr'], tmp['nd']
    else:
        LX = np.zeros((len(t_all), 6))
        ndl = 0
    t0 = time.time()
    while ndl < len(t_all):
        ndl = LX_scan(t_all, LX, ndl)
        np.save(PL, {'arr': LX, 'nd': ndl}, allow_pickle=True)
        print('L8 %d/%d (%.0fs)' % (ndl, len(t_all), time.time() - t0), flush=True)
        if time.time() - t0 > 380:
            print('L8 checkpoint exit', flush=True)
            raise SystemExit(0)
    np.save(FINL, {'LX_pairs': LX[:len(pairs8)], 'LX_ctrl': LX[len(pairs8):],
                   'ctrl_idx': sel}, allow_pickle=True)
    if os.path.exists(PL):
        os.remove(PL)
    print('L8 done (%.0fs)' % (time.time() - t0), flush=True)

# ================================================== STAGE F2: floors 2M (fp64)
FIN2 = D + 'floors_all_2M.npy'
if not os.path.exists(FIN2):
    t0 = time.time()
    ga, gb = z6[pairs2], z6[pairs2 + 1]
    g2 = gb - ga
    t02 = 0.5 * (ga + gb)
    s2 = g2 * np.log(t02 / (2 * np.pi)) / (2 * np.pi)
    fmax = np.empty(len(pairs2))
    fmin = np.empty(len(pairs2))

    def Z_fp64(t):
        t = np.atleast_1d(np.asarray(t, float))
        tau = np.sqrt(t / (2 * np.pi))
        N = np.floor(tau).astype(int)
        p = tau - N
        th = (t / 2 * np.log(t / (2 * np.pi)) - t / 2 - np.pi / 8
              + 1 / (48 * t) + 7 / (5760 * t ** 3))
        out = np.empty(len(t))
        for j in range(len(t)):
            nn = np.arange(1, N[j] + 1)
            out[j] = 2.0 * np.sum(np.cos(th[j] - t[j] * np.log(nn)) / np.sqrt(nn))
        psi = np.cos(2 * np.pi * (p * p - p - 1.0 / 16.0)) / np.cos(2 * np.pi * p)
        return out + ((-1) ** (N + 1)) * tau ** -0.5 * psi

    for j in range(len(pairs2)):
        us = np.linspace(ga[j], gb[j], 124)[1:-1]      # 122 interior samples
        zv = np.abs(Z_fp64(us))
        fmin[j] = zv.min()
        k = int(np.argmax(zv))
        f = zv[k]
        if 0 < k < len(zv) - 1:
            y0, y1, y2 = zv[k - 1], zv[k], zv[k + 1]
            h = us[1] - us[0]
            den = y0 - 2 * y1 + y2
            if den < 0:
                du = 0.5 * h * (y0 - y2) / den
                f = y1 - 0.25 * (y0 - y2) * du / h
        fmax[j] = f
        if (j + 1) % 1000 == 0:
            print('F2 %d/%d (%.0fs)' % (j + 1, len(pairs2), time.time() - t0),
                  flush=True)
    arr2 = np.column_stack([pairs2 + 1, t02, g2, s2, r2[:, 5], fmax, fmin])
    np.save(FIN2, arr2)
    print('F2 done (%.0fs)' % (time.time() - t0), flush=True)

# ================================================== STAGE L2: L_X(1e4) 2M fp64
FINL2 = D + 'LX_all_2M.npy'
if not os.path.exists(FINL2):
    t0 = time.time()
    arr2 = np.load(FIN2)
    t02 = arr2[:, 1]
    PR4 = sieve(10 ** 4).astype(float)
    LNP4 = np.log(PR4)
    LX2 = np.empty(len(t02))
    for j in range(len(t02)):
        c1 = np.cos((t02[j] * LNP4) % (2 * np.pi))
        c2 = 2.0 * c1 ** 2 - 1.0
        c3 = 4.0 * c1 ** 3 - 3.0 * c1
        LX2[j] = (np.sum(c1 / np.sqrt(PR4)) + 0.5 * np.sum(c2 / PR4)
                  + (1.0 / 3.0) * np.sum(c3 / PR4 ** 1.5))
    np.save(FINL2, LX2)
    print('L2 done (%.0fs)' % (time.time() - t0), flush=True)

print('ALL DONE', flush=True)

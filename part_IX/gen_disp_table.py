#!/usr/bin/env python3
"""Regenerate the corrected 15-row displacement table (T8 --slow values,
longdouble t0 reference) and save as CSV."""
import os
import numpy as np
import mpmath as mp
import time

d = '" + os.environ.get("RH_DATA","../data") + "/'
zl = np.load(d + 'lmfdb_zeros_parsed.npy')
zll = zl.astype(np.longdouble)
floors = np.load(d + 'lehmer_floors_8p4e9.npy')

recs = []
t0 = time.time()
for rank, row in enumerate(floors[:15], 1):
    i = int(row[0]) - 1
    t0_ld = (zll[i] + zll[i + 1]) / 2
    g_ld = zll[i + 1] - zll[i]
    win = np.concatenate([np.arange(i - 250, i), np.arange(i + 2, i + 252)])
    lam = np.sum(1.0 / (t0_ld - zll[win]))          # lambda_far, window +-250 excl. pair
    pred = (g_ld ** 2 / 8) * lam                    # displacement prediction
    t0_mp = (mp.mpf(float(zl[i])) + mp.mpf(float(zl[i + 1]))) / 2  # exact reference t0
    mp.mp.dps = 30
    tstar = mp.findroot(lambda x: mp.diff(mp.siegelz, x), t0_mp)
    delta = float(tstar - t0_mp)
    rel = abs(delta - float(pred)) / abs(float(pred)) * 100.0
    recs.append((rank, float(g_ld), float(lam), delta, float(pred), rel))
    print('%2d  g=%.6e  lam=%+.6f  delta=%+.9e  pred=%+.9e  rel=%.3f%%'
          % (rank, float(g_ld), float(lam), delta, float(pred), rel), flush=True)

corr = np.corrcoef([r[3] for r in recs], [r[4] for r in recs])[0, 1]
print('corr(delta, pred) = %.6f   elapsed %.0fs' % (corr, time.time() - t0))

hdr = 'pair_rank,g,lambda_far,delta_corrected,prediction,abs_rel_err_pct'
out = '\n'.join('%d,%.10e,%.10f,%.10e,%.10e,%.4f' % r for r in recs)
with open(d + 'disp_table_8p4e9_corrected.csv', 'w') as fh:
    fh.write(hdr + '\n' + out + '\n')
print('saved', d + 'disp_table_8p4e9_corrected.csv')

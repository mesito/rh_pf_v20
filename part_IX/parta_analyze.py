#!/usr/bin/env python3
"""Part A analysis: full-ensemble fits, statistics, CSV + report md."""
import os
import numpy as np

D = os.environ.get("RH_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")) + "/"
XCUTS = [10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7]

f8 = np.load(D + 'floors_all_8p4e9.npy')          # idx,t0,g,s,far,floor
L8 = np.load(D + 'LX_all_8p4e9.npy', allow_pickle=True).item()
LXp, LXc = L8['LX_pairs'], L8['LX_ctrl']
f2 = np.load(D + 'floors_all_2M.npy')             # idx,t0,g,s,far,fmax,fmin
LX2 = np.load(D + 'LX_all_2M.npy')


def ols(Xm, y):
    A = np.column_stack([Xm, np.ones(len(y))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef
    r2 = 1 - float(np.sum(res ** 2) / np.sum((y - y.mean()) ** 2))
    sig2 = float(np.sum(res ** 2) / (len(y) - A.shape[1]))
    se = np.sqrt(np.diag(sig2 * np.linalg.inv(A.T @ A)))
    return coef, r2, res, se


def resid_on(x, C):
    A = np.column_stack([C, np.ones(len(C))])
    cf, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ cf


# ---------------- fits: 8.4e9 full (2736) per X ----------------
lg8 = np.log(f8[:, 2])
lf8 = np.log(f8[:, 5])
far8 = f8[:, 4]
rows_csv = []
fit8 = {}
for q, X in enumerate(XCUTS):
    (a, b, c), r2v, res, se = ols(np.column_stack([lg8, LXp[:, q]]), lf8)
    fit8[X] = (a, b, c, r2v, res, se)
    rows_csv.append(('8.4e9', 2736, 'max', X, a, se[0], b, se[1], c, r2v))

# ---------------- fits: 2M full (6088) at X=1e4, both conventions
lg2 = np.log(f2[:, 2])
far2 = f2[:, 4]
res2 = {}
for conv, col in (('max', 5), ('min122', 6)):
    lf2 = np.log(f2[:, col])
    (a, b, c), r2v, res, se = ols(np.column_stack([lg2, LX2]), lf2)
    res2[conv] = (a, b, c, r2v, res, se, lf2)
    rows_csv.append(('2M', 6088, conv, 10 ** 4, a, se[0], b, se[1], c, r2v))

with open(D + 'fits_all_pairs.csv', 'w') as fh:
    fh.write('ensemble,n_pairs,floor_convention,X,a,se_a,b,se_b,c,R2\n')
    for r in rows_csv:
        fh.write('%s,%d,%s,%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n' % r)

# ---------------- full-set statistics at 8.4e9 ----------------
q4 = XCUTS.index(10 ** 4)
L4 = LXp[:, q4]
(_, _), r2g, resg, _ = ols(lg8[:, None], lf8)                    # g only
coef_, r2gf, _, _ = ols(np.column_stack([lg8, np.log(far8)]), lf8)
coef_, r2gL, _, _ = ols(np.column_stack([lg8, L4]), lf8)
coef_, r2gfL, _, _ = ols(np.column_stack([lg8, np.log(far8), L4]), lf8)
pc_far = float(np.corrcoef(resid_on(resg, L4), resid_on(far8, L4))[0, 1])
pc_L = float(np.corrcoef(resid_on(resg, far8), resid_on(L4, far8))[0, 1])
k5 = float(np.corrcoef(resg, np.log(far8))[0, 1])
cL = float(np.corrcoef(L4, resg)[0, 1])
clf = float(np.corrcoef(L4, np.log(far8))[0, 1])

shifts = {}
for q, X in enumerate(XCUTS):
    dm = float(LXp[:, q].mean() - LXc[:, q].mean())
    shifts[X] = (dm / float(LXc[:, q].std()), float(np.exp(-dm)))

# 2M stats (max convention)
a2, b2, c2, r22, res2f, se2, lf2m = res2['max']
(_, _), _, resg2, _ = ols(lg2[:, None], lf2m)
cL2 = float(np.corrcoef(LX2, resg2)[0, 1])
clf2 = float(np.corrcoef(LX2, np.log(far2))[0, 1])
k52 = float(np.corrcoef(resg2, np.log(far2))[0, 1])

# ---------------- report ----------------
B300 = {10 ** 2: (1.847, 0.752, 0.9339), 10 ** 3: (1.978, 0.733, 0.9348),
        10 ** 4: (1.978, 0.783, 0.9447), 10 ** 5: (1.969, 0.829, 0.9505),
        10 ** 6: (1.932, 0.899, 0.9439), 10 ** 7: (1.868, 0.944, 0.9546)}
L = []
L.append('# Part A — full-ensemble floor fits (all tight pairs)\n')
L.append('Data: `floors_all_8p4e9.npy` (2736 pairs, floor = max|Z_rs2| over the gap, '
         'paper convention of the floor convention of Section 52 eq. floor-def, 33-pt grid + parabolic refine), '
         '`LX_all_8p4e9.npy` (L_X at 6 cutoffs, one cumsum pass over 664,579 primes), '
         '`floors_all_2M.npy` (6088 pairs, fp64 RS, max and min122 conventions), '
         '`LX_all_2M.npy`.\n')
L.append('Convention check: on the 300-pair subset, max-convention floors are ~29.4x the '
         'archived min-122-sample floors (constant factor, absorbed in c); refit at X=1e4 '
         'gives a=2.006/1.978, b=0.772/0.783, R2=0.9442/0.9447 — no material difference '
         'in a, b, R2 (all scale-invariant).\n')
L.append('## Table 52.1 (full): 8.4e9, 2736 pairs vs 300-pair values\n')
L.append('| X | a (full) | se(a) | b (full) | se(b) | R2 (full) | a/b/R2 (300) | move? |')
L.append('|---|---|---|---|---|---|---|---|')
for X in XCUTS:
    a, b, c, r2v, res, se = fit8[X]
    t = B300[X]
    flag = '**YES**' if (abs(a - t[0]) > 0.05 or abs(b - t[1]) > 0.05
                         or abs(r2v - t[2]) > 0.01) else 'no'
    L.append('| 1e%d | %.4f | %.4f | %.4f | %.4f | %.4f | %.3f/%.3f/%.4f | %s |'
             % (int(np.log10(X)), a, se[0], b, se[1], r2v, t[0], t[1], t[2], flag))
L.append('\n## Table 52.2(a) (full): 2M, 6088 pairs, X=1e4\n')
L.append('| convention | a | se(a) | b | se(b) | R2 | 300-pair a/b/R2 |')
L.append('|---|---|---|---|---|---|---|')
for conv in ('max', 'min122'):
    a, b, c, r2v, res, se, _ = res2[conv]
    L.append('| %s | %.4f | %.4f | %.4f | %.4f | %.4f | 1.892/0.906/0.9463 |'
             % (conv, a, se[0], b, se[1], r2v))
L.append('\n## Full-set statistics (8.4e9, X=1e4 unless stated)\n')
L.append('| statistic | full (2736) | 300-pair value |')
L.append('|---|---|---|')
L.append('| R2(g only) | %.4f | 0.4556 |' % r2g)
L.append('| R2(g+far) | %.4f | 0.7615 |' % r2gf)
L.append('| R2(g+L_X) | %.4f | 0.9447 |' % r2gL)
L.append('| R2(g+far+L_X) | %.4f | 0.9447 |' % r2gfL)
L.append('| corr(resid, L_X) | %.4f | 0.946 |' % cL)
L.append('| partial corr(resid, far | L_X) | %.4f | 0.004 |' % pc_far)
L.append('| partial corr(resid, L_X | far) | %.4f | 0.873 |' % pc_L)
L.append('| K5 corr(resid, log far) | %.4f | -0.746 |' % k5)
L.append('| corr(L_X, log far) | %.4f | -0.641..-0.791 |' % clf)
L.append('\nEuler deficit at Lehmer midpoints vs 300 controls (seed 11, s in [0.95,1.05]):\n')
L.append('| X | shift (sigma) full | 300-pair | ratio P_X smaller full | 300-pair |')
L.append('|---|---|---|---|---|')
s300 = {10 ** 2: -0.54, 10 ** 3: -0.96, 10 ** 4: -1.54, 10 ** 5: -2.28,
        10 ** 6: -3.13, 10 ** 7: -3.90}
for X in XCUTS:
    sh, rt = shifts[X]
    L.append('| 1e%d | %+.2f | %+.2f | %.1fx | %s |'
             % (int(np.log10(X)), sh, s300[X], rt,
                ('7.4x' if X == 10 ** 6 else ('12x' if X == 10 ** 7 else '-'))))
L.append('\n## Full-set statistics (2M, 6088 pairs, X=1e4, max convention)\n')
L.append('| statistic | full | 300-pair |')
L.append('|---|---|---|')
L.append('| corr(L_X, resid) | %.4f | 0.916 |' % cL2)
L.append('| corr(L_X, log far) | %.4f | -0.422 |' % clf2)
L.append('| K5 corr(resid, log far) | %.4f | -0.50 |' % k52)

with open(D + 'parta_full_pairs_report.md', 'w') as fh:
    fh.write('\n'.join(L) + '\n')
print('\n'.join(L))

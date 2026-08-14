"""
verify_28_channel_witness.py -- Theorem 28.2 and Lemma 28.3 (explicit interference witness).

Checks the closed forms c2 = kappa log2, c3 = -kappa log3, c6 = (1+kappa^2) log6,
the coefficient envelope max|c_n| over n<=3000, and the functional equation /
reality of the completed DH function.
"""
import mpmath as mp
import numpy as np
from ic_core import (KAPPA, dh_coeffs, dh_coeffs_fast, Xi_DH, H, RHO1, RHO2, banner)

mp.mp.dps = 25

def run():
    banner("Theorem 28.2 / Lemma 28.3: explicit interference witness")
    c = dh_coeffs(60)
    print(f"  kappa = {mp.nstr(KAPPA,8)}  (paper 0.284079)")
    print(f"  c2 = {mp.nstr(c[2],8)}   kappa*log2  = {mp.nstr(KAPPA*mp.log(2),8)}")
    print(f"  c3 = {mp.nstr(c[3],8)}   -kappa*log3 = {mp.nstr(-KAPPA*mp.log(3),8)}")
    print(f"  c6 = {mp.nstr(c[6],8)}   (1+k^2)log6 = {mp.nstr((1+KAPPA**2)*mp.log(6),8)}  (paper 1.9364)")
    cf = dh_coeffs_fast(3000)
    am = int(np.argmax(np.abs(cf)))
    print(f"  max|c_n| (n<=3000) = {float(np.abs(cf).max()):.4f} at n = {am}  (paper 30.8 at 2856)")
    print(f"  von Mangoldt ceiling log 3000 = {float(mp.log(3000)):.4f}")

    banner("Section 28 setup: functional equation, reality, and the two off-line zeros")
    s = mp.mpf('0.37') + mp.mpf('12.3')*mp.mpc(0,1)
    print(f"  |Xi_DH(s) - Xi_DH(1-s)| = {mp.nstr(abs(Xi_DH(s)-Xi_DH(1-s)),3)}  (paper 3e-16; mpmath Hurwitz is sharper)")
    half = mp.mpf('0.5') + mp.mpf('20.1')*mp.mpc(0,1)
    print(f"  |Im Xi_DH(1/2+it)|/|Xi| = {mp.nstr(abs(mp.im(Xi_DH(half)))/abs(Xi_DH(half)),3)}  (paper 2e-16)")
    print(f"  |H(rho1)| = {mp.nstr(abs(H(RHO1)),3)}, |H(rho2)| = {mp.nstr(abs(H(RHO2)),3)}  (limited by 12-digit input)")

if __name__ == "__main__":
    run()

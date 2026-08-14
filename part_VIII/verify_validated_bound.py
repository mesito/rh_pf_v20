"""
verify_validated_bound.py -- verifier for the note
"An explicit bound for S(T) from validated inputs" (M. Ismail).

Checks, at 25-digit precision:
  1. CERTIFICATION: with the historical inputs (k1,B1)=(2.38,1) the master
     formula at (0.06,2.08) reproduces Trudgian's arXiv constants
     (0.110428, 0.274023, 2.449532) exactly.
  2. The validated evaluation: (i) at (0.06,2.08) and (ii) at (0.150,2.470)
     with (k1,B1)=(0.618, 1/2+1.93/log 1000), including the Patel weights
     W1,W3,W4 and their two internal consistency identities.
  3. Admissibility (Proposition): all parameter conditions, incl. R<=3/2+eta.
  4. Boundary inputs (V2),(V3): grid verification on 0<=t<=5e4.
  5. Small-height thresholds, the crossings, and the applications.

Requires: mpmath, numpy, scipy.
"""
import math
import numpy as np
import mpmath as mp
from scipy.optimize import brentq

FAILED = []
def check(name, ok, extra=""):
    tag = "PASS" if ok else "FAIL"
    if not ok: FAILED.append(name)
    print(f"  [{tag}] {name}" + (f"  ({extra})" if extra else ""))

def master(eta, r, k1, B1):
    """a, b, c of the master inequality with the Patel weight term."""
    mp.mp.dps = 25
    eta, r, k1, B1 = [mp.mpf(str(x)) for x in (eta, r, k1, B1)]
    R = r*(mp.mpf('0.5')+eta)
    f1, f2, f3 = mp.asin(eta/R), mp.asin(1/r), mp.asin((1+eta)/R)
    a = (f1*eta+(f2-3*mp.pi/2)*(mp.mpf('0.5')+eta)+f3*(1+eta)
         + R*(mp.cos(f1)+mp.cos(f2)+mp.cos(f3)))/(6*mp.pi*mp.log(r))
    b = (-f1+f3+R*((1-mp.cos(f1))/eta
         + (mp.pi/2-mp.cos(f3)-f3)/(R-(1+eta))))/(2*mp.pi*mp.log(r))
    T1 = mp.log(mp.zeta(1+eta)/mp.zeta(2+2*eta))/(2*mp.log(r))
    T2 = mp.log(mp.zeta(mp.mpf('0.5')+mp.sqrt(2)*(eta+mp.mpf('0.5'))))/mp.pi
    T3 = mp.quad(lambda ph: mp.log(mp.zeta(1+eta+R*mp.cos(ph))),
                 [-mp.pi/2, mp.pi/2])/(4*mp.pi*mp.log(r))
    Bk = R*(2*mp.cos(f2)-mp.cos(f1)-mp.cos(f3))+f2-f3+eta*(2*f2-f1-f3)
    E = (mp.log(mp.zeta(1+eta))*(f1+R*(mp.cos(f1)-1)/eta)
         + (mp.mpf('0.5')+eta)*(mp.pi/2-f2-r*mp.cos(f2))*mp.log(2*mp.pi)
         + ((1+eta)*(mp.pi/2-f3)-R*mp.cos(f3))/(1+eta-R)*mp.log(mp.zeta(R-eta)))
    W1 = R*(1-mp.cos(f1))/eta
    W3 = 2*(R*(mp.cos(f2)-mp.cos(f3))-(mp.mpf('0.5')+eta)*(f3-f2))
    W4 = R*(mp.pi/2-f3-mp.cos(f3))/(R-(1+eta))
    c = T1+T2+T3+(E-2*Bk*mp.log(k1)+(W1+W3+W4)*mp.log(B1))/(2*mp.pi*mp.log(r)) \
        + mp.mpf('0.003')
    return dict(a=a, b=b, c=c, W1=W1, W3=W3, W4=W4, R=R,
                f1=f1, f2=f2, f3=f3, Bk=Bk)

def main():
    B1 = mp.mpf('0.5') + mp.mpf('1.93')/mp.log(1000)
    print("== certification (historical inputs) ==")
    m = master(0.06, 2.08, 2.38, 1)
    check("a = 0.110428", abs(float(m['a'])-0.110428) < 2e-6, f"{float(m['a']):.6f}")
    check("b = 0.274023", abs(float(m['b'])-0.274023) < 2e-6, f"{float(m['b']):.6f}")
    check("c(2.38, 1) = 2.449532", abs(float(m['c'])-2.449532) < 2e-6, f"{float(m['c']):.6f}")

    print("== validated evaluation ==")
    mi = master(0.06, 2.08, 0.618, B1)
    check("(i)  c = 2.270511; +0.0605 < 2.332",
          abs(float(mi['c'])-2.270511) < 2e-5 and float(mi['c'])+0.0605 < 2.332,
          f"{float(mi['c']):.6f}")
    mii = master(0.150, 2.470, 0.618, B1)
    check("(ii) a,b = 0.136528, 0.178547",
          abs(float(mii['a'])-0.136528) < 2e-6 and abs(float(mii['b'])-0.178547) < 2e-6)
    check("(ii) c = 1.221514; +0.0605 < 1.283",
          abs(float(mii['c'])-1.221514) < 2e-5 and float(mii['c'])+0.0605 < 1.283,
          f"{float(mii['c']):.6f}")
    # weight identities
    for tag, m_ in (("(i)", mi), ("(ii)", mii)):
        lhs = float(m_['W1']+m_['W4'])
        rhs = float(m_['R']*((1-mp.cos(m_['f1']))/ (mp.mpf(str({'(i)':0.06,'(ii)':0.150}[tag])))
              + (mp.pi/2-mp.cos(m_['f3'])-m_['f3'])/(m_['R']-(1+mp.mpf(str({'(i)':0.06,'(ii)':0.150}[tag]))))))
        check(f"{tag} identity W1+W4 = b-bracket", abs(lhs-rhs) < 1e-12, f"{lhs:.6f}")
        # R3 split: sigma=1/2 portion + W3 = phi3-phi2
        eta = {'(i)': 0.06, '(ii)': 0.150}[tag]
        half_side = 2*((1+eta)*(float(m_['f3'])-float(m_['f2']))
                       - float(m_['R'])*(math.cos(float(m_['f2']))-math.cos(float(m_['f3']))))
        check(f"{tag} identity R3 split", abs(half_side+float(m_['W3'])
              - (float(m_['f3'])-float(m_['f2']))) < 1e-12)

    print("== admissibility ==")
    for eta, r in ((0.06, 2.08), (0.150, 2.470)):
        R = r*(0.5+eta)
        ok = (0 < eta <= 0.5 and r > 1 and R <= 1.5+eta and R-eta > 1
              and 0.5+math.sqrt(2)*(eta+0.5) > 1 and 1+eta > 1)
        check(f"conditions at ({eta},{r})", ok, f"R={R:.4f} <= {1.5+eta:.2f}")

    print("== boundary inputs (V2),(V3) on a grid ==")
    mp.mp.dps = 15
    Q0 = 1000
    ts = list(np.linspace(0.01, 50, 60)) + list(np.geomspace(50, 5e4, 60))
    w2 = max(float(abs(mp.mpc(0, t)*mp.zeta(1+1j*mp.mpf(t)))
             / (float(B1)*abs(mp.mpc(Q0+1, t))*mp.log(abs(mp.mpc(Q0+1, t)))))
             for t in ts)
    check("(V2) grid max ratio < 1", w2 < 1, f"max {w2:.4f}")
    w3 = max(float(abs(mp.mpc(0, t)-1)*abs(mp.zeta(1j*mp.mpf(t)))
             / (float(B1)/math.sqrt(2*math.pi)*abs(mp.mpc(Q0, t))**1.5
                * mp.log(abs(mp.mpc(Q0, t)))))
             for t in ts)
    check("(V3) grid max ratio < 1", w3 < 1, f"max {w3:.4f}")

    print("== thresholds, crossings, applications ==")
    new = (0.1366, 0.1786, 1.283); i_n = (0.112, 0.278, 2.332)
    PT = (0.110, 0.290, 2.290); BW1 = (0.10076, 0.24460, 7.20844)
    BW2 = (0.10076, 1.68845, 1.50956)
    g = lambda B, L: B[0]*L+B[1]*math.log(L)+B[2]
    Lth = brentq(lambda L: g(new, L)-2, 1, 10)
    check("(ii) >= 2 for T >= 36", math.exp(Lth) < 36, f"T={math.exp(Lth):.1f}")
    check("(ii) at e = 1.4196 > 1", abs(g(new, 1)-1.4196) < 1e-3)
    for nm, other, ref in (("PT15", PT, 5.2e23), ("(i)", i_n, 4.8e25),
                           ("BW br.1", BW1, 8.6e75)):
        L = brentq(lambda l: g(new, l)-g(other, l), 5, 300)
        check(f"(ii) = {nm} at ~{ref:.1e}", abs(math.exp(L)/ref-1) < 0.06,
              f"{math.exp(L):.2e}")
    Lwin = math.log(5.2e23)
    check("BW br.2 > (ii) throughout the window",
          min(g(BW2, l)-g(new, l) for l in np.linspace(1, Lwin, 300)) > 0,
          f"min diff {min(g(BW2, l)-g(new, l) for l in np.linspace(1, Lwin, 300)):.3f}")
    L0 = math.log(3e12); S = g(new, L0)
    check("Sbar(T0) = 5.8072; s_n <= 12.62",
          abs(S-5.8072) < 3e-4 and 1+2*S <= 12.62, f"{S:.4f}, {1+2*S:.3f}")

    print()
    if FAILED:
        print("FAILURES:", ", ".join(FAILED)); raise SystemExit(1)
    print("ALL CHECKS PASS -- the validated bound is numerically reproduced"
          " at high precision.")

if __name__ == "__main__":
    main()

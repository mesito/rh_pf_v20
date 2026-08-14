"""
ic_core.py -- Core objects for Part VI (interference channels) of the manuscript.

Davenport-Heilbronn function, log-derivative coefficients c_n, the constant
kappa, and the zeta control. All other verify_*.py scripts import from here.

Conventions (matching the manuscript, Eq. (28.1)):
    chi = Dirichlet character mod 5 with chi(2) = i
    kappa = (sqrt(10 - 2 sqrt5) - 2) / (sqrt5 - 1)
    H(s) = ((1 - i kappa)/2) L(s, chi) + ((1 + i kappa)/2) L(s, chibar)
    coefficients a_n have period 5: (1, kappa, -kappa, -1, 0)
    -H'/H(s) = sum_{n>=2} c_n n^{-s},  recursion (28.2):
        c_n = a_n log n - sum_{d|n, d>1} a_d c_{n/d}

All heavy spectral computations use mpmath; light sweeps use float.
"""
import math
import mpmath as mp

mp.mp.dps = 25

# ---- kappa, Eq. (1) ----
KAPPA = (mp.sqrt(10 - 2*mp.sqrt(5)) - 2) / (mp.sqrt(5) - 1)

# period-5 coefficients a_n, n mod 5 -> value  (a_0 corresponds to n divisible by 5)
A_PERIOD = {1: mp.mpf(1), 2: KAPPA, 3: -KAPPA, 4: mp.mpf(-1), 0: mp.mpf(0)}

def a_n(n):
    """Period-5 Dirichlet coefficient of H."""
    return A_PERIOD[n % 5]

def dh_coeffs(N):
    """
    Log-derivative coefficients c_n of the DH function via recursion (2),
    returned as a list c[0..N] (c[0], c[1] unused; c[1]=0).
    Exact in mpmath.
    """
    logs = [mp.mpf(0)] + [mp.log(n) for n in range(1, N+1)]
    c = [mp.mpf(0)]*(N+1)
    for n in range(2, N+1):
        s = a_n(n)*logs[n]
        d = 2
        while d <= n:
            if n % d == 0 and d > 1:
                s -= a_n(d)*c[n//d]
            d += 1
        c[n] = s
    return c

def dh_coeffs_fast(N):
    """
    Float version of c_n via sieve-style accumulation (the implementation
    described in Appendix A), for n up to 6e4. Returns numpy array.
    """
    import numpy as np
    kap = float(KAPPA)
    av = {1: 1.0, 2: kap, 3: -kap, 4: -1.0, 0: 0.0}
    logs = np.zeros(N+1)
    logs[1:] = np.log(np.arange(1, N+1))
    c = np.zeros(N+1)
    acc = np.zeros(N+1)         # accumulates -sum_{d|n,d>1} a_d c_{n/d}
    for m in range(2, N+1):
        c[m] = av[m % 5]*logs[m] + acc[m]
        if c[m] != 0.0:
            d = 2
            while d*m <= N:
                acc[d*m] -= av[d % 5]*c[m]
                d += 1
    return c

def vonmangoldt(N):
    """Lambda(n) as float array c[0..N], the zeta control for the recursion."""
    import numpy as np
    L = np.zeros(N+1)
    for p in range(2, N+1):
        if all(p % q for q in range(2, int(p**0.5)+1)):
            pk = p
            while pk <= N:
                L[pk] = math.log(p)
                pk *= p
    return L

# ---- the two L-functions and H ----
def L_chi(s, conj=False):
    """L(s, chi) (or L(s, chibar)) via Hurwitz zeta, chi mod 5, chi(2)=i."""
    # chi(1)=1, chi(2)=i, chi(3)=-i, chi(4)=-1, chi(5)=0  (since chi(2)=i => chi(3)=chi(2)^3=-i? )
    # chi has order 4: chi(2)=i, chi(4)=chi(2)^2=-1, chi(3)=chi(2)*chi(... ); fix by table:
    i = mp.mpc(0, 1)
    chi = {1: mp.mpf(1), 2: i, 3: -i, 4: mp.mpf(-1)}
    if conj:
        chi = {k: mp.conj(v) for k, v in chi.items()}
    return mp.mpf(5)**(-s) * sum(chi[a]*mp.zeta(s, mp.mpf(a)/5) for a in range(1, 5))

def H(s):
    """DH function H(s), Eq. (1)."""
    i = mp.mpc(0, 1)
    return (1 - i*KAPPA)/2 * L_chi(s, conj=False) + (1 + i*KAPPA)/2 * L_chi(s, conj=True)

def H_via_coeffs(s, N=4000):
    """H(s) by its Dirichlet series with period-5 coefficients (check)."""
    return mp.mpf(1) + sum(a_n(n)*mp.mpf(n)**(-s) for n in range(2, N+1))

def Xi_DH(s):
    """Completed DH function Xi_DH(s) = (pi/5)^{-(s+1)/2} Gamma((s+1)/2) H(s)."""
    return (mp.pi/5)**(-(s+1)/2) * mp.gamma((s+1)/2) * H(s)

# ---- the two verified off-line zeros ----
RHO1 = mp.mpf('0.808517182457') + mp.mpf('85.6993484854')*mp.mpc(0, 1)
RHO2 = mp.mpf('0.65083008061') + mp.mpf('114.16334273')*mp.mpc(0, 1)
H1 = mp.mpf('0.308517')   # depth of rho1
H2 = mp.mpf('0.150830')   # depth of rho2

def banner(title):
    print("="*70)
    print(title)
    print("="*70)

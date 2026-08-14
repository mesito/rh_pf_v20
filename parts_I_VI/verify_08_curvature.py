"""
Part I (curvature detector): Curvature Test (the curvature detector)
Residual T_N(t) after subtracting N on-line zeros from d^2/dh^2 log|zeta|.

Paper (Section 8 table): |T30| < 0.02, T30/Tail in [0.49, 0.92].
An off-line zero at h0=0.1 would produce signal 2/h0^2 = 200, S/N > 10^4.
"""
# This script verifies results of the manuscript
# "Off-Line Zeros of the Riemann xi-Function" (repository v20).
# Section and theorem numbers refer to the manuscript.


import mpmath
from config import print_header


def curvature_test_residual(t, zeros, N_zeros=30, h=0.001, n_trivial=200):
    """
    Compute the residual T_N after subtracting pole, N on-line zeros, and trivial zeros
    from d^2/dh^2 log|zeta(1/2 + h + it)|.
    """
    s = mpmath.mpc(0.5 + h, t)

    # LHS: numerical second derivative
    eps = mpmath.mpf(5e-5)
    f_plus = mpmath.log(abs(mpmath.zeta(mpmath.mpc(0.5 + h + float(eps), t))))
    f_mid = mpmath.log(abs(mpmath.zeta(s)))
    f_minus = mpmath.log(abs(mpmath.zeta(mpmath.mpc(0.5 + h - float(eps), t))))
    d2h_numerical = (f_plus - 2 * f_mid + f_minus) / eps**2

    # Pole contribution: Re[-1/(s-1)^2]
    pole = mpmath.re(-1 / (s - 1)**2)

    # Nontrivial zeros (on-line): Re[-1/(s - rho)^2] for rho = 1/2 + i*gamma
    nontrivial_sum = mpmath.mpf(0)
    for k in range(min(N_zeros, len(zeros))):
        rho = mpmath.mpc(0.5, zeros[k])
        rho_conj = mpmath.mpc(0.5, -zeros[k])
        nontrivial_sum += mpmath.re(-1 / (s - rho)**2) + mpmath.re(-1 / (s - rho_conj)**2)

    # Trivial zeros: Re[-1/(s + 2n)^2]
    trivial_sum = mpmath.mpf(0)
    for n in range(1, n_trivial + 1):
        trivial_sum += mpmath.re(-1 / (s + 2 * n)**2)

    # RHS from partial fractions
    rhs = pole + nontrivial_sum + trivial_sum

    # Residual = LHS - RHS (should equal tail + off-line)
    residual = d2h_numerical - rhs

    # Tail estimate: log(gamma_N) / (2*pi*(gamma_N - t))
    gamma_N = zeros[N_zeros - 1] if N_zeros <= len(zeros) else zeros[-1]
    tail_est = mpmath.log(gamma_N) / (2 * mpmath.pi * (gamma_N - t))

    return float(residual), float(tail_est)


def verify_curvature_test(zeros, verbose=True):
    if verbose:
        print_header("Section 8: the Hadamard curvature detector (Numerical Observation 8.2)")

    t_values = [17.5, 20.0, 30.0, 35.0, 40.0, 45.0, 55.0, 60.0]
    N_zeros = 30
    h = 0.001

    if verbose:
        print(f"\nUsing N = {N_zeros} on-line zeros, h = {h}")
        print(f"\n  {'t':>6s}  {'T30':>10s}  {'Tail est':>10s}  {'T30/Tail':>10s}  {'RH?':>5s}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*5}")

    results = []
    all_consistent = True
    for t in t_values:
        residual, tail = curvature_test_residual(t, zeros, N_zeros, h)
        ratio = residual / tail if abs(tail) > 1e-20 else float('inf')
        consistent = abs(residual) < 0.02
        if not consistent:
            all_consistent = False

        results.append((t, residual, tail, ratio, consistent))
        if verbose:
            print(f"  {t:6.1f}  {residual:10.4f}  {tail:10.4f}  {ratio:10.2f}  {'Yes' if consistent else 'NO'}")

    # Signal from off-line zero at h0=0.1
    signal = 2 / 0.1**2  # = 200
    max_residual = max(abs(r[1]) for r in results)
    sn_ratio = signal / max_residual if max_residual > 0 else float('inf')

    if verbose:
        print(f"\nSignal from off-line zero at h0=0.1: 2/h0^2 = {signal:.0f}")
        print(f"Max |T30| = {max_residual:.4f}")
        print(f"Signal-to-noise > {sn_ratio:.0f}")
        print(f"All |T30| < 0.02: {all_consistent}")
        print(f"\nPaper reference: |T30| < 0.02, T30/Tail in [0.49, 0.92], S/N > 10^4")

    return {"all_consistent": all_consistent, "sn_ratio": sn_ratio}


if __name__ == "__main__":
    from config import load_all_zeros
    zeros = load_all_zeros()
    verify_curvature_test(zeros)

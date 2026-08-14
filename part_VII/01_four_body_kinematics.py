#!/usr/bin/env python3
"""
01_four_body_kinematics.py

Validates the exact four-body kinematics of Section 2 of the paper:

  * The four-body ODE for the quartet {+q, -q, +iy, -iy},
        dz_k/dt = 2 * sum_{j != k} (z_k - z_j)^{-1},
    conserves I = y^4 + 10 q^2 y^2 + q^4 and has d(q^2 - y^2)/dt = 12.
  * Closed form: with D0 = q0^2 - y0^2, I0 = I(q0,y0),
        D(t) = D0 + 12 t,  S(t) = sqrt((2 D^2 + I0)/3)   [= q^2 + y^2],
        q^2(t) = (S + D)/2,  y^2(t) = (S - D)/2.
  * Rate identity: d(q^2)/dt = 6 + 4 D/S = 2 + 8 q^2/(q^2 + y^2).
  * Mean rate over the full episode [t2, tau_dyn] equals 6, where
    t2 = (-sqrt(I0) - D0)/12 (the q -> 0 event) and
    tau_dyn = (sqrt(I0) - D0)/12 (the y -> 0 landing singularity).
  * Convexity: d^2(q^2)/dt^2 = 16 I0 / S^3.
  * Film formula: the monic quartic E(z;t) = z^4 + A(t) z^2 + B(t) with roots
    {+-q, +-iy} has A = K/2, B = M with M(t) = M0 - K0 t + 12 t^2,
    K(t) = K0 - 24 t  (equivalently A(t) = A0 - 12 t,
    B(t) = B0 - 2 A0 t + 12 t^2,  M0 = B0 = -q0^2 y0^2, K0 = 2 A0).
  * Identity swap: y^2 at the q -> 0 event equals sqrt(I0) and q^2 at the
    y -> 0 landing equals sqrt(I0).

Checks (a)-(g) per the task specification, for two initial conditions
(y0, q0) = (1, 2.7588) and (0.5, 1.2) (a third IC is added in full mode).

Tolerance justifications:
  * The ODE is integrated with DOP853 at rtol=1e-10 (fast) / 1e-12 (full),
    so trajectory-vs-closed-form agreement is demanded at < 1e-7 relative
    (integrator accuracy ~1e-9 or better; margin of ~100x).
  * Quantities computed algebraically from the ODE state (rate, convexity,
    film coefficients) agree with the analytic formulas to integrator
    accuracy; tolerance 1e-6 relative (convexity uses the analytically
    differentiated rate, exact given positions, so agreement is ~1e-15 in
    exact arithmetic and ~1e-9 in practice; 1e-6 is a safe margin).
  * The mean-rate check uses scipy.quad on a smooth analytic integrand;
    tolerance 1e-8 (achieved ~1e-13).
Usage: python 01_four_body_kinematics.py [--fast]
Exit code 0 iff every check passes.
"""
import sys
import numpy as np
from scipy.integrate import solve_ivp, quad

FAST = "--fast" in sys.argv

RESULTS = []


def report(name, ok, detail):
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} | {detail}")


def rhs(t, z):
    """Four-body RHS dz_k/dt = 2 sum_{j != k} (z_k - z_j)^{-1}."""
    out = np.empty_like(z)
    for k in range(len(z)):
        d = z[k] - z
        d[k] = np.inf
        out[k] = 2.0 * np.sum(1.0 / d)
    return out


def run_case(y0, q0, rtol, atol, nsamp):
    tag = f"(y0={y0}, q0={q0})"
    I0 = y0**4 + 10 * q0**2 * y0**2 + q0**4
    D0 = q0**2 - y0**2
    sqI = np.sqrt(I0)
    tau_dyn = (sqI - D0) / 12.0
    t2 = (-sqI - D0) / 12.0

    # (a) integrate the ODE up to 0.98 * tau_dyn (landing singularity at tau_dyn)
    tend = 0.98 * tau_dyn
    z0 = np.array([q0, -q0, 1j * y0, -1j * y0], dtype=complex)
    sol = solve_ivp(rhs, (0.0, tend), z0, method="DOP853",
                    rtol=rtol, atol=atol, dense_output=True)
    report(f"(a) ODE integration to 0.98*tau_dyn {tag}",
           sol.success and len(sol.t) > 5,
           f"t_end={tend:.6f} (tau_dyn={tau_dyn:.6f}), nfev={sol.nfev}, "
           f"success={sol.success}")
    if not sol.success:
        return

    ts = np.linspace(0.0, tend, nsamp)
    Z = sol.sol(ts)
    q = Z[0].real
    y = Z[2].imag
    u = q * q          # q^2
    v = y * y          # y^2

    D = D0 + 12.0 * ts
    S = np.sqrt((2.0 * D**2 + I0) / 3.0)
    q2_cl = (S + D) / 2.0
    y2_cl = (S - D) / 2.0

    # invariant checks: d(q^2 - y^2)/dt = 12  and  I conserved
    err_dudv = np.max(np.abs((u - v) - (D0 + 12.0 * ts)))
    I_traj = v**2 + 10.0 * u * v + u**2
    err_I = np.max(np.abs(I_traj - I0)) / I0
    report(f"(a2) invariants d(q^2-y^2)/dt=12, I const {tag}",
           err_dudv < 1e-8 and err_I < 1e-7,
           f"max |(q^2-y^2)-(D0+12t)|={err_dudv:.2e} (tol 1e-8), "
           f"max rel dI={err_I:.2e} (tol 1e-7)")

    # (b) closed form vs ODE trajectory
    err_b = max(np.max(np.abs(u - q2_cl) / q2_cl),
                np.max(np.abs(v - y2_cl) / y2_cl))
    report(f"(b) closed form q^2,y^2 vs ODE {tag}", err_b < 1e-7,
           f"max rel err={err_b:.2e} (tol 1e-7)")

    # (c) rate identity: 2 q qdot (from ODE RHS) = 6 + 4D/S = 2 + 8q^2/(q^2+y^2)
    rate_ode = np.array([2.0 * q[i] * rhs(0.0, Z[:, i])[0].real
                         for i in range(len(ts))])
    rate_c1 = 6.0 + 4.0 * D / S
    rate_c2 = 2.0 + 8.0 * u / (u + v)
    err_c1 = np.max(np.abs(rate_ode - rate_c1) / rate_c1)
    err_c2 = np.max(np.abs(rate_ode - rate_c2) / rate_c2)
    report(f"(c) rate identity d(q^2)/dt=6+4D/S=2+8q^2/(q^2+y^2) {tag}",
           err_c1 < 1e-7 and err_c2 < 1e-7,
           f"max rel err vs 6+4D/S: {err_c1:.2e}, vs 2+8q^2/(q^2+y^2): "
           f"{err_c2:.2e} (tol 1e-7)")

    # (d) mean rate over the episode [t2, tau_dyn] equals 6
    integrand = lambda t: 6.0 + 4.0 * (D0 + 12.0 * t) / np.sqrt(
        (2.0 * (D0 + 12.0 * t)**2 + I0) / 3.0)
    num, _ = quad(integrand, t2, tau_dyn, epsabs=1e-13, epsrel=1e-13)
    mean_rate = num / (tau_dyn - t2)
    # endpoint cross-check from the closed form: (q^2(tau)-q^2(t2))/(tau-t2)
    mean_endpoints = (sqI - 0.0) / (tau_dyn - t2)
    err_d = abs(mean_rate - 6.0)
    report(f"(d) mean rate over episode = 6 {tag}",
           err_d < 1e-8 and abs(mean_endpoints - 6.0) < 1e-12,
           f"quad mean={mean_rate:.12f}, endpoint mean={mean_endpoints:.12f} "
           f"(tol 1e-8)")

    # (e) convexity: d^2(q^2)/dt^2 = 16 I0 / S^3
    #     computed exactly from ODE positions: r = 2+8u/(u+v),
    #     dr/dt = 8 (u' v - u v')/(u+v)^2 with u' = 2+8u/(u+v), v' = -2-8v/(u+v)
    udot = 2.0 + 8.0 * u / (u + v)
    vdot = -2.0 - 8.0 * v / (u + v)
    drdt = 8.0 * (udot * v - u * vdot) / (u + v)**2
    convex = 16.0 * I_traj / (u + v)**3
    err_e = np.max(np.abs(drdt - convex) / convex)
    report(f"(e) convexity d^2(q^2)/dt^2 = 16 I0/S^3 {tag}", err_e < 1e-6,
           f"max rel err={err_e:.2e} (tol 1e-6)")

    # (f) film formula: E(z;t) = z^4 + A z^2 + B, A = K/2, B = M,
    #     M(t) = M0 - K0 t + 12 t^2, K(t) = K0 - 24 t
    A0 = v[0] - u[0]
    B0 = -u[0] * v[0]
    M0, K0 = B0, 2.0 * A0
    A_t = K0 / 2.0 - 12.0 * ts              # A(t) = K(t)/2
    B_t = M0 - K0 * ts + 12.0 * ts**2       # M(t)
    err_A = np.max(np.abs(A_t - (v - u)))
    err_B = np.max(np.abs(B_t - (-u * v)))
    report(f"(f) film formula A=K/2, B=M, M=M0-K0 t+12t^2 {tag}",
           err_A < 1e-8 and err_B < 1e-8,
           f"max |A err|={err_A:.2e}, max |B err|={err_B:.2e} (tol 1e-8; "
           f"coefficients O(10), integrator-limited)")

    # (g) identity swap: y^2 at q->0 event = sqrt(I0); q^2 at landing = sqrt(I0)
    S_at_t2 = np.sqrt((2.0 * (D0 + 12.0 * t2)**2 + I0) / 3.0)
    y2_at_q0event = (S_at_t2 - (D0 + 12.0 * t2)) / 2.0
    S_at_tau = np.sqrt((2.0 * (D0 + 12.0 * tau_dyn)**2 + I0) / 3.0)
    q2_at_landing = (S_at_tau + (D0 + 12.0 * tau_dyn)) / 2.0
    err_g = max(abs(y2_at_q0event - sqI) / sqI, abs(q2_at_landing - sqI) / sqI)
    report(f"(g) identity swap: y^2(q->0)=q^2(landing)=sqrt(I0) {tag}",
           err_g < 1e-12,
           f"y^2(q->0)={y2_at_q0event:.12f}, q^2(landing)={q2_at_landing:.12f},"
           f" sqrt(I0)={sqI:.12f}, max rel err={err_g:.2e} (tol 1e-12)")


def main():
    if FAST:
        rtol, atol, nsamp = 1e-10, 1e-13, 200
        cases = [(1.0, 2.7588), (0.5, 1.2)]
    else:
        rtol, atol, nsamp = 1e-12, 1e-14, 500
        cases = [(1.0, 2.7588), (0.5, 1.2), (2.0, 0.9)]
    print(f"01_four_body_kinematics.py  mode={'fast' if FAST else 'full'}")
    for y0, q0 in cases:
        run_case(y0, q0, rtol, atol, nsamp)
    n_pass = sum(RESULTS)
    print(f"SUMMARY: {n_pass}/{len(RESULTS)} checks passed")
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()

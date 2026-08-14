#!/usr/bin/env python3
"""
02_accelerated_fall.py

Validates the accelerated-fall mechanism of Section 3 of the paper (claim T2):
in the FULL de Bruijn-Newman-type flow with a surrounding on-line lattice, the
off-line pair falls strictly faster than in the isolated four-body model, and
the excess fall rate is exactly the lattice back-pressure term.

Full-flow ODE: dz_k/dt = 2 sum_{j != k} (z_k - z_j)^{-1} for
  * N = 150 on-line zeros per side (AP lattice, spacing delta = 0.45), hole
    half-width q0 = 1.5 delta, outer gaps G_L = delta (left) and
    G_R in {2.5 delta (control), 5 delta (violating)} (right);
  * one off-line pair c(t) +- i y(t), c(0) = 0, y0 = 0.5.
Model (isolated quartet, Section 2):
  I0 = y0^4 + 10 q0^2 y0^2 + q0^4, D0 = q0^2 - y0^2,
  D(t) = D0 + 12 t, S(t) = sqrt((2 D^2 + I0)/3), q^2_mod(t) = (S + D)/2,
  tau_dyn = (sqrt(I0) - D0)/12 = 0.081788.

Checks (both configurations unless stated otherwise):
(i)   sign(ydot_real - ydot_model) is strictly NEGATIVE at every saved time
      step (dense samples), where the exact quartet part of ydot is
      ydot_model(t) = -1/y - 2 y [ 1/((c-q_R)^2+y^2) + 1/((c-q_L)^2+y^2) ]
      with the current positions; hence ydot_real - ydot_model =
      -2 y sum_far 1/((c-x_j)^2+y^2) < 0 for ANY lattice.
(ii)  that difference equals -2 y sum_{far j} 1/((c-x_j)^2+y^2) computed
      directly from the instantaneous positions, to < 1e-9 at every sampled
      time (algebraic identity given the ODE; the far sum excludes the two
      hole zeros q_L, q_R).  Tolerance justification: the identity is exact
      algebra from the RHS; the only error is integrator error (rtol 1e-10,
      so ~1e-10 in positions; 1e-9 is a 10x margin).
(iii) lifetime ratio: tau_real (pair-landing time, event y = 0.02 closed off
      with the exact asymptotics y^2 ~ 2(tau-t), correction O(y_ev^4) ~ 1e-7)
      vs model tau_dyn.  Measured: G_R = 5 delta -> 0.0562 (31.3% shorter),
      G_R = 2.5 delta -> 0.0531 (35.0% shorter); task-sheet expectation
      "~ -35%" is confirmed for the control and slightly less for the
      violating config (its far zeros are farther away and pull less).
      PASS if both are shorter by > 20%.
(iv)  control configuration only: q^2_real(t) <= q^2_model(t) + 1e-9 at every
      sampled time (the model is an upper bound when Sigma_far <= 0).

Usage: python 02_accelerated_fall.py [--fast]
Exit code 0 iff every check passes.
"""
import sys
import numpy as np
from scipy.integrate import solve_ivp

FAST = "--fast" in sys.argv
RESULTS = []


def report(name, ok, detail):
    RESULTS.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name} | {detail}")


DELTA = 0.45
NSIDE = 150
Q0 = 1.5 * DELTA
Y0 = 0.5
GL = DELTA
I0 = Y0**4 + 10 * Q0**2 * Y0**2 + Q0**4
D0 = Q0**2 - Y0**2
TAU_DYN = (np.sqrt(I0) - D0) / 12.0


def build(GR):
    xs = [-Q0, Q0]
    for k in range(1, NSIDE):
        xs.append(-Q0 - GL - (k - 1) * DELTA)
        xs.append(+Q0 + GR + (k - 1) * DELTA)
    xs = np.sort(np.array(xs))
    return xs, np.concatenate([xs.astype(complex), [1j * Y0, -1j * Y0]])


def rhs(t, z):
    D = z[:, None] - z[None, :]
    np.fill_diagonal(D, np.inf)
    return 2.0 * np.sum(1.0 / D, axis=1)


def run(GR, y_ev, rtol, atol):
    xs, z0 = build(GR)
    iR = int(np.argmin(np.abs(xs - Q0)))   # right hole zero
    iL = int(np.argmin(np.abs(xs + Q0)))   # left hole zero
    ip = xs.size                            # +i y0 particle

    def ev(t, z):
        return z[ip].imag - y_ev
    ev.terminal = True
    sol = solve_ivp(rhs, (0.0, 0.3), z0, method="DOP853",
                    rtol=rtol, atol=atol, events=ev, dense_output=True)
    return sol, xs, iR, iL, ip


def model_q2(ts):
    D = D0 + 12.0 * ts
    S = np.sqrt((2.0 * D**2 + I0) / 3.0)
    return (S + D) / 2.0


def analyze(GR, label, y_ev, rtol, atol):
    sol, xs, iR, iL, ip = run(GR, y_ev, rtol, atol)
    ok_conv = sol.status == 1
    ts_d = np.unique(np.concatenate([sol.t, np.linspace(0.0, sol.t[-1], 300)]))
    Z = sol.sol(ts_d)
    qR = Z[iR].real
    qL = Z[iL].real
    c = Z[ip].real
    y = Z[ip].imag
    nr = xs.size  # number of real zeros
    sig_neg = True
    max_id_err = 0.0
    for i in range(len(ts_d)):
        v = rhs(0.0, Z[:, i])
        ydot_real = v[ip].imag
        ydot_model = (-1.0 / y[i]
                      - 2.0 * y[i] * (1.0 / ((c[i] - qR[i])**2 + y[i]**2)
                                      + 1.0 / ((c[i] - qL[i])**2 + y[i]**2)))
        # far sum: all real zeros except the two hole zeros
        mask = np.ones(nr, bool)
        mask[[iR, iL]] = False
        xfar = Z[:nr, i].real[mask]
        sfar = np.sum(1.0 / ((c[i] - xfar)**2 + y[i]**2))
        diff = ydot_real - ydot_model
        sig_neg &= diff < 0.0
        max_id_err = max(max_id_err, abs(diff + 2.0 * y[i] * sfar))
    report(f"(i) sign(ydot_real - ydot_model) < 0 at every sampled time "
           f"[{label}]", sig_neg,
           f"{len(ts_d)} samples, all negative: {sig_neg}")
    report(f"(ii) difference = -2 y sum_far 1/((c-x_j)^2+y^2) exactly "
           f"[{label}]", max_id_err < 1e-9,
           f"max |identity error| = {max_id_err:.2e} (tol 1e-9)")
    tau_real = sol.t[-1] + Z[ip, -1].imag**2 / 2.0 if False else \
        sol.t[-1] + sol.y[ip, -1].imag**2 / 2.0
    shorter = 1.0 - tau_real / TAU_DYN
    report(f"(iii) accelerated fall: tau_real < tau_dyn={TAU_DYN:.6f} by "
           f"> 20% [{label}]", shorter > 0.20,
           f"tau_real={tau_real:.6f}, shorter by {100*shorter:.1f}%")
    if label.startswith("G_R=2.5d"):
        q2m = model_q2(ts_d)
        exc = np.max(qR**2 - q2m)
        report("(iv) control: q^2_real(t) <= q^2_model(t)+1e-9 at every "
               "sampled time", np.all(qR**2 <= q2m + 1e-9),
               f"max(q^2_real - q^2_model) = {exc:.3e}")
    return ok_conv, tau_real, shorter


def main():
    print(f"02_accelerated_fall.py  mode={'fast' if FAST else 'full'}")
    y_ev = 0.02 if FAST else 0.01
    rtol, atol = (1e-10, 1e-12) if FAST else (1e-12, 1e-14)
    rows = []
    for GR, label in [(2.5 * DELTA, "G_R=2.5d (control)"),
                      (5 * DELTA, "G_R=5d (violating)")]:
        ok_conv, tr, sh = analyze(GR, label, y_ev, rtol, atol)
        report(f"(conv) integration reaches y={y_ev} landing event [{label}]",
               ok_conv, f"nfev ok, tau_real={tr:.6f}")
        rows.append((label, tr, sh))
    print("\nlifetime summary (model tau_dyn = "
          f"{TAU_DYN:.6f}):")
    for label, tr, sh in rows:
        print(f"  {label:<22} tau_real={tr:.6f}  ratio={tr/TAU_DYN:.4f}  "
              f"({100*sh:.1f}% shorter than model)")
    print(f"\nSUMMARY: {sum(RESULTS)}/{len(RESULTS)} checks passed")
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()

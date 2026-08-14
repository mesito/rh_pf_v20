#!/usr/bin/env python3
"""
04_counterexample_16_1.py

Validates the OP16.1 counterexample, the frame distinction of Theorems 5.1/5.2,
and the accelerated-fall sign check (Sections 3 and 5 of the paper).

Full-flow ODE: dz_k/dt = 2 sum_{j != k} (z_k - z_j)^{-1} (self-term masked with
np.inf) for N = 150 on-line zeros per side (AP lattice, spacing delta = 0.45),
hole half-width q0 = 1.5 delta, outer gaps G_L = delta (left) and G_R (right)
in {2.5 delta (satisfying/control), 5 delta (violating)}, plus one off-line
pair c(t) +- i y(t), c(0) = 0, y0 = 0.5.  Integration: DOP853, rtol=1e-10,
atol=1e-12, terminal event y = 0.02; the pair-landing singularity is expected
and the lifetime is closed off with the exact asymptotics y^2 ~ 2 (tau - t):
tau_real = t_event + y_event^2/2 (error O(y_event^4) ~ 1e-7, negligible
against the >= 25% effects tested).

Model trajectory (isolated quartet, Section 2):
  I0 = y0^4 + 10 q0^2 y0^2 + q0^4, D0 = q0^2 - y0^2, D(t) = D0 + 12 t,
  S(t) = sqrt((2 D^2 + I0)/3), q^2_mod(t) = (S + D)/2,
  tau_dyn = (sqrt(I0) - D0)/12 = 0.081788, delivered q_mod(tau_dyn) =
  I0^{1/4} = 1.0895.

TWO READINGS of the hole half-width (the frame distinction of Thm 5.1/5.2):
  birth-anchored:  q_ba(t) = x_R(t) - c(0)   (= x_R(t), since c(0) = 0)
  foot-relative:   q_fr(t) = x_R(t) - c(t)   (relative to the moving pair foot)
with eta(t) = d(q^2)/dt - (2 + 8 q^2/(q^2 + y^2)) in each reading (the rate
uses the true velocity: 2 q_ba x_R' for birth-anchored, 2 q_fr (x_R' - c')
for foot-relative).

CANONICAL CONSTANTS (lead-verified clean re-run; this script uses the masked
np.inf self-term throughout and reproduces them):
  G_R = 5 delta, birth-anchored: eta in [+1.215, +1.469] throughout,
  int(eta) = +0.080, mid-episode q^2 overshoot +0.103, delivered
  q_T = 1.020 vs model 1.0895, lifetimes 0.0562 vs 0.0818 (~ -31%).
  G_R = 5 delta, foot-relative: eta in [-7.360, -4.249] -- NO violation
  anywhere, int(eta) = -0.327.
  Control G_R = 2.5 delta: the bound holds in BOTH readings.

Checks:
(a) control G_R = 2.5 delta: eta_ba <= 1e-9 and eta_fr <= 1e-9 at every
    sampled time (bound holds in both readings), and
    q^2_ba(t) <= q^2_mod(t) + 1e-9 at every sampled time.
(b) G_R = 5 delta:
    (b1) eta_ba > 0 at every sampled time (violation throughout in the
         birth-anchored reading); measured band [+1.219, +1.467] vs canonical
         [+1.215, +1.469] within 5% on the edges;
    (b2) int(eta_ba) dt = +0.080 within 5%; mid-episode q^2 overshoot
         +0.103 within 5%; delivered q_T = 1.020 vs model 1.0895 within 5%;
         lifetime shorter than model by > 25% (measured 31.3%, canonical
         ~ -31%);
    (b3) eta_fr < 0 at every sampled time (NO violation in the foot-relative
         reading); int(eta_fr) = -0.327 within 5%.
(c) Thm 5.2 foot-relative decomposition (EXACT algebra, both configs):
    d(q^2)/dt - [2 + 8 q^2/(q^2+y^2)] = A1 + A2 + sum_j T_j,  q = x_R - c,
    q' = c - x_L,
      A1 = 2(q - q')/(q + q'),
      A2 = 4 q [ q/(q^2+y^2) - q'/(q'^2+y^2) ],
      T_j = 4 q [ 1/(x_R - x_j) - (c - x_j)/((c - x_j)^2 + y^2) ],
    verified to < 1e-8 relative at every sampled time (LHS from trajectory
    velocities, RHS from positions), plus the sign statements:
    right-far T_j < 0 always; left-far T_j < 0 iff u := c - x_j > y^2/q.
(d) accelerated-fall sign check at t = 0 (imaginary parts; never affected by
    the self-term issue): ydot_real < ydot_model and the difference equals
    -2 y0 sum_{far j} 1/((c - x_j)^2 + y0^2), verified to < 1e-12.

A summary table of both configurations is printed.
Usage: python 04_counterexample_16_1.py [--fast]
Exit code 0 iff every check passes.
"""
import sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import digamma

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
Q_MODEL_DELIVERED = I0**0.25          # 1.0895

# canonical constants (lead-verified)
CANON = {
    "eta_ba_min": 1.215, "eta_ba_max": 1.469, "int_eta_ba": 0.080,
    "overshoot": 0.103, "q_T": 1.020, "q_model": 1.0895,
    "eta_fr_min": -7.360, "eta_fr_max": -4.249, "int_eta_fr": -0.327,
    "tau_real": 0.0562,
}


def build(GR):
    xs = [-Q0, Q0]
    for k in range(1, NSIDE):
        xs.append(-Q0 - GL - (k - 1) * DELTA)
        xs.append(+Q0 + GR + (k - 1) * DELTA)
    xs = np.sort(np.array(xs))
    assert len(xs) == 2 * NSIDE
    return xs, np.concatenate([xs.astype(complex), [1j * Y0, -1j * Y0]])


def rhs(t, z):
    D = z[:, None] - z[None, :]
    np.fill_diagonal(D, np.inf)
    return 2.0 * np.sum(1.0 / D, axis=1)


def run(GR, y_ev, rtol, atol):
    xs, z0 = build(GR)
    iR = int(np.argmin(np.abs(xs - Q0)))
    iL = int(np.argmin(np.abs(xs + Q0)))
    ip = xs.size

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


def diagnostics(sol, xs, iR, iL, ip):
    """Per-time diagnostics in both readings, evaluated from the ODE RHS."""
    ts = np.unique(np.concatenate([sol.t, np.linspace(0.0, sol.t[-1], 400)]))
    Z = sol.sol(ts)
    nr = xs.size
    xR = Z[iR].real
    xL = Z[iL].real
    c = Z[ip].real
    y = Z[ip].imag
    eta_ba = np.empty(len(ts))
    eta_fr = np.empty(len(ts))
    dec_err = np.zeros(len(ts))       # Thm 5.2 decomposition error
    sign_right_ok = True
    sign_left_ok = True
    for i in range(len(ts)):
        v = rhs(0.0, Z[:, i])
        vxR = v[iR].real
        vc = v[ip].real
        q_ba = xR[i]                  # c(0) = 0
        q_fr = xR[i] - c[i]
        eta_ba[i] = (2.0 * q_ba * vxR
                     - (2.0 + 8.0 * q_ba**2 / (q_ba**2 + y[i]**2)))
        eta_fr[i] = (2.0 * q_fr * (vxR - vc)
                     - (2.0 + 8.0 * q_fr**2 / (q_fr**2 + y[i]**2)))
        # ---- Thm 5.2 decomposition (foot-relative) -----------------------
        q = q_fr
        qp = c[i] - xL[i]
        mask = np.ones(nr, bool)
        mask[[iR, iL]] = False
        xj = Z[:nr, i].real[mask]
        A1 = 2.0 * (q - qp) / (q + qp)
        A2 = 4.0 * q * (q / (q**2 + y[i]**2) - qp / (qp**2 + y[i]**2))
        u = c[i] - xj
        T = 4.0 * q * (1.0 / (xR[i] - xj) - u / (u**2 + y[i]**2))
        lhs = (2.0 * q * (vxR - vc)
               - (2.0 + 8.0 * q**2 / (q**2 + y[i]**2)))
        rhsv = A1 + A2 + np.sum(T)
        dec_err[i] = abs(lhs - rhsv) / max(abs(lhs), 1e-30)
        # sign statements
        right = xj > xR[i]
        left = xj < xL[i]
        if np.any(T[right] >= 0.0):
            sign_right_ok = False
        # left-far T_j < 0 iff u > y^2/q
        pred_neg = u[left] > y[i]**2 / q
        if not np.all((T[left] < 0.0) == pred_neg):
            sign_left_ok = False
    return dict(ts=ts, xR=xR, xL=xL, c=c, y=y, eta_ba=eta_ba, eta_fr=eta_fr,
                dec_err=dec_err, sign_right_ok=sign_right_ok,
                sign_left_ok=sign_left_ok)


def main():
    print(f"04_counterexample_16_1.py  mode={'fast' if FAST else 'full'}")
    print(f"model: tau_dyn={TAU_DYN:.6f}, I0={I0:.6f}, D0={D0:.6f}, "
          f"delivered q_mod={Q_MODEL_DELIVERED:.4f}")
    y_ev = 0.02 if FAST else 0.01
    rtol, atol = (1e-10, 1e-12) if FAST else (1e-12, 1e-14)

    data = {}
    for GR, label in [(5 * DELTA, "G_R=5d"), (2.5 * DELTA, "G_R=2.5d")]:
        sol, xs, iR, iL, ip = run(GR, y_ev, rtol, atol)
        report(f"(conv) integration reaches y={y_ev} landing event [{label}]",
               sol.status == 1, f"t_ev={sol.t[-1]:.6f}, nfev={sol.nfev}")
        d = diagnostics(sol, xs, iR, iL, ip)
        d["tau_real"] = sol.t[-1] + sol.y[ip, -1].imag**2 / 2.0
        d["sol"] = sol
        data[label] = d

    # ---- (a) control: bound in both readings ------------------------------
    d = data["G_R=2.5d"]
    report("(a1) control: eta_ba <= 1e-9 at every sampled time",
           np.all(d["eta_ba"] <= 1e-9),
           f"eta_ba in [{d['eta_ba'].min():.4f}, {d['eta_ba'].max():.4f}]")
    report("(a2) control: eta_fr <= 1e-9 at every sampled time",
           np.all(d["eta_fr"] <= 1e-9),
           f"eta_fr in [{d['eta_fr'].min():.4f}, {d['eta_fr'].max():.4f}]")
    q2m = model_q2(d["ts"])
    report("(a3) control: q^2_ba(t) <= q^2_mod(t)+1e-9 at every sampled time",
           np.all(d["xR"]**2 <= q2m + 1e-9),
           f"max(q^2_ba - q^2_mod) = {np.max(d['xR']**2 - q2m):.3e}")

    # ---- (b) violating configuration --------------------------------------
    d = data["G_R=5d"]
    eb = d["eta_ba"]
    ef = d["eta_fr"]
    ts = d["ts"]
    int_eb = np.trapezoid(eb, ts)
    int_ef = np.trapezoid(ef, ts)
    q2m = model_q2(ts)
    overshoot = np.max(d["xR"]**2 - q2m)
    # delivered q_T: extrapolate q^2 from the event point to y = 0 using the
    # terminal rate (constant to O(y_ev^2))
    sol5 = d["sol"]
    xs5 = build(5 * DELTA)[0]
    iR5 = int(np.argmin(np.abs(xs5 - Q0)))
    ip5 = xs5.size
    v_end = rhs(0.0, sol5.y[:, -1])
    q_end = sol5.y[iR5, -1].real
    y_end = sol5.y[ip5, -1].imag
    rate_q2_end = 2.0 * q_end * v_end[iR5].real
    dt_rem = y_end**2 / 2.0
    qT = np.sqrt(q_end**2 + rate_q2_end * dt_rem)
    shorter = 1.0 - d["tau_real"] / TAU_DYN
    ok_band = (abs(eb.min() - CANON["eta_ba_min"]) / CANON["eta_ba_min"] < 0.05
               and abs(eb.max() - CANON["eta_ba_max"]) / CANON["eta_ba_max"]
               < 0.05)
    report("(b1) birth-anchored violation throughout: eta_ba > 0 every "
           "sampled time; band matches canonical [1.215,1.469] within 5%",
           np.all(eb > 0.0) and ok_band,
           f"eta_ba in [{eb.min():.4f}, {eb.max():.4f}] over {len(ts)} "
           f"samples (canonical [1.215, 1.469])")
    ok_nums = (abs(int_eb - CANON["int_eta_ba"]) / CANON["int_eta_ba"] < 0.05
               and abs(overshoot - CANON["overshoot"]) / CANON["overshoot"]
               < 0.05
               and abs(qT - CANON["q_T"]) / CANON["q_T"] < 0.05)
    report("(b2) canonical numbers: int(eta_ba)=0.080, overshoot=0.103, "
           "q_T=1.020 (all within 5%); lifetime > 25% shorter",
           ok_nums and shorter > 0.25,
           f"int(eta_ba)={int_eb:.4f}, overshoot={overshoot:.4f}, "
           f"q_T={qT:.4f} (model delivered {Q_MODEL_DELIVERED:.4f}), "
           f"tau_real={d['tau_real']:.6f} vs tau_dyn={TAU_DYN:.6f} "
           f"({100 * shorter:.1f}% shorter, canonical ~-31%)")
    ok_fr = (np.all(ef < 0.0)
             and abs(int_ef - CANON["int_eta_fr"]) / abs(CANON["int_eta_fr"])
             < 0.05)
    report("(b3) foot-relative: NO violation anywhere (eta_fr < 0 every "
           "sampled time); int(eta_fr) = -0.327 within 5%",
           ok_fr,
           f"eta_fr in [{ef.min():.4f}, {ef.max():.4f}] "
           f"(canonical [-7.360,-4.249]), int={int_ef:.4f}")

    # ---- (c) Thm 5.2 decomposition + sign statements ----------------------
    for label in ["G_R=5d", "G_R=2.5d"]:
        d = data[label]
        report(f"(c) Thm 5.2 foot-relative decomposition identity < 1e-8 "
               f"rel [{label}]", np.max(d["dec_err"]) < 1e-8,
               f"max rel err={np.max(d['dec_err']):.2e} over {len(d['ts'])}"
               f" samples")
        report(f"(c2) sign statements: right-far T_j<0 always; left-far "
               f"T_j<0 iff u>y^2/q [{label}]",
               d["sign_right_ok"] and d["sign_left_ok"],
               f"right-far always negative: {d['sign_right_ok']}, "
               f"left-far iff-rule: {d['sign_left_ok']}")

    # ---- (d) accelerated-fall identity at t = 0 ---------------------------
    xs, z0 = build(5 * DELTA)
    ip = xs.size
    v0 = rhs(0.0, z0)
    ydot_real = v0[ip].imag
    ydot_model = -4.0 * Y0 / (Q0**2 + Y0**2) - 1.0 / Y0
    far = np.delete(xs, [np.argmin(np.abs(xs + Q0)), np.argmin(np.abs(xs - Q0))])
    s_far = np.sum(1.0 / (far**2 + Y0**2))
    diff = ydot_real - ydot_model
    pred = -2.0 * Y0 * s_far
    report("(d) ydot_real < ydot_model and diff = -2 y0 sum_far "
           "1/(x_j^2+y0^2) at t=0",
           diff < 0.0 and abs(diff - pred) < 1e-12,
           f"ydot_real={ydot_real:.6f}, ydot_model={ydot_model:.6f}, "
           f"diff={diff:.10f}, predicted={pred:.10f}, "
           f"|diff-pred|={abs(diff - pred):.2e} (tol 1e-12)")

    # ---- summary table -----------------------------------------------------
    print("\nSUMMARY TABLE (canonical constants in parentheses)")
    print(f"{'config':<12}{'reading':<16}{'eta_min':>10}{'eta_max':>10}"
          f"{'int_eta':>10}{'tau_real':>10}{'vs model':>10}")
    for label in ["G_R=5d", "G_R=2.5d"]:
        d = data[label]
        for rd, en in [("birth-anchored", d["eta_ba"]),
                       ("foot-relative", d["eta_fr"])]:
            print(f"{label:<12}{rd:<16}{en.min():>10.4f}{en.max():>10.4f}"
                  f"{np.trapezoid(en, d['ts']):>10.4f}{d['tau_real']:>10.5f}"
                  f"{100 * (1 - d['tau_real'] / TAU_DYN):>8.1f}%")
    print(f"(canonical: 5d birth-anchored [1.215,1.469], int 0.080, "
          f"overshoot 0.103, q_T 1.020 vs model 1.0895, -31%; "
          f"5d foot-relative [-7.360,-4.249], int -0.327)")

    print(f"\nSUMMARY: {sum(RESULTS)}/{len(RESULTS)} checks passed")
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()

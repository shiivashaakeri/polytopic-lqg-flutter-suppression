"""
Experiment 4: continuous Dryden turbulence.

Three sub-experiments:
    4a. Time-domain diagnostic at V = 55 m/s.
    4b. Frequency-domain PSD comparison (closed vs open).
    4c. Envelope sweep: RMS flutter amplitude vs V.

Run from project root:
    python -m examples.lqg_flipased_demo.experiment_turbulence
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sps

from examples.lqg_flipased_demo.dynamics import alpha_at
from examples.lqg_flipased_demo.gust import (
    GustForcing,
    short_period_B_w,
    simulate_with_gust,
)
from examples.lqg_flipased_demo.pipeline import (
    SimResult,
    _blend_poly,
    _blend_tp,
    synthesize,
)


warnings.filterwarnings(
    "ignore", message=".*horizon may be insufficient.*", category=UserWarning
)


CHANNELS = ["sym", "asym"]
HULL = "cno"
V_TEST  = 55.0
T_DIAG  = 20.0
T_RMS   = 10.0

SIGMA_W_MOD = 1.5
SIGMA_W_SEV = 3.0
L_W         = 162.0

M_4C = 20

COL_CLOSED  = "#0072B2"
COL_OPEN    = "#D55E00"
COL_TURB    = "#999999"
COL_CONTROL = "#009E73"
COL_BAND_A  = 0.20

FLUTTER_FULL_IDX = {"sym": (8, 9), "asym": (5, 6)}


# ---------------------------------------------------------------------
# Dryden filter
# ---------------------------------------------------------------------

def dryden_continuous_tf(sigma_w, L_w, V):
    a = L_w / V
    gain = sigma_w * np.sqrt(L_w / (np.pi * V))
    num = np.array([gain * np.sqrt(3.0) * a, gain])
    den = np.array([a * a, 2.0 * a, 1.0])
    return num, den


def dryden_realization(sigma_w, L_w, V, dt, n_steps, rng):
    num_c, den_c = dryden_continuous_tf(sigma_w, L_w, V)
    sys_d = sps.lti(num_c, den_c).to_discrete(dt, method="bilinear")
    pilot_steps = max(n_steps, 20000)
    rng_pilot = np.random.default_rng(seed=12345)
    u_pilot = rng_pilot.standard_normal(pilot_steps + 1) / np.sqrt(dt)
    w_pilot = np.asarray(sps.dlsim(sys_d, u_pilot,
                          t=np.arange(pilot_steps+1)*dt)[1]).ravel()
    realized_sigma = float(np.std(w_pilot[pilot_steps // 5:]))
    cal = sigma_w / realized_sigma if realized_sigma > 0 else 1.0
    u_white = rng.standard_normal(n_steps + 1) / np.sqrt(dt)
    w_g = np.asarray(sps.dlsim(sys_d, u_white,
                      t=np.arange(n_steps+1)*dt)[1]).ravel() * cal
    return w_g


def make_dryden_forcing(channel, n_full, sigma_w, L_w, V, dt, T_final,
                         rng, label=None):
    n_steps = int(np.round(T_final / dt))
    w_arr = dryden_realization(sigma_w, L_w, V, dt, n_steps, rng)
    def profile(t, _w=w_arr, _dt=dt):
        idx = int(np.round(float(t) / _dt))
        return float(_w[max(0, min(idx, len(_w)-1))])
    if label is None:
        label = (f"Dryden $\\sigma_w$={sigma_w:.1f} m/s, "
                 f"$L_w$={L_w:.0f} m, V={V:.0f} m/s")
    return GustForcing(B_w=short_period_B_w(channel, n_full),
                       profile=profile, label=label)


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def _flutter_amp(x_full, channel):
    return np.linalg.norm(x_full[list(FLUTTER_FULL_IDX[channel]), :], axis=0)


def _style_ax(ax):
    ax.grid(True, lw=0.3, alpha=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in")


# ---------------------------------------------------------------------
# 4a
# ---------------------------------------------------------------------

def experiment_4a(synth, channel, fig_dir, res_dir):
    print(f"\n  4a [{channel}]: time-domain diagnostic at V={V_TEST}, T={T_DIAG}s")
    rng = np.random.default_rng(seed=0xDED)
    gust = make_dryden_forcing(channel=channel, n_full=synth.poly_full_d.n,
                                sigma_w=SIGMA_W_MOD, L_w=L_W, V=V_TEST,
                                dt=synth.dt, T_final=T_DIAG, rng=rng)
    sim = simulate_with_gust(synth, V=V_TEST, gust=gust,
                              T_final=T_DIAG, verbose=True)
    amp = _flutter_amp(sim.x_full, channel)
    rms_amp = float(np.sqrt(np.mean(amp**2)))
    rms_u   = float(np.sqrt(np.mean(sim.u**2))) * 1e3
    max_u   = float(np.max(np.abs(sim.u))) * 1e3
    print(f"    RMS amp={rms_amp:.3e}, RMS u={rms_u:.3f} mrad, max u={max_u:.3f} mrad")

    fig, axes = plt.subplots(4, 1, figsize=(9.0, 9.5), sharex=True)
    ax = axes[0]
    ax.plot(sim.t, sim.w_g, color=COL_TURB, lw=0.8)
    ax.set_ylabel(r"$w_g$ (m/s)"); ax.set_title(f"(a) Dryden turbulence  ({gust.label})")
    _style_ax(ax)

    ax = axes[1]
    ax.semilogy(sim.t, np.maximum(amp, 1e-12), color=COL_CLOSED, lw=0.9)
    ax.set_ylabel(r"$\|x_{\rm flutter}\|$  (log)")
    ax.set_title(f"(b) flutter amplitude  (RMS={rms_amp:.2e})")
    _style_ax(ax)

    ax = axes[2]
    ax.plot(sim.t, sim.u[0,:]*1e3, color=COL_CONTROL, lw=0.8)
    ax.set_ylabel(r"$u$ (mrad)")
    ax.set_title(f"(c) control  (RMS={rms_u:.2f} mrad, max={max_u:.1f} mrad)")
    _style_ax(ax)

    ax = axes[3]
    ax.plot(sim.t, sim.y[0,:], color="#009E73", lw=0.8, label=r"$r_{y}$")
    ax.plot(sim.t, sim.y[1,:], color="#CC79A7", lw=0.8, label=r"$r_{x}$")
    ax.set_ylabel("y (rad/s)"); ax.set_xlabel("time (s)")
    ax.set_title("(d) measured outputs")
    ax.legend(frameon=False, loc="upper right", fontsize=9); _style_ax(ax)

    fig.suptitle(f"Experiment 4a [{channel}]: Dryden turbulence closed-loop response", fontsize=11)
    fig.tight_layout(rect=(0,0,1,0.97))
    fig.savefig(fig_dir / f"turbulence_4a_{channel}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    np.savez(res_dir / f"turbulence_4a_{channel}.npz",
             t=sim.t, w_g=sim.w_g, x_full=sim.x_full,
             u=sim.u, y=sim.y, flutter_amp=amp,
             sigma_w=SIGMA_W_MOD, L_w=L_W, V=V_TEST)
    print(f"    saved turbulence_4a_{channel}.png")
    return sim


# ---------------------------------------------------------------------
# 4b
# ---------------------------------------------------------------------

def experiment_4b(synth, sim_4a, channel, fig_dir, res_dir):
    print(f"\n  4b [{channel}]: frequency-domain PSD at V={V_TEST}")
    poly_full_d = synth.poly_full_d
    poly_red_d  = synth.poly_red_d
    tp     = synth.tp
    domain = tp.domain if hasattr(tp, "domain") else np.array([[40.0, 61.0]])
    n_full = poly_full_d.n
    n_red  = poly_red_d.n
    V_grid = poly_red_d.V_grid
    dt     = synth.dt

    # blended matrices at V_TEST
    A_full = _blend_poly(poly_full_d.A, V_TEST, V_grid)
    B_full = _blend_poly(poly_full_d.B, V_TEST, V_grid)
    C_full = _blend_poly(poly_full_d.C, V_TEST, V_grid)
    A_red  = _blend_poly(poly_red_d.A,  V_TEST, V_grid)
    B_red  = _blend_poly(poly_red_d.B,  V_TEST, V_grid)
    C_red  = _blend_poly(poly_red_d.C,  V_TEST, V_grid)
    K = _blend_tp(synth.K, V_TEST, tp, domain)
    L = _blend_tp(synth.L, V_TEST, tp, domain)

    B_w = short_period_B_w(channel, n_full)
    fidx = FLUTTER_FULL_IDX[channel]

    BK = B_red @ K
    LC = L @ C_full
    LCA = L @ C_red @ A_red
    LCBK = L @ C_red @ BK
    A_cl = np.block([[A_full, -B_full @ K],
                     [LC,     A_red - BK - LCA + LCBK]])
    Bw_cl = np.zeros((n_full + n_red, 1))
    Bw_cl[:n_full, :] = B_w * dt

    C_xa = np.zeros((1, n_full + n_red)); C_xa[0, fidx[0]] = 1.0
    C_xb = np.zeros((1, n_full + n_red)); C_xb[0, fidx[1]] = 1.0
    C_xa_ol = np.zeros((1, n_full)); C_xa_ol[0, fidx[0]] = 1.0
    C_xb_ol = np.zeros((1, n_full)); C_xb_ol[0, fidx[1]] = 1.0

    n_freq = 600
    omega  = np.logspace(-1, np.log10(np.pi/dt) - 0.05, n_freq)
    freq_hz = omega / (2*np.pi)

    a = L_W / V_TEST
    Phi_w = (SIGMA_W_MOD**2) * (a/np.pi) * (
        (1 + 3*(a*omega)**2) / (1 + (a*omega)**2)**2)
    pilot_rng = np.random.default_rng(seed=12345)
    pilot_w = dryden_realization(SIGMA_W_MOD, L_W, V_TEST, dt, 20000, pilot_rng)
    var_anal = float(np.trapezoid(Phi_w, omega) / np.pi * 2.0)
    cal_psd = SIGMA_W_MOD**2 / var_anal if var_anal > 0 else 1.0
    Phi_w *= cal_psd
    print(f"    Dryden PSD cal factor: {cal_psd:.3f}")

    P_cl = np.zeros(n_freq); P_ol = np.zeros(n_freq)
    I_cl = np.eye(A_cl.shape[0]); I_ol = np.eye(n_full)
    for i, zk in enumerate(np.exp(1j * omega * dt)):
        try:
            H = np.linalg.solve(zk*I_cl - A_cl, Bw_cl)
            P_cl[i] = (abs((C_xa@H)[0,0])**2 + abs((C_xb@H)[0,0])**2) * Phi_w[i]
        except: P_cl[i] = np.nan
        try:
            Ho = np.linalg.solve(zk*I_ol - A_full, B_w*dt)
            P_ol[i] = (abs((C_xa_ol@Ho)[0,0])**2 + abs((C_xb_ol@Ho)[0,0])**2) * Phi_w[i]
        except: P_ol[i] = np.nan

    f_flutter = 7.78
    i_fl = int(np.argmin(np.abs(freq_hz - f_flutter)))
    atten_db = (10*np.log10(P_ol[i_fl]/P_cl[i_fl])
                if np.isfinite(P_cl[i_fl]) and P_cl[i_fl] > 0 else float("nan"))
    print(f"    flutter band attenuation at {f_flutter} Hz: {atten_db:.1f} dB")

    amp_cl = _flutter_amp(sim_4a.x_full, channel)
    f_emp, P_emp = sps.welch(amp_cl, fs=1.0/dt, nperseg=2048, noverlap=1024)

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True)
    ax = axes[0]
    ax.loglog(freq_hz, np.maximum(P_ol, 1e-30), color=COL_OPEN,   lw=1.6, label="open-loop (analytical)")
    ax.loglog(freq_hz, np.maximum(P_cl, 1e-30), color=COL_CLOSED, lw=1.6, label="closed-loop (analytical)")
    ax.axvline(f_flutter, color="k", lw=0.8, ls=":", label=f"flutter {f_flutter} Hz")
    ax.set_ylabel(r"$\Phi_{x_{\rm flutter}}$"); ax.set_title("(a) flutter-mode PSD  (analytical)")
    ax.legend(frameon=False, loc="lower left", fontsize=9); _style_ax(ax)

    ax = axes[1]
    ax.loglog(f_emp, np.maximum(P_emp, 1e-30), color=COL_CLOSED, lw=0.7, alpha=0.9, label="empirical (Welch)")
    ax.loglog(freq_hz, np.maximum(P_cl, 1e-30), color="k", lw=1.4, ls="--", label="analytical CL")
    ax.axvline(f_flutter, color="k", lw=0.8, ls=":")
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel(r"$\Phi$")
    ax.set_title("(b) empirical vs analytical CL PSD")
    ax.legend(frameon=False, loc="lower left", fontsize=9); _style_ax(ax)

    fig.suptitle(f"Experiment 4b [{channel}]: PSD under Dryden turbulence  "
                 f"(attenuation {atten_db:.0f} dB at flutter)", fontsize=11)
    fig.tight_layout(rect=(0,0,1,0.96))
    fig.savefig(fig_dir / f"turbulence_4b_{channel}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    np.savez(res_dir / f"turbulence_4b_{channel}.npz",
             freq_hz=freq_hz, P_cl=P_cl, P_ol=P_ol,
             f_emp=f_emp, P_emp=P_emp, atten_db=atten_db)
    print(f"    saved turbulence_4b_{channel}.png")


# ---------------------------------------------------------------------
# 4c
# ---------------------------------------------------------------------

def experiment_4c(synth, channel, fig_dir, res_dir):
    print(f"\n  4c [{channel}]: envelope sweep M={M_4C}")
    V_grid = synth.poly_red_d.V_grid
    n_V = len(V_grid)
    rms_amp = np.zeros((n_V, M_4C))
    rms_u   = np.zeros((n_V, M_4C))
    max_u   = np.zeros((n_V, M_4C))

    for i, V in enumerate(V_grid):
        rng_root = np.random.default_rng(seed=int(V*100))
        for m in range(M_4C):
            rng = np.random.default_rng(rng_root.integers(0, 2**32))
            gust = make_dryden_forcing(channel=channel, n_full=synth.poly_full_d.n,
                                       sigma_w=SIGMA_W_MOD, L_w=L_W, V=float(V),
                                       dt=synth.dt, T_final=T_RMS, rng=rng)
            sim = simulate_with_gust(synth, V=float(V), gust=gust,
                                      T_final=T_RMS, verbose=False)
            amp = _flutter_amp(sim.x_full, channel)
            rms_amp[i, m] = float(np.sqrt(np.mean(amp**2)))
            rms_u[i, m]   = float(np.sqrt(np.mean(sim.u**2)))
            max_u[i, m]   = float(np.max(np.abs(sim.u)))
        print(f"    V={V:5.2f}: RMS amp={rms_amp[i].mean():.2e}, "
              f"RMS u={rms_u[i].mean()*1e3:.2f} mrad")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    mu_a = rms_amp.mean(1); sd_a = rms_amp.std(1)
    ax.fill_between(V_grid, np.maximum(mu_a-sd_a,1e-12), mu_a+sd_a,
                    color=COL_CLOSED, alpha=COL_BAND_A, lw=0, label=r"$\pm1\sigma$")
    ax.plot(V_grid, mu_a, "o-", color=COL_CLOSED, lw=1.5, ms=5, label="mean")
    ax.axvline(50.2, color=COL_OPEN, lw=0.8, ls=":", label="flutter onset")
    ax.set_yscale("log"); ax.set_xlabel("V (m/s)"); ax.set_ylabel(r"RMS $\|x_{\rm flutter}\|$")
    ax.set_title("(a) RMS flutter vs V"); ax.legend(frameon=False, fontsize=9); _style_ax(ax)

    ax = axes[1]
    mu_u = rms_u.mean(1)*1e3; sd_u = rms_u.std(1)*1e3; mx_u = max_u.mean(1)*1e3
    ax.fill_between(V_grid, np.maximum(mu_u-sd_u,0), mu_u+sd_u,
                    color=COL_CLOSED, alpha=COL_BAND_A, lw=0)
    ax.plot(V_grid, mu_u, "o-", color=COL_CLOSED, lw=1.5, ms=5, label="mean RMS")
    ax.plot(V_grid, mx_u, "s--", color=COL_OPEN,   lw=1.0, ms=4, label="mean max")
    ax.axhline(349.0, color="r", lw=0.8, ls=":", label="saturation")
    ax.set_xlabel("V (m/s)"); ax.set_ylabel(r"$u$ (mrad)")
    ax.set_title("(b) control effort vs V"); ax.legend(frameon=False, fontsize=9); _style_ax(ax)

    fig.suptitle(f"Experiment 4c [{channel}]: envelope RMS under Dryden turbulence  "
                 f"(M={M_4C}, $\\sigma_w$={SIGMA_W_MOD} m/s)", fontsize=11)
    fig.tight_layout(rect=(0,0,1,0.94))
    fig.savefig(fig_dir / f"turbulence_4c_{channel}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    np.savez(res_dir / f"turbulence_4c_{channel}.npz",
             V=V_grid, rms_amp=rms_amp, rms_u=rms_u, max_u=max_u,
             sigma_w=SIGMA_W_MOD, L_w=L_W, M=M_4C)
    print(f"    saved turbulence_4c_{channel}.png")


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main():
    out_dir = Path("examples/lqg_flipased_demo")
    fig_dir = out_dir / "figures"
    res_dir = out_dir / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    for ch in CHANNELS:
        print("\n" + "=" * 70)
        print(f"Experiment 4: Dryden turbulence  (channel={ch}, hull={HULL})")
        print("=" * 70)
        synth = synthesize(channel=ch, hull=HULL, verbose=True)
        sim_4a = experiment_4a(synth, ch, fig_dir, res_dir)
        experiment_4b(synth, sim_4a, ch, fig_dir, res_dir)
        experiment_4c(synth, ch, fig_dir, res_dir)
        print(f"\n  all three sub-experiments complete for channel={ch}.")


if __name__ == "__main__":
    main()
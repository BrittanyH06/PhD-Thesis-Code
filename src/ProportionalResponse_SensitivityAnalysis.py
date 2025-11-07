#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# =============================================================================
# Purpose (user data required)
# =============================================================================
# Simulates antibiotic pulse dynamics, computes growth-based delays, RNA/P
# time-series (N=4), inhibition curves (via a cubic), and sensitivity/Hessian
# analyses for a 7-parameter mapping.
#
# You must provide your own experimental inputs (placeholders below):
#  - LAB_IC50_GLU, LAB_IC50_GLY (IC50 values from your experiments)
#  - exp_delay_times_glu/gly and std_glu/gly (delay vs N with standard deviations)
#  - lab_RNA_P_glu/gly and lab_RNA_P_glu_std/gly_std for N=4 (same time order)
#  - time_points_4H (hours)
#  - TET_concentrations, growth_glucose/glycerol, stdev_glucose/glycerol
#  - Optimization-based weights (if used in optimization runs)
#
# Formatting notes:
#  - exp_delay_times_*: dict mapping N (int/float) -> delay (float)
#  - std_*:              dict mapping N -> stdev > 0 (float)
#  - lab_RNA_P_*:        {4: [y1, y2, ...]}  (N=4 only; list of floats)
#  - lab_RNA_P_*_std:    same length/order as lab_RNA_P_*[4]
#  - time_points_4H:     list of floats (same length/order as the N=4 series)
#  - growth_* arrays:    numpy arrays normalized so growth[0] == 1 at TET=0
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from timeit import default_timer as timer
import matplotlib as mpl
from pathlib import Path

start = timer()
VERBOSE_MSE = True

# Editable fonts in vector outputs
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

def save_vector(fig, save_name, outdir="figures", formats=("svg", "pdf"), transparent=True):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(
            Path(outdir) / f"{save_name}.{ext}",
            bbox_inches="tight",
            transparent=transparent
        )

# ===========================
# Palette utilities
# ===========================
tol_vibrant = [
    "#EE7733",  # 0 orange
    "#0077BB",  # 1 blue
    "#003C5E",  # 2 blue-black
    "#33BBEE",  # 3 cyan
    "#009988",  # 4 teal
    "#004C44",  # 5 teal-black
    "#CC3311",  # 6 red
    "#661A08",  # 7 red-black
    "#EE3377",  # 8 pink
    "#BBBBBB",  # 9 grey (light)
    "#AA4499",  # 10 purple/magenta
    "#44AA99",  # 11 green-teal
    "#FEC44F",  # 12 yellow
    "#7FBC41",  # 13 lime-ish green
    "#8C510A",  # 14 brown
    "#AAAA00",  # 15 olive
]
def blend_colors(hex1, hex2, ratio=0.5):
    h1 = [int(hex1[i:i+2], 16) for i in (1, 3, 5)]
    h2 = [int(hex2[i:i+2], 16) for i in (1, 3, 5)]
    blended = [round((1 - ratio) * c1 + ratio * c2) for c1, c2 in zip(h1, h2)]
    return '#' + ''.join(f'{c:02X}' for c in blended)

mixed_color = blend_colors(tol_vibrant[3], tol_vibrant[8])  # cyan + pink
custom_palette = [
    tol_vibrant[0], tol_vibrant[1], tol_vibrant[2], tol_vibrant[3],
    tol_vibrant[4], tol_vibrant[5], tol_vibrant[6], tol_vibrant[7],
    tol_vibrant[8], tol_vibrant[9], tol_vibrant[10],tol_vibrant[11],
    tol_vibrant[12],tol_vibrant[13],tol_vibrant[14],tol_vibrant[15],
    mixed_color
]

# ===========================
# LaTeX-style (mathtext) labels
# ===========================
def latexify_param_list(param_list):
    mapping = {
        "K_D": r"$K_{D}$",
        "R_P_glu": r"$R_{P,\mathrm{glu}}$",
        "R_P_gly": r"$R_{P,\mathrm{gly}}$",
        "alpha0_glu": r"$\alpha_{0,\mathrm{glu}}$",
        "alphaA_glu": r"$\alpha_{A,\mathrm{glu}}$",
        "alpha0_gly": r"$\alpha_{0,\mathrm{gly}}$",
        "alphaA_gly": r"$\alpha_{A,\mathrm{gly}}$",
    }
    return [mapping.get(x, x) for x in param_list]

# =============================================================================
# USER INPUT REQUIRED — Laboratory data placeholders
# =============================================================================

# --- Delay time vs pulse length N (hours) ---
# Provide measured delay times (in hours) and their standard deviations
# Example format shown with dummy placeholders only
exp_delay_times_glu = {
    # e.g., 2: <delay_time>, 4: <delay_time>, ...
}
std_glu = {
    # e.g., 2: <std_value>, 4: <std_value>, ...
}

exp_delay_times_gly = {
    # e.g., 4: <delay_time>, 6: <delay_time>, ...
}
std_gly = {
    # e.g., 4: <std_value>, 6: <std_value>, ...
}

# --- RNA / Protein ratio time series for N=4 ---
# Lists must have the same length and order as time_points_4H
lab_RNA_P_glu = {
    4: [
        # e.g., <val>, <val>, ...
    ]
}
lab_RNA_P_glu_std = {
    4: [
        # e.g., <std_val>, <std_val>, ...
    ]
}

lab_RNA_P_gly = {
    4: [
        # e.g., <val>, <val>, ...
    ]
}
lab_RNA_P_gly_std = {
    4: [
        # e.g., <std_val>, <std_val>, ...
    ]
}

# Times at which RNA/P was sampled during a 4-hour pulse
time_points_4H = [
    # e.g., <t1>, <t2>, <t3>, ...
]

# --- Growth inhibition curves (normalized growth vs tetracycline concentration) ---
import numpy as np

TET_concentrations = np.array([
    # e.g., <conc1>, <conc2>, <conc3>, ...
], dtype=float)

growth_glucose = np.array([
    # normalized growth in glucose (same length as TET_concentrations)
], dtype=float)

stdev_glucose = np.array([
    # standard deviations for glucose (same length)
], dtype=float)

growth_glycerol = np.array([
    # normalized growth in glycerol (same length as TET_concentrations)
], dtype=float)

stdev_glycerol = np.array([
    # standard deviations for glycerol (same length)
], dtype=float)

# =============================================================================
# Base parameters — separate for glucose and glycerol
# NOTE: Fill the optimization-derived entries locally before running.
#       Keep these out of version control if they reflect lab-calibrated fits.
# =============================================================================
params_glu = {
    "r_min":   5.4,
    "r_max":   54.4,
    "kappa_t": 0.058,
    # ---- optimization-derived (FILL IN LOCALLY) ----
    "k_off":   None,   # FILL IN: off-rate (1/h)
    "k_on":    None,   # FILL IN: on-rate  (1/(µM·h))
    "P_in":    None,   # FILL IN: influx rate  (1/h)
    "P_out":   None,   # FILL IN: efflux rate  (1/h)
    "alpha0":  None,   # FILL IN: feedback parameter (post-pulse)
    "alphaA":  None,   # FILL IN: feedback parameter (during pulse)
    # ---- measured/assumed (safe to publish) ----
    "lambda_0": 0.64   # drug-free growth (h⁻¹)
}
params_gly = {
    "r_min":   5.4,
    "r_max":   54.4,
    "kappa_t": 0.058,
    # ---- optimization-derived (FILL IN LOCALLY) ----
    "k_off":   None,   # FILL IN: off-rate (1/h)
    "k_on":    None,   # FILL IN: on-rate  (1/(µM·h))
    "P_in":    None,   # FILL IN: influx rate  (1/h)
    "P_out":   None,   # FILL IN: efflux rate  (1/h)
    "alpha0":  None,   # FILL IN: feedback parameter (post-pulse)
    "alphaA":  None,   # FILL IN: feedback parameter (during pulse)
    # ---- measured/assumed (safe to publish) ----
    "lambda_0": 0.40   # drug-free growth (h⁻¹)
}
def update_derived_params(p):
    p = p.copy()
    p["delta_r"]    = p["r_max"] - p["r_min"]
    p["lambda_max"] = p["kappa_t"] * p["delta_r"]
    p["kappa_n"]    = 1.0 / (p["delta_r"] * ((1.0 / p["lambda_0"]) - (1.0 / (p["kappa_t"] * p["delta_r"]))))
    return p
params_glu = update_derived_params(params_glu)
params_gly = update_derived_params(params_gly)

# Globals for inhibition cubic (shared geometry)
r_min    = params_glu["r_min"]
r_max    = params_glu["r_max"]
kappa_t  = params_glu["kappa_t"]
delta_r  = r_max - r_min

# User-provided IC50 values (from your experiments)
LAB_IC50_GLU = <float>
LAB_IC50_GLY = <float>
    
# Literature/nominal Greulich IC50s (or set your own nominal values)
GREU_IC50_GLU = 1.75
GREU_IC50_GLY = 1.45

# =============================================================================
# ODE model (substrate-specific via params dict; NOTE: piecewise is kept)
# =============================================================================
def model_equations(t, y, p, N, substrate):
    a, r_u, r_b = y
    r_min, r_max   = p["r_min"], p["r_max"]
    lambda_0       = p["lambda_0"]
    delta_r        = p["delta_r"]
    kappa_t        = p["kappa_t"]
    lambda_max     = p["lambda_max"]
    kappa_n        = p["kappa_n"]
    alpha0, alphaA = p["alpha0"], p["alphaA"]

    t_on, t_off = 1.0, 1.0 + N

    # Lab prefactors (piecewise constants)
    if substrate == "glucose":
        lab_ic50, greu_ic50 = LAB_IC50_GLU, GREU_IC50_GLU
    else:
        lab_ic50, greu_ic50 = LAB_IC50_GLY, GREU_IC50_GLY

    pref = (4.0 * greu_ic50) / lab_ic50

    x_syn = (r_u - r_min) * kappa_t
    x_dil = (r_max - r_u - r_b) * kappa_n
    x     = min(x_syn, x_dil)  # NOTE: non-smooth kept

    s_ss0 = lambda_0 * (r_max - (lambda_0 * delta_r * ((1.0 / lambda_0) - (1.0 / lambda_max))))
    lam_f = lambda_0 / (1.0 + (pref / N))
    s_ssA = lam_f * (r_max - lam_f / kappa_n)

    if t < t_on:
        a_ex = 0.0; s = s_ss0
    elif t <= t_off:
        a_ex = (pref * lab_ic50) / N
        s    = s_ssA * (1.0 + alphaA * (x_dil - x_syn))
    else:
        a_ex = 0.0
        s    = s_ss0 * (1.0 + alpha0 * (x_dil - x_syn))

    F       = p["k_on"] * a * (r_u - r_min) - (p["k_off"] * r_b)
    da_dt   = -F - x * a + p["P_in"] * a_ex - p["P_out"] * a
    dr_u_dt = -F - x * r_u + s
    dr_b_dt =  F - x * r_b
    return [da_dt, dr_u_dt, dr_b_dt]

def simulate_once(p, N, substrate, rtol=5e-9, atol=1e-11):
    y0 = [0.0, p["r_min"] + (p["lambda_0"] / p["kappa_t"]), 0.0]
    tMax  = (N + 1) + 4 + (30.0 / p["lambda_0"])
    t_eval = np.linspace(0.0, tMax, int(200 * tMax))
    sol = solve_ivp(lambda t, y: model_equations(t, y, p, N, substrate),
                    [t_eval[0], t_eval[-1]], y0, method="BDF",
                    t_eval=t_eval, rtol=rtol, atol=atol)
    time = sol.t
    ru, rb = sol.y[1], sol.y[2]
    gr1 = (ru - p["r_min"]) * p["kappa_t"]
    gr2 = (p["r_max"] - ru - rb) * p["kappa_n"]
    gr  = np.minimum(gr1, gr2)
    return time, ru, rb, gr

def estimate_delay_from_gr(time, gr, lambda0):
    dt = time[1] - time[0]
    OD = np.cumsum(gr) * dt
    pre, post = OD[1], OD[-1]
    return ((pre - post) / lambda0) + (time[-1] - time[1])

def predict_delays_for_set(p, N_list, substrate):
    out = {}
    for N in N_list:
        time, ru, rb, gr = simulate_once(p, N, substrate)
        out[N] = estimate_delay_from_gr(time, gr, p["lambda_0"])
    return out

def predict_rnap_timeseries_N4(p, sample_times, substrate):
    N = 4
    time, ru, rb, gr = simulate_once(p, N, substrate)
    rtot = ru + rb
    t0 = 0.5
    idx0 = np.argmin(np.abs(time - t0))
    base = rtot[idx0] if rtot[idx0] != 0 else 1.0
    rnorm = rtot / base
    sample_times = np.asarray(sample_times, dtype=float)
    return np.interp(sample_times, time, rnorm)

# =============================================================================
# Weighted MSEs
# =============================================================================
def weighted_mse(pred, obs, std):
    pred, obs, std = np.asarray(pred, float), np.asarray(obs, float), np.asarray(std, float)
    if np.any(std <= 0):
        pos = std[std > 0]
        fill = np.median(pos) if pos.size > 0 else 1.0
        std[std <= 0] = fill
    w = 1.0 / (std**2)
    return np.mean(w * (pred - obs)**2)

def mse_delay(p, substrate):
    if substrate == "glucose":
        N_lab, obs_d, std_d = sorted(exp_delay_times_glu.keys()), exp_delay_times_glu, std_glu
    else:
        N_lab, obs_d, std_d = sorted(exp_delay_times_gly.keys()), exp_delay_times_gly, std_gly
    pred = predict_delays_for_set(p, N_lab, substrate)
    pred_v = np.array([pred[N] for N in N_lab], dtype=float)
    obs_v  = np.array([obs_d[N] for N in N_lab], dtype=float)
    std_v  = np.array([std_d[N] for N in N_lab], dtype=float)
    return weighted_mse(pred_v, obs_v, std_v)

def mse_rnap(p, substrate):
    if substrate == "glucose":
        obs = np.asarray(lab_RNA_P_glu[4], dtype=float)
        std = np.asarray(lab_RNA_P_glu_std[4], dtype=float)
    else:
        obs = np.asarray(lab_RNA_P_gly[4], dtype=float)
        std = np.asarray(lab_RNA_P_gly_std[4], dtype=float)
    pred = predict_rnap_timeseries_N4(p, time_points_4H, substrate)
    return weighted_mse(pred, obs, std)

# =============================================================================
# Inhibition curves via cubic (Cardano) — normalized growth x = λ/λ0
# =============================================================================
def solve_lambda_cubic(a_ex, lambda_0, k_off, k_on, P_in, P_out):
    if abs(a_ex) < 1e-12:
        return 1.0
    lambda0_star = 2.0 * np.sqrt(P_out * kappa_t * (k_off / k_on))
    IC50_star    = (delta_r * lambda0_star) / (2.0 * P_in)
    a = 0.25 * (lambda0_star / lambda_0)**2
    b = (a_ex / (2.0 * IC50_star)) * (lambda0_star / lambda_0)
    # x^3 - x^2 + (a+b)x - a = 0
    A = -1.0; B = a + b; C = -a
    shift = 1.0/3.0
    p = B - (A**2)/3.0
    q = (2.0*(A**3))/27.0 - (A*B)/3.0 + C
    Delta = (q/2.0)**2 + (p/3.0)**3
    def cbrt(z): return np.sign(z) * abs(z)**(1.0/3.0)
    if Delta >= 0:
        y = cbrt(-q/2.0 + np.sqrt(Delta)) + cbrt(-q/2.0 - np.sqrt(Delta))
        x = y + shift
    else:
        r_val = 2.0 * np.sqrt(-p/3.0)
        theta = np.arccos(-q / (2.0 * np.sqrt((-p/3.0)**3)))
        y1 = r_val * np.cos(theta/3.0)
        y2 = r_val * np.cos((theta + 2*np.pi)/3.0)
        y3 = r_val * np.cos((theta + 4*np.pi)/3.0)
        roots = [y1 + shift, y2 + shift, y3 + shift]
        roots = [r for r in roots if r > 0]
        x = max(roots) if roots else np.nan
    return x

def simulate_inhibition_curve_cubic(full_params, substrate, normalize=True):
    k_off, k_on, P_in, P_out, lambda_0 = full_params
    predicted = []
    for TET in TET_concentrations:
        x = solve_lambda_cubic(TET, lambda_0, k_off, k_on, P_in, P_out)
        if x is None or x < 0:
            predicted.append(0.0)
        else:
            predicted.append(x if normalize else x * lambda_0)
    return np.array(predicted)

def inhib_curve_mse(full_params, substrate):
    pred = simulate_inhibition_curve_cubic(full_params, substrate, normalize=True)
    if substrate == "glucose":
        obs, stdev_arr = growth_glucose, stdev_glucose
    else:
        obs, stdev_arr = growth_glycerol, stdev_glycerol
    return np.mean(((obs - pred) / stdev_arr)**2)

# =============================================================================
# θ mapping (7D) and helpers
# =============================================================================
PARAM_ALL   = ["K_D", "R_P_glu", "R_P_gly", "alpha0_glu", "alphaA_glu", "alpha0_gly", "alphaA_gly"]
PARAM_GLU   = ["K_D", "R_P_glu", "alpha0_glu", "alphaA_glu"]   # kept for reference
PARAM_GLY   = ["K_D", "R_P_gly", "alpha0_gly", "alphaA_gly"]   # kept for reference

THETA_TICK_LABELS = latexify_param_list(PARAM_ALL)
THETA_LABELS = PARAM_ALL[:]  # plain-text versions

KON_REF   = params_glu["k_on"]   # shared
POUT_GLU  = params_glu["P_out"]
POUT_GLY  = params_gly["P_out"]

def params_from_theta_sub(theta, base_glu, base_gly, substrate):
    K_D, RP_g, RP_y, a0g, aAg, a0y, aAy = theta
    if substrate == "glucose":
        p = base_glu.copy()
        p["k_on"]  = KON_REF
        p["k_off"] = K_D * KON_REF
        p["P_out"] = POUT_GLU
        p["P_in"]  = RP_g * POUT_GLU
        p["alpha0"] = a0g
        p["alphaA"] = aAg
    else:
        p = base_gly.copy()
        p["k_on"]  = KON_REF
        p["k_off"] = K_D * KON_REF
        p["P_out"] = POUT_GLY
        p["P_in"]  = RP_y * POUT_GLY
        p["alpha0"] = a0y
        p["alphaA"] = aAy
    return update_derived_params(p)

def inhib_params_from_theta(theta, substrate):
    K_D, RP_g, RP_y, *_ = theta
    k_on  = KON_REF
    k_off = K_D * k_on
    if substrate == "glucose":
        P_out = POUT_GLU
        P_in  = RP_g * P_out
        lam0  = params_glu["lambda_0"]
    else:
        P_out = POUT_GLY
        P_in  = RP_y * P_out
        lam0  = params_gly["lambda_0"]
    return np.array([k_off, k_on, P_in, P_out, lam0], dtype=float)

def theta_from_params_both(glu, gly):
    return np.array([
        glu["k_off"]/glu["k_on"],           # K_D (shared)
        glu["P_in"]/glu["P_out"],           # R_P_glu
        gly["P_in"]/gly["P_out"],           # R_P_gly
        glu["alpha0"], glu["alphaA"],       # alphas glu
        gly["alpha0"], gly["alphaA"]        # alphas gly
    ], dtype=float)

theta0 = theta_from_params_both(params_glu, params_gly)

# =============================================================================
# USER: Weights (your values from optimization). Set before running.
# =============================================================================
W_DELAY_GLU = None
W_RNAP_GLU = None
W_DELAY_GLY = None
W_RNAP_GLY = None
W_INHIB = None

# =============================================================================
# Objectives and residuals
# =============================================================================
def objective_substrate(theta, substrate):
    if substrate == "glucose":
        p  = params_from_theta_sub(theta, params_glu, params_gly, "glucose")
        fp = inhib_params_from_theta(theta, "glucose")
        return (W_DELAY_GLU * mse_delay(p, "glucose")
              + W_RNAP_GLU  * mse_rnap(p, "glucose")
              + W_INHIB     * inhib_curve_mse(fp, "glucose"))
    else:
        p  = params_from_theta_sub(theta, params_glu, params_gly, "glycerol")
        fp = inhib_params_from_theta(theta, "glycerol")
        return (W_DELAY_GLY * mse_delay(p, "glycerol")
              + W_RNAP_GLY  * mse_rnap(p, "glycerol")
              + W_INHIB     * inhib_curve_mse(fp, "glycerol"))

def objective_total(theta):
    return objective_substrate(theta, "glucose") + objective_substrate(theta, "glycerol")

# =============================================================================
# Baseline prints
# =============================================================================
if VERBOSE_MSE:
    print("\n=== BASELINE COSTS (at theta0) ===")
    print(f"Glucose component:  {objective_substrate(theta0, 'glucose'):.6f}")
    print(f"Glycerol component: {objective_substrate(theta0, 'glycerol'):.6f}")
    print(f"TOTAL:              {objective_total(theta0):.6f}")

# =============================================================================
# Sensitivities — TOTAL COST ONLY (unitless log–log)
# =============================================================================
perturbation_values = [0.1]  # percent

def central_log_elasticity(f, theta, i, eps):
    """
    S_i = ∂ ln f / ∂ ln θ_i via central multiplicative step ±eps (unitless).
    """
    th = np.asarray(theta, float).copy()
    h  = eps * (th[i] if th[i] != 0 else 1.0)
    e  = np.zeros_like(th); e[i] = 1.0
    f_p = max(f(th + h*e), 1e-300)
    f_m = max(f(th - h*e), 1e-300)
    return (np.log(f_p) - np.log(f_m)) / (2.0 * np.log1p(eps))

def total_cost_sensitivities(theta, dp_percent=0.1):
    eps = dp_percent / 100.0
    rows = []
    for i, name in enumerate(PARAM_ALL):
        S = central_log_elasticity(objective_total, theta, i, eps)
        rows.append({"Parameter": name, "DeltaPercent": dp_percent, "Sensitivity": float(S)})
    df = pd.DataFrame(rows)
    df["Parameter"] = pd.Categorical(df["Parameter"], categories=PARAM_ALL, ordered=True)
    return df.sort_values("Parameter")

def print_summary(df, title):
    print(f"\n\n================== {title} ==================")
    colw = 16
    hdr = (f"{'Parameter'.ljust(colw)}|"
           f"{'Δ(%)'.rjust(6)}|"
           f"{'Sensitivity'.rjust(14)}")
    print(hdr)
    print("-" * (len(hdr) + 8))
    for _, r in df.iterrows():
        print(f"{str(r['Parameter']).ljust(colw)}|"
              f"{str(r['DeltaPercent']).rjust(6)}|"
              f"{r['Sensitivity']:>14.6f}")

# =============================================================================
# Sensitivities — PULSE-ONLY RNA/P MSE (unitless log–log), combined (Glucose vs Glycerol)
# Times used: 1.75, 2.5, 3.25, 4.0, 4.75
# =============================================================================
PULSE_TIMES = np.array([1.75, 2.5, 3.25, 4.0, 4.75], dtype=float)

def _pulse_indices_in(times_all, pulse_times=PULSE_TIMES):
    all_arr = np.asarray(times_all, float)
    return [int(np.argmin(np.abs(all_arr - t))) for t in pulse_times]

def mse_rnap_pulse_only(p, substrate):
    """Weighted MSE between model RNA/P (N=4) and lab data, ONLY at pulse times."""
    if substrate == "glucose":
        obs_full = np.asarray(lab_RNA_P_glu[4], dtype=float)
        std_full = np.asarray(lab_RNA_P_glu_std[4], dtype=float)
    else:
        obs_full = np.asarray(lab_RNA_P_gly[4], dtype=float)
        std_full = np.asarray(lab_RNA_P_gly_std[4], dtype=float)
    idxs = _pulse_indices_in(time_points_4H, PULSE_TIMES)
    obs = obs_full[idxs]
    std = std_full[idxs]
    pred = predict_rnap_timeseries_N4(p, PULSE_TIMES, substrate)
    return weighted_mse(pred, obs, std)

def objective_rnap_pulse(theta, substrate):
    p = params_from_theta_sub(theta, params_glu, params_gly, substrate)
    return mse_rnap_pulse_only(p, substrate)

def pulse_mse_sensitivities(theta, substrate, dp_percent=0.1):
    """S_i = ∂ln(MSE_pulse)/∂ln θ_i for params that affect a given substrate."""
    eps = dp_percent / 100.0
    if substrate == "glucose":
        names = PARAM_GLU
        idxs  = [PARAM_ALL.index(nm) for nm in PARAM_GLU]
    else:
        names = PARAM_GLY
        idxs  = [PARAM_ALL.index(nm) for nm in PARAM_GLY]
    f = (lambda th: objective_rnap_pulse(th, substrate))
    rows = []
    for i, nm in zip(idxs, names):
        S = central_log_elasticity(f, theta, i, eps)   # unitless log–log
        rows.append({"Parameter": nm, "DeltaPercent": dp_percent, "Sensitivity": float(S)})
    df = pd.DataFrame(rows)
    df["Parameter"] = pd.Categorical(df["Parameter"], categories=names, ordered=True)
    return df.sort_values("Parameter")

def plot_pulse_sensitivities_combined(
    df_glu, df_gly, *,
    save_name="August_pulse_only_sens_combined",
    figsize=(10.8, 6.6),
    color_glu="#303030",   # dark neutral gray (distinct from eigenvector #BBBBBB)
    color_gly="#6A6A6A",   # mid gray; clearly distinct from glucose & eigenvector bars
    legend_loc="best",
    legend_outside=False
):
    # Map substrate-specific names to common slots
    map_glu = {"K_D":"K_D", "R_P_glu":"R_P", "alpha0_glu":"alpha0", "alphaA_glu":"alphaA"}
    map_gly = {"K_D":"K_D", "R_P_gly":"R_P", "alpha0_gly":"alpha0", "alphaA_gly":"alphaA"}
    slots   = ["K_D","R_P","alpha0","alphaA"]
    tick_labels = [r"$K_{D}$", r"$R_{P}$", r"$\alpha_{0}$", r"$\alpha_{A}$"]

    dp = float(df_glu["DeltaPercent"].iloc[0])

    sens_glu = {map_glu[r["Parameter"]]: float(r["Sensitivity"]) for _, r in df_glu.iterrows()}
    sens_gly = {map_gly[r["Parameter"]]: float(r["Sensitivity"]) for _, r in df_gly.iterrows()}
    y_glu = [sens_glu.get(s, np.nan) for s in slots]
    y_gly = [sens_gly.get(s, np.nan) for s in slots]

    x = np.arange(len(slots), dtype=float)
    width = 0.38

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width/2, y_glu, width=width, color=color_glu, edgecolor="black", label="Glucose")
    ax.bar(x + width/2, y_gly, width=width, color=color_gly, edgecolor="black", label="Glycerol")

    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=16)
    ax.set_ylabel(f"{dp:.1f}% Perturbation Sensitivity (Pulse-only RNA/P MSE)", fontsize=14)
    ax.set_title("Pulse-only RNA/P (N=4) Sensitivities — Glucose vs Glycerol", fontsize=16, pad=8)

    m = np.nanmax(np.abs(np.r_[y_glu, y_gly])) if len(y_glu)+len(y_gly) else 1.0
    ax.set_ylim(-max(1.2*m, 0.6), max(1.2*m, 0.6))

    if legend_outside:
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=12, frameon=True)
        plt.tight_layout(rect=(0, 0, 0.82, 1))
    else:
        ax.legend(loc=legend_loc, fontsize=12, frameon=True)
        plt.tight_layout()

    if save_name:
        save_vector(fig, save_name)
    plt.show()
    plt.close(fig)

# ---- compute, print, and plot ----
df_glu_pulse = pulse_mse_sensitivities(theta0, "glucose",  dp_percent=0.1)
df_gly_pulse = pulse_mse_sensitivities(theta0, "glycerol", dp_percent=0.1)

print_summary(df_glu_pulse, "SENSITIVITIES — Pulse-only RNA/P MSE (Glucose)")
print_summary(df_gly_pulse, "SENSITIVITIES — Pulse-only RNA/P MSE (Glycerol)")

plot_pulse_sensitivities_combined(df_glu_pulse, df_gly_pulse,
                                  save_name=Proportional_RNAP_MSE_sensitivity, # FILL IN AS WHATEVER MEETS YOUR NEEDS
                                  figsize=(11.2, 6.8),
                                  color_glu=custom_palette[2], color_gly=custom_palette[7],
                                  legend_loc="best", legend_outside=False)

# =============================================================================
# Unitless Hessian (log–log): H_ij = ∂^2 ln f / ∂ ln θ_i ∂ ln θ_j
# =============================================================================
LOG_H = 7e-5  # step in log-parameter space

def hessian_loglog_fd(eval_cost, theta, h=LOG_H, floor=1e-12):
    """
    Unitless Hessian: H_ij = ∂^2 ln f(θ) / ∂ ln θ_i ∂ ln θ_j
    eval_cost(theta) -> scalar cost f(θ) > 0
    """
    theta = np.asarray(theta, float)
    phi = np.log(np.maximum(theta, floor))
    n = len(theta)

    def g(phi_vec):
        f = float(eval_cost(np.exp(phi_vec)))
        return np.log(max(f, 1e-300))  # safe if f is tiny

    g0 = g(phi)
    H = np.empty((n, n), float)
    E = np.eye(n)

    # diagonal
    for i in range(n):
        gp = g(phi + h*E[i])
        gm = g(phi - h*E[i])
        H[i, i] = (gp - 2.0*g0 + gm) / (h*h)

    # mixed partials
    for i in range(n):
        for j in range(i+1, n):
            g_pp = g(phi + h*E[i] + h*E[j])
            g_pm = g(phi + h*E[i] - h*E[j])
            g_mp = g(phi - h*E[i] + h*E[j])
            g_mm = g(phi - h*E[i] - h*E[j])
            Hij = (g_pp - g_pm - g_mp + g_mm) / (4.0*h*h)
            H[i, j] = H[j, i] = Hij

    return 0.5 * (H + H.T)

def pretty_eigvals(H, name="Hessian"):
    Hs = 0.5 * (H + H.T)
    w = np.linalg.eigvalsh(Hs).astype(float)
    print(f"\n--- Eigenvalues for {name} (sorted) ---")
    for i, val in enumerate(w, 1):
        print(f"  λ_{i:<2d} = {val: .6e}")
    mn, mx = np.min(w), np.max(w)
    if mn < -1e-10:
        print("  Note: significant negative eigenvalue -> not PSD at this θ")
    elif mn < 0:
        print("  Note: tiny negative likely finite-difference/roundoff.")

print("\n=== Hessian Analysis of TOTAL COST (UNITLESS log–log) ===")
H_fd = hessian_loglog_fd(objective_total, theta0, h=LOG_H)
print("FD Hessian approximately symmetric:", np.allclose(H_fd, H_fd.T, atol=1e-7))
print("Finite-difference Hessian (unitless log–log):\n", H_fd)
pretty_eigvals(H_fd, name="Full Total-Cost Hessian (log–log FD)")

# =============================================================================
# Eigenvectors (θ-space histograms)
# =============================================================================
def eigenpairs_theta_from_Hfd(H_fd, orient=True):
    Hs = 0.5 * (H_fd + H_fd.T)
    w, V = np.linalg.eigh(Hs)
    order = np.argsort(w)     # sloppy → stiff
    w = w[order]; V = V[:, order]
    V = V / (np.linalg.norm(V, axis=0, keepdims=True) + 1e-18)
    if orient:
        for k in range(V.shape[1]):
            i_max = int(np.argmax(np.abs(V[:, k])))
            if V[i_max, k] < 0:
                V[:, k] *= -1.0
    return w, V

def _set_symmetric_ylim(ax, vec, *, pad_frac=0.10, min_span=0.40):
    m = float(np.max(np.abs(vec))) if np.size(vec) and np.isfinite(vec).any() else 0.0
    span = max(m * (1.0 + pad_frac), min_span / 2.0)
    ax.set_ylim(-span, span)

def plot_all_eigvec_histograms_theta(
    H_fd,
    *,
    bar_width=0.9,
    tick_fs=15, label_fs=17, title_fs=18,
    figsize=(10.5, 5.8),
    title_y=0.94,
    tight_rect=(0, 0, 1, 0.94),
    color=tol_vibrant[9],   # light gray bars (different from total-sens gray)
    save_prefix=None
):
    w, V = eigenpairs_theta_from_Hfd(H_fd)
    nvec = V.shape[1]
    x = np.arange(len(PARAM_ALL))  # 7 bars

    for k in range(nvec):
        v = V[:, k]
        lam = w[k]
        fig = plt.figure(figsize=figsize)
        tag = " — SLOPPY" if k == 0 else (" — STIFF" if k == nvec - 1 else "")
        fig.suptitle(
            f"Eigenvector #{k+1} (for $\\lambda_{{{k+1}}}={lam:.2e}$){tag}",
            fontsize=title_fs, y=title_y
        )
        ax = fig.add_subplot(1, 1, 1)
        ax.bar(x, v, width=bar_width, color=color, edgecolor="black", align="center")
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(THETA_TICK_LABELS, rotation=15, fontsize=tick_fs)
        ax.set_ylabel("Eigen-direction components\n($L^2$-normalized)", fontsize=label_fs, labelpad=10)
        _set_symmetric_ylim(ax, v, pad_frac=0.10, min_span=0.40)
        ax.tick_params(axis="y", labelsize=tick_fs)
        plt.tight_layout(rect=tight_rect)
        if save_prefix:
            save_vector(fig, save_prefix + f"_eigvec{k+1}")
        plt.show()
        plt.close(fig)

plot_all_eigvec_histograms_theta(H_fd, color=tol_vibrant[9], save_prefix="Proportional_Hessian")

# ===========================
# Print the 7D eigenvector components (θ-space)
# ===========================
def print_eigenvector_components_theta(
    H_fd,
    labels=THETA_LABELS,
    *,
    which="all",
    decimals=6,
    normalize=True
):
    w, V = eigenpairs_theta_from_Hfd(H_fd)
    n = V.shape[1]
    if which == "all":
        idxs = list(range(n))
    elif which == "sloppy_stiff":
        idxs = [0, n-1]
    else:
        idxs = list(which)

    for k in idxs:
        v = V[:, k]
        if normalize:
            v = v / (np.linalg.norm(v) + 1e-18)
        print(f"\nEigenvector #{k+1}  (λ_{k+1} = {w[k]:.{decimals-2}e})")
        print("-" * 46)
        for name, val in zip(labels, v):
            print(f"{name:<14s} : {val:+.{decimals}f}")

print_eigenvector_components_theta(H_fd, which="all", decimals=6)

# ==== Coupling utilities (on unitless H) ====
import itertools

def _safe_inv(A, ridge=0.0):
    Ause = A if ridge <= 0 else (A + ridge * np.eye(A.shape[0]))
    try:
        return np.linalg.inv(Ause)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(Ause, rcond=1e-12)

def _Hnorm_and_Corr(H_fd, ridge=0.0):
    """
    Normalized Hessian (curvature coupling) and correlation from Σ≈(Hs+ridge I)^(-1).
    """
    Hs = 0.5 * (H_fd + H_fd.T)
    d = np.sqrt(np.clip(np.diag(Hs), 1e-18, np.inf))
    Hnorm = Hs / np.outer(d, d)
    Sigma = _safe_inv(Hs, ridge=ridge)
    s = np.sqrt(np.clip(np.diag(Sigma), 1e-18, np.inf))
    Corr = Sigma / np.outer(s, s)
    return Hnorm, Corr, Hs

def coupling_table(H_fd, labels, ridge=0.0):
    Hnorm, Corr, _ = _Hnorm_and_Corr(H_fd, ridge=ridge)
    n = len(labels)
    rows = []
    for i, j in itertools.combinations(range(n), 2):
        hij = float(Hnorm[i, j]); sij = float(Corr[i, j])
        rows.append({
            "i": i, "j": j, "pi": labels[i], "pj": labels[j],
            "rho_H": hij, "rho_S": sij, "absH": abs(hij), "absS": abs(sij),
            "score": max(abs(hij), abs(sij))
        })
    return pd.DataFrame(rows).sort_values("score", ascending=False, ignore_index=True)

def coupling_reports(H_fd, labels, top_k=8, thresh=0.60, ridge=0.0):
    Hnorm, Corr, _ = _Hnorm_and_Corr(H_fd, ridge=ridge)
    n = len(labels)
    def top_pairs(M):
        out = []
        for i in range(n):
            for j in range(i+1, n):
                out.append((abs(M[i,j]), M[i,j], i, j))
        out.sort(reverse=True, key=lambda t: t[0])
        return out
    def pretty_block(title, items):
        print(f"\n--- {title} ---")
        printed = 0
        for mag, val, i, j in items:
            flag = "  **strong**" if abs(val) >= thresh else ""
            print(f" {labels[i]:<15s} ↔ {labels[j]:<15s} : C({i+1},{j+1})={val:+.3f}{flag}")
            printed += 1
            if printed >= top_k and abs(val) < thresh:
                break
    pretty_block("Curvature couplings (normalized Hessian)", top_pairs(Hnorm))
    # pretty_block("Posterior correlations (Σ ≈ H^{-1})", top_pairs(Corr))  # optional

# ==== Contour plots for coupled pairs (unitless H; pixel-safe labels) ====
def plot_subspace_contour(
    H_fd, i, j, labels,
    *,
    title=None,
    ngrid=201, span_factor=3.0, cmap="Spectral_r",
    figsize=(6.5, 6.0), save_name=None, ridge=0.0,
    eig_mode="subspace",     # "subspace" | "global_minmax" | "global_best2"
    arrow_length_frac=0.60,
    equal_cost=True,
    stiff_eta=0.50,
    min_arrow_frac=0.35,
    max_arrow_frac=0.95,
    annotate=True,
    show_coupling_box=True
):
    """
    2D cost contours from the true 2×2 sub-Hessian in (θ_i, θ_j).
    Arrows: subspace (default) or projected global eigendirections.
    Labels: placed near arrow tips; overlap is checked in DISPLAY PIXELS against both
            arrow segments, then iteratively adjusted until clear and inside the axes.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    try:
        latex_labels = latexify_param_list(labels)
    except Exception:
        latex_labels = labels
    lab_i, lab_j = latex_labels[i], latex_labels[j]

    H2 = np.array([[H_fd[i, i] + ridge, H_fd[i, j]],
                   [H_fd[i, j],         H_fd[j, j] + ridge]], float)

    _, _, Hs = _Hnorm_and_Corr(H_fd, ridge=ridge)

    w2, V2 = np.linalg.eigh(H2)
    lam_s_sub, lam_t_sub = float(w2[0]), float(w2[-1])
    v_s_sub,  v_t_sub    = V2[:, 0], V2[:, -1]

    lam_min_vis = max(min(w2), 1e-18)
    R = span_factor / np.sqrt(lam_min_vis)
    x = np.linspace(-R, R, ngrid); y = np.linspace(-R, R, ngrid)
    X, Y = np.meshgrid(x, y)
    Z = 0.5 * (H2[0, 0]*X*X + 2*H2[0, 1]*X*Y + H2[1, 1]*Y*Y)

    zmax = float(np.nanquantile(Z, 0.98))
    levels = np.linspace(zmax/14, zmax, 16)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.imshow(Z, extent=[x.min(), x.max(), y.min(), y.max()],
              origin="lower", cmap=cmap, alpha=0.55, interpolation="bilinear")
    ax.contour(X, Y, Z, levels=levels, colors="k", linewidths=0.9, alpha=0.75)

    ax.set_xlim(x.min(), x.max()); ax.set_ylim(y.min(), y.max())
    ax.margins(x=0.08, y=0.08)

    if eig_mode == "subspace":
        lam_s, lam_t = lam_s_sub, lam_t_sub
        v_s2, v_t2   = v_s_sub.copy(), v_t_sub.copy()
        # superscript only (no global index in subspace mode)
        label_s = rf"$\lambda^{{\mathrm{{sloppy}}}}={lam_s:.2e}$"
        label_t = rf"$\lambda^{{\mathrm{{stiff}}}}={lam_t:.2e}$"
    else:
        W, V = np.linalg.eigh(0.5 * (H_fd + H_fd.T))
        order = np.argsort(W); W, V = W[order], V[:, order]
        def inplane_energy(k): 
            vv = V[[i, j], k]; return float(vv @ vv)
        if eig_mode == "global_minmax":
            idxs = [0, len(W)-1]
        elif eig_mode == "global_best2":
            es = sorted(((inplane_energy(k), k) for k in range(len(W))), reverse=True)
            idxs = sorted([es[0][1], es[1][1]], key=lambda k: W[k])
        else:
            raise ValueError("eig_mode must be 'subspace', 'global_minmax', or 'global_best2'")
        k_s, k_t = idxs
        lam_s, lam_t = float(W[k_s]), float(W[k_t])
        v_s2, v_t2   = V[[i, j], k_s], V[[i, j], k_t]
        if np.linalg.norm(v_s2) < 1e-14: v_s2, lam_s = v_s_sub, lam_s_sub
        if np.linalg.norm(v_t2) < 1e-14: v_t2, lam_t = v_t_sub, lam_t_sub
        # add global index as subscript and sloppy/stiff as superscript
        label_s = rf"$\lambda_{{{k_s+1}}}^{{\mathrm{{sloppy}}}}={lam_s:.2e}$"
        label_t = rf"$\lambda_{{{k_t+1}}}^{{\mathrm{{stiff}}}}={lam_t:.2e}$"


    Ls = arrow_length_frac * R
    if equal_cost:
        ratio = max(lam_s, 1e-300) / max(lam_t, 1e-300)
        Lt = Ls * (ratio ** stiff_eta)
    else:
        Lt = Ls
    Lt = max(Ls * min_arrow_frac, min(Lt, Ls * max_arrow_frac))

    if v_s2[0] < 0: v_s2 = -v_s2
    if v_t2[0] < 0: v_t2 = -v_t2
    vsu = v_s2 / (np.linalg.norm(v_s2) + 1e-18)
    vtu = v_t2 / (np.linalg.norm(v_t2) + 1e-18)

    xs, ys = Ls * vsu[0], Ls * vsu[1]
    xt, yt = Lt * vtu[0], Lt * vtu[1]

    head_w = 0.04 * R; head_l = 0.06 * R
    ax.arrow(0, 0, xs, ys, head_width=head_w, head_length=head_l,
             fc="dimgray", ec="dimgray", lw=2.8, length_includes_head=True, alpha=0.95, zorder=2)
    ax.arrow(0, 0, xt, yt, head_width=head_w, head_length=head_l,
             fc="k", ec="k", lw=3.2, length_includes_head=True, alpha=0.95, zorder=2)

    # ================= renderer-aware, pixel-space overlap avoidance =================
    to_disp = ax.transData.transform
    A1, B1 = to_disp([0, 0]), to_disp([xs, ys])
    A2, B2 = to_disp([0, 0]), to_disp([xt, yt])

    INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8
    def _code(x, y, x0, y0, x1, y1):
        c = INSIDE
        if x < x0: c |= LEFT
        elif x > x1: c |= RIGHT
        if y < y0: c |= BOTTOM
        elif y > y1: c |= TOP
        return c

    def _seg_intersects_rect(p0, p1, rect):
        x0, y0, x1, y1 = rect
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        xA, yA = p0; xB, yB = p1
        cA = _code(xA, yA, x0, y0, x1, y1)
        cB = _code(xB, yB, x0, y0, x1, y1)
        while True:
            if not (cA | cB):
                return True
            if cA & cB:
                return False
            c = cA or cB
            if c & TOP:
                x = xA + (xB - xA) * (y1 - yA) / (yB - yA); y = y1
            elif c & BOTTOM:
                x = xA + (xB - xA) * (y0 - yA) / (yB - yA); y = y0
            elif c & RIGHT:
                y = yA + (yB - yA) * (x1 - xA) / (xB - xA); x = x1
            else:
                y = yA + (yB - yA) * (x0 - xA) / (xB - xA); x = x0
            if c == cA:
                xA, yA = x, y; cA = _code(xA, yA, x0, y0, x1, y1)
            else:
                xB, yB = x, y; cB = _code(xB, yB, x0, y0, x1, y1)

    def _place_label_near_tip_pixel_safe(ax, tip_xy, u_unit, text, color,
                                         back0=0.30, side0=0.20, px_clear=24,
                                         max_back=0.70, max_side=0.60, fs=12):
        fig = ax.figure
        L = max(np.linalg.norm(tip_xy), 1e-12)
        u = u_unit / (np.linalg.norm(u_unit) + 1e-12)
        n = np.array([-u[1], u[0]]); n = n / (np.linalg.norm(n) + 1e-12)
        xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
        mx, my = 0.03*(xmax-xmin), 0.03*(ymax-ymin)
        back = back0; side = side0 * R
        txt = ax.text(0, 0, text, ha="center", va="center",
                      fontsize=fs, color=color,
                      bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.6),
                      clip_on=True, zorder=3, visible=False)
        for _ in range(60):
            base = tip_xy - back * L * u
            cand = [base + side * n, base - side * n]
            for c in cand:
                c[0] = min(max(c[0], xmin+mx), xmax-mx)
                c[1] = min(max(c[1], ymin+my), ymax-my)
            scores = []
            for c in cand:
                txt.set_position((c[0], c[1])); txt.set_visible(True)
                fig.canvas.draw()
                bb = txt.get_window_extent(renderer=fig.canvas.get_renderer())
                rect = (bb.x0 - px_clear, bb.y0 - px_clear, bb.x1 + px_clear, bb.y1 + px_clear)
                hit = _seg_intersects_rect(A1, B1, rect) or _seg_intersects_rect(A2, B2, rect)
                scores.append((not hit, c, rect))
            scores.sort(reverse=True, key=lambda t: (t[0],))
            ok, best_c, _ = scores[0]
            if ok:
                txt.set_position((best_c[0], best_c[1])); txt.set_visible(True)
                return
            back = min(back + 0.04, max_back)
            side = min(side + 0.06 * R, max_side * R)
        txt.set_position((best_c[0], best_c[1])); txt.set_visible(True)

    if annotate:
        tip_s = np.array([xs, ys]); tip_t = np.array([xt, yt])
        _place_label_near_tip_pixel_safe(ax, tip_s, vsu,
                                         text=label_s,   # << use global/subspace-aware label
                                         color="dimgray",
                                         back0=0.30, side0=0.20, px_clear=24, max_back=0.70, max_side=0.60)
        _place_label_near_tip_pixel_safe(ax, tip_t, vtu,
                                         text=label_t,   # << use global/subspace-aware label
                                         color="k",
                                         back0=0.30, side0=0.20, px_clear=26, max_back=0.70, max_side=0.60)


    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(lab_i, fontsize=13)
    ax.set_ylabel(lab_j, fontsize=13)
    ax.set_title(title or rf"Cost contours in ({lab_i}, {lab_j})", fontsize=15, pad=8)

    if show_coupling_box:
        d = np.sqrt(np.clip(np.diag(Hs), 1e-18, np.inf))
        Cij = float(Hs[i, j] / (d[i] * d[j]))
        txt = rf"$C_{{{i+1},{j+1}}} = {Cij:+.2f}$"
        ax.text(0.98, 0.98, txt, transform=ax.transAxes,
                ha="right", va="top", fontsize=11,
                bbox=dict(fc="white", ec="gray", alpha=0.9, pad=3))

    if save_name:
        save_vector(fig, save_name)
    plt.show()
    plt.close(fig)

def auto_plot_coupled_pairs(
    H_fd, labels,
    *,
    thresh_H=0.60,
    use_sigma=False,
    thresh_S=0.60,
    logic="and",
    limit=None,
    save_prefix="Hessian_contour_auto",
    ngrid=201, span_factor=3.0, cmap="Spectral_r", figsize=(6.5,6.0),
    ridge=0.0,
    eig_mode="subspace"   # <<< NEW
):
    df = coupling_table(H_fd, labels, ridge=ridge).copy()

    mask_H = (df["absH"] >= thresh_H)
    if use_sigma:
        mask_S = (df["absS"] >= thresh_S)
        mask = (mask_H & mask_S) if logic.lower()=="and" else (mask_H | mask_S)
        sort_key = "score"
    else:
        mask = mask_H
        sort_key = "absH"

    chosen = df[mask].sort_values(sort_key, ascending=False).reset_index(drop=True)
    if limit is not None:
        chosen = chosen.head(int(limit))

    if chosen.empty:
        print("No parameter pairs exceeded the coupling criteria.")
        return df, chosen

    print("\n=== Strongly coupled pairs to plot ===")
    for k, r in chosen.iterrows():
        i, j = int(r["i"]), int(r["j"])
        print(f"[{k+1}] {labels[i]} ↔ {labels[j]}  |  C({i+1},{j+1})={r['rho_H']:+.3f}")

    try:
        latex_labels = latexify_param_list(labels)
    except Exception:
        latex_labels = labels

    for k, r in chosen.iterrows():
        i, j = int(r["i"]), int(r["j"])
        main_title = rf"Coupling: {latex_labels[i]} vs {latex_labels[j]}"
        fname = f"{save_prefix}_{i}-{j}"

        plot_subspace_contour(
            H_fd, i, j, labels,
            title=main_title,
            ngrid=ngrid, span_factor=span_factor,
            cmap=cmap, figsize=figsize, save_name=fname, ridge=ridge,
            eig_mode=eig_mode   # <<< forward it
        )

    print(f"\nPlotted {len(chosen)} pairs (of {mask.sum()} that met the criteria).")
    return df, chosen

# 1) Quick printed report of strongest couplings
coupling_reports(H_fd, THETA_LABELS, top_k=10, thresh=0.60, ridge=0.0)

# 2) Auto-plot only the coupled pairs (true 2D tilt)
df_all_pairs, df_plotted = auto_plot_coupled_pairs(
    H_fd, THETA_LABELS,
    thresh_H=0.60, use_sigma=False, limit=None,
    save_prefix="August_Hessian_contour_auto",
    ngrid=201, span_factor=3.0, cmap="Spectral_r", figsize=(6.6, 6.1),
    ridge=0.0,
    eig_mode="global_best2"   # <<< add this arg
)

end = timer()
print("\nTotal runtime: {:.2f} minutes.".format((end - start) / 60))


# In[ ]:





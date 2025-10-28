#!/usr/bin/env python
# coding: utf-8

# In[1]:


# =============================================================================
# Purpose (user inputs required)
# =============================================================================
# Simulates antibiotic pulse dynamics and computes:
#  - Delay vs. pulse length N (fully coupled model)
#  - Peak RNA/Protein ratio for N = 4
#  - Derived sensitivities for K_D = k_off/k_on and R_P = P_in/P_out
#  - IC50 sensitivity summary
#  - Finite-difference Hessians (and eigenvalues) for: delay(N), RNA/P, max delay, IC50
#
# You must provide base model parameters (replace Nones below):
#   k_off, k_on, P_in, P_out - parameters from Steady State Optimization
#
# Optional user controls (leave as-is unless you need changes):
#   - N_vec: pulse lengths analyzed (default: [4, 6, 8, 10])
#   - perturbation_values: % perturbations for sensitivities (default: [0.5, 1, 10, 50])
#   - ODE tolerances/method: solve_ivp(..., method="BDF", rtol=1e-6, atol=1e-9)
#
# Outputs (printed to console):
#   - Derived sensitivity table (K_D, R_P)
#   - IC50 sensitivity summary
#   - Hessian matrices, eigenvalues, symmetry checks for each objective
#   - Total runtime (minutes)
#
# Formatting notes:
#   - All base params are floats; units must be consistent with your model.
#   - IC50 is computed analytically inside the script from the base params.
#   - Initial state uses r_min, kappa_t, lambda_0_glu (see y0 in code).
# =============================================================================

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from timeit import default_timer as timer

start = timer()

# ----------------------------------------------------------------
# BASE + DERIVED PARAMETERS
# ----------------------------------------------------------------
params = {
    "r_min": 5.4,
    "r_max": 54.4,
    "kappa_t": 0.058,
    "k_off": None,
    "k_on": None,
    "P_in": None,
    "P_out": None,
    "lambda_0_gly": 0.4
}

def update_derived_params(p):
    """
    Given a parameter dictionary p with keys r_min, r_max, kappa_t, and lambda_0_gly,
    compute and add derived parameters:
      - delta_r: r_max - r_min
      - lambda_max: kappa_t * delta_r
      - kappa_n: 1 / (delta_r * (1/lambda_0_gly - 1/(kappa_t*delta_r)))
    """
    p = p.copy()
    p["delta_r"] = p["r_max"] - p["r_min"]
    p["lambda_max"] = p["kappa_t"] * p["delta_r"]
    p["kappa_n"] = 1.0 / (p["delta_r"] * ((1.0 / p["lambda_0_gly"]) - (1.0 / (p["kappa_t"] * p["delta_r"]))))
    return p

# Update parameters with derived values.
params = update_derived_params(params)

# ----------------------------------------------------------------
# IC50 CALCULATION
# ----------------------------------------------------------------
def compute_IC50(p):
    """
    New IC50 formula:
      - K_D = k_off / k_on
      - lambda_0_star = 2 * sqrt(P_out * kappa_t * K_D)
      - IC50_star = (delta_r * lambda_0_star) / (2 * P_in)
      - IC50 = 0.5 * IC50_star * (lambda_0_gly/lambda_0_star + lambda_0_star/lambda_0_gly)
    """
    K_D = p["k_off"] / p["k_on"]
    lambda_0_gly = p["lambda_0_gly"]
    delta_r = p["delta_r"]
    kappa_t = p["kappa_t"]
    P_in = p["P_in"]
    P_out = p["P_out"]

    lambda_0_star = 2.0 * np.sqrt(P_out * kappa_t * K_D)
    IC50_star = (delta_r * lambda_0_star) / (2.0 * P_in)
    return 0.5 * IC50_star * ((lambda_0_gly / lambda_0_star) + (lambda_0_star / lambda_0_gly))

# ----------------------------------------------------------------
# ODE MODEL EQUATIONS
# ----------------------------------------------------------------
def model_equations(t, y, p, IC50_val, N):
    """
    Computes the time derivatives for the model:
      y[0] = a (antibiotic concentration)
      y[1] = r_u (unbound ribosomes)
      y[2] = r_b (bound ribosomes)
    The synthesis rate and external antibiotic input depend on time (t) and parameter N.
    """
    a, r_u, r_b = y
    r_min, r_max = p["r_min"], p["r_max"]
    kappa_t, lambda_0_gly = p["kappa_t"], p["lambda_0_gly"]
    delta_r, kappa_n = p["delta_r"], p["kappa_n"]

    if t <= 1:
        a_ex = 0
        x = (r_u - r_min) * kappa_t
        s = x * (r_max - ((x * delta_r) * ((1.0 / lambda_0_gly) - (1.0 / (kappa_t * delta_r)))))
    elif t <= N + 1:
        a_ex = (4 * IC50_val) / N
        x = (r_u - r_min) * kappa_t
        s = x * (r_max - ((x * delta_r) * ((1.0 / lambda_0_gly) - (1.0 / (kappa_t * delta_r)))))
    else:
        a_ex = 0
        x = (r_max - r_u - r_b) * p["kappa_n"]
        s = x * (r_min + (x / kappa_t))
    
    F = (p["k_on"] * a * (r_u - r_min)) - (p["k_off"] * r_b)
    da_dt = -F - x * a + p["P_in"] * a_ex - p["P_out"] * a
    dr_u_dt = -F - x * r_u + s
    dr_b_dt = F - x * r_b
    return [da_dt, dr_u_dt, dr_b_dt]

# ----------------------------------------------------------------
# BASELINE OUTPUT (FULLY-COUPLED MODEL)
# ----------------------------------------------------------------
N_vec = [4, 6, 8, 10]
y0 = [0, params["r_min"] + (params["lambda_0_gly"] / params["kappa_t"]), 0]

def calculate_baseline_output(p):
    """
    For each N in N_vec:
      - Recompute IC50 from p.
      - Solve the ODE system over a time interval that depends on N.
      - Calculate the delay from the growth curves.
    Also, compute the peak RNA/P (using the normalized sum of ribosomal components) for N = 4.
    Returns: (delay_dict, rnap_val, max_delay)
    """
    p = update_derived_params(p)
    delay_dict = {}
    for N in N_vec:
        tMax = (N + 1) + 4 + (30.0 / p["lambda_0_gly"])
        time = np.linspace(0, tMax, int(10 * tMax))
        IC50_val = compute_IC50(p)
        sol = solve_ivp(lambda t, y: model_equations(t, y, p, IC50_val, N),
                        [time[0], time[-1]], y0, method="BDF", t_eval=time,
                        rtol=1e-6, atol=1e-9)
        
        gr1 = np.where((time > 0) & (time < N + 1),
                       (sol.y[1] - p["r_min"]) * p["kappa_t"],
                       0.0)
        gr2 = np.where(time >= N + 1,
                       (p["r_max"] - sol.y[1] - sol.y[2]) * p["kappa_n"],
                       0.0)
        dt = time[1] - time[0]
        OD = np.cumsum(gr1 + gr2) * dt
        pre, post = OD[1], OD[-1]
        delay_dict[N] = ((pre - post) / p["lambda_0_gly"]) + (time[-1] - time[1])
    
    max_delay = max(delay_dict.values())
    
    # Compute peak RNA/P for N = 4.
    N_val = 4
    tMax4 = (N_val + 1) + 4 + (30.0 / p["lambda_0_gly"])
    time4 = np.linspace(0, tMax4, int(10 * tMax4))
    IC50_val = compute_IC50(p)
    sol4 = solve_ivp(lambda t, y: model_equations(t, y, p, IC50_val, 4),
                     [time4[0], time4[-1]], y0, method="BDF", t_eval=time4,
                     rtol=1e-6, atol=1e-9)
    rtot = sol4.y[1] + sol4.y[2]
    idx0 = np.abs(time4 - 0.5).argmin()
    rnorm = rtot / rtot[idx0]
    rnap_val = np.max(rnorm)
    return delay_dict, rnap_val, max_delay

# ----------------------------------------------------------------
# SENSITIVITY ANALYSIS FUNCTION (FINITE DIFFERENCES)
# ----------------------------------------------------------------
def calculate_sensitivity(param_name, delta_percent, params):
    """
    Compute log-elasticities (sensitivities) for:
      - delay (per N),
      - peak RNA/P, and
      - maximum delay
    when parameter 'param_name' is perturbed by +delta_percent%.
    For the parameter "delta_r", we perturb r_max (keeping r_min fixed).
    """
    epsilon = delta_percent / 100.0
    baseline_delays, baseline_rnap, baseline_max = calculate_baseline_output(params)
    
    # Create perturbed parameters.
    params_perturbed = params.copy()
    params_perturbed[param_name] *= (1 + epsilon)
    if param_name == "delta_r":
        params_perturbed["r_max"] *= (1 + epsilon)
    params_perturbed = update_derived_params(params_perturbed)
    
    pos_delays, pos_rnap, pos_max = calculate_baseline_output(params_perturbed)
    
    delay_sens_dict = {}
    for N_val in baseline_delays:
        base_delay = baseline_delays[N_val]
        pos_delay = pos_delays[N_val]
        if base_delay > 0 and pos_delay > 0:
            sens = (np.log(pos_delay) - np.log(base_delay)) / np.log(1 + epsilon)
        else:
            sens = 0.0
        delay_sens_dict[N_val] = sens

    max_delay_sens = (np.log(pos_max) - np.log(baseline_max)) / np.log(1 + epsilon) \
                     if baseline_max > 0 and pos_max > 0 else 0.0
    rnap_sens = (np.log(pos_rnap) - np.log(baseline_rnap)) / np.log(1 + epsilon) \
                if baseline_rnap and baseline_rnap > 0 and pos_rnap and pos_rnap > 0 else 0.0
    
    return delay_sens_dict, rnap_sens, max_delay_sens

# ----------------------------------------------------------------
# DERIVED SENSITIVITY ANALYSIS for K_D and R_P
# (K_D = k_off/k_on, R_P = P_in/P_out)
# ----------------------------------------------------------------
perturbation_values = [0.5, 1, 10, 50]
derived_results = []

for dp in perturbation_values:
    ds_koff, r_koff, max_off = calculate_sensitivity("k_off", dp, params)
    ds_kon,  r_kon,  max_on  = calculate_sensitivity("k_on",  dp, params)
    for N_val in N_vec:
        kd_delay_sens = ds_koff[N_val] - ds_kon[N_val]
        kd_rnap_sens = r_koff - r_kon
        derived_results.append({
            "Parameter": "K_D",
            "DeltaPercent": dp,
            "N": N_val,
            "Delay Sensitivity": kd_delay_sens,
            "Peak RNA/P Sensitivity": kd_rnap_sens,
            "Max Delay Sensitivity": None
        })
    kd_max_delay_sens = max_off - max_on
    kd_rnap_max = r_koff - r_kon
    derived_results.append({
        "Parameter": "K_D",
        "DeltaPercent": dp,
        "N": "Max",
        "Delay Sensitivity": None,
        "Peak RNA/P Sensitivity": kd_rnap_max,
        "Max Delay Sensitivity": kd_max_delay_sens
    })
    
    ds_Pin,  r_Pin,  max_Pin = calculate_sensitivity("P_in",  dp, params)
    ds_Pout, r_Pout, max_Pout = calculate_sensitivity("P_out", dp, params)
    for N_val in N_vec:
        rp_delay_sens = ds_Pin[N_val] - ds_Pout[N_val]
        rp_rnap_sens = r_Pin - r_Pout
        derived_results.append({
            "Parameter": "R_P",
            "DeltaPercent": dp,
            "N": N_val,
            "Delay Sensitivity": rp_delay_sens,
            "Peak RNA/P Sensitivity": rp_rnap_sens,
            "Max Delay Sensitivity": None
        })
    rp_max_delay_sens = max_Pin - max_Pout
    rp_rnap_max = r_Pin - r_Pout
    derived_results.append({
        "Parameter": "R_P",
        "DeltaPercent": dp,
        "N": "Max",
        "Delay Sensitivity": None,
        "Peak RNA/P Sensitivity": rp_rnap_max,
        "Max Delay Sensitivity": rp_max_delay_sens
    })

df_derived = pd.DataFrame(derived_results)
df_derived["Parameter"] = pd.Categorical(df_derived["Parameter"], categories=["K_D", "R_P"], ordered=True)
df_derived["N_numeric"] = df_derived["N"].apply(lambda v: float(v) if str(v).replace('.', '', 1).isdigit() else np.inf)
df_derived = df_derived.sort_values(by=["Parameter", "DeltaPercent", "N_numeric"]).drop(columns="N_numeric")

col_width = 18
header = (f"{'Parameter'.ljust(col_width)}|"
          f"{'Delta(%)'.rjust(10)}|"
          f"{'N'.rjust(5)}|"
          f"{'Delay Sensitivity'.rjust(col_width)}|"
          f"{'Max Delay Sens'.rjust(col_width)}|"
          f"{'Peak RNA/P Sensitivity'.rjust(col_width)}")
print("\n\n================== DERIVED SENSITIVITY SUMMARY ==================")
print(header)
print("-" * (len(header) + 8))
for _, row in df_derived.iterrows():
    param    = row["Parameter"]
    dp       = row["DeltaPercent"]
    n_val    = row["N"]
    ds_val   = row["Delay Sensitivity"]
    max_val  = row["Max Delay Sensitivity"]
    rna_val  = row["Peak RNA/P Sensitivity"]
    ds_str = f"{ds_val:.4f}" if pd.notnull(ds_val) else "   -"
    max_str = f"{max_val:.4f}" if pd.notnull(max_val) else "   -"
    rna_str = f"{rna_val:.4f}" if pd.notnull(rna_val) else "   -"
    row_str = (f"{str(param).ljust(col_width)}|"
               f"{str(dp).rjust(10)}|"
               f"{str(n_val).rjust(5)}|"
               f"{ds_str.rjust(col_width)}|"
               f"{max_str.rjust(col_width)}|"
               f"{rna_str.rjust(col_width)}")
    print(row_str)

# ----------------------------------------------------------------
# IC50 & ratio‐sensitivities SUMMARY (using named variables)
# ----------------------------------------------------------------
ic50_summary = []
for dp in perturbation_values:
    # use a named variable instead of ε
    epsilon = dp / 100.0

    # --- baseline quantities ---
    IC50_0 = compute_IC50(params)
    KD_0   = params["k_off"] / params["k_on"]
    RP_0   = params["P_in"]  / params["P_out"]

    # --- IC50 sensitivity wrt k_off (i.e. K_D) ---
    p1 = params.copy()
    p1["k_off"] *= (1 + epsilon)
    p1 = update_derived_params(p1)
    IC50_koff = compute_IC50(p1)
    S_IC50_koff = (np.log(IC50_koff) - np.log(IC50_0)) / np.log(1 + epsilon)

    # --- IC50 sensitivity wrt P_in (i.e. R_P) ---
    p2 = params.copy()
    p2["P_in"] *= (1 + epsilon)
    p2 = update_derived_params(p2)
    IC50_pin = compute_IC50(p2)
    S_IC50_pin = (np.log(IC50_pin) - np.log(IC50_0)) / np.log(1 + epsilon)

    # --- KD sensitivity wrt k_on ---
    p3 = params.copy()
    p3["k_on"] *= (1 + epsilon)
    p3 = update_derived_params(p3)
    KD_kon = p3["k_off"] / p3["k_on"]
    S_KD_kon = (np.log(KD_kon) - np.log(KD_0)) / np.log(1 + epsilon)

    # --- RP sensitivity wrt P_out ---
    p4 = params.copy()
    p4["P_out"] *= (1 + epsilon)
    p4 = update_derived_params(p4)
    RP_pout = p4["P_in"] / p4["P_out"]
    S_RP_pout = (np.log(RP_pout) - np.log(RP_0)) / np.log(1 + epsilon)

    ic50_summary.append({
        "Delta%":           dp,
        "S(IC50, k_off)":   S_IC50_koff,
        "S(IC50, P_in)":    S_IC50_pin,
        "S(KD, k_on)":      S_KD_kon,
        "S(RP, P_out)":     S_RP_pout,
    })

# --- pretty print ---
print("\n\n========= IC50 SENSITIVITY SUMMARY =========")
headers = ["Delta(%)","S(IC50, k_off)","S(IC50, P_in)","S(KD, k_on)","S(RP, P_out)"]
print(" | ".join(h.ljust(15) for h in headers))
print("-" * 100)
for row in ic50_summary:
    print(
        f"{row['Delta%']:<15}|"
        f"{row['S(IC50, k_off)']:<15.4f}|"
        f"{row['S(IC50, P_in)']:<15.4f}|"
        f"{row['S(KD, k_on)']:<15.4f}|"
        f"{row['S(RP, P_out)']:<15.4f}"
    )

# ----------------------------------------------------------------
# HESSIAN COMPUTATION VIA FINITE DIFFERENCES
# ----------------------------------------------------------------
def compute_hessian(f, param_vector, eps_val=1e-4):
    """
    Compute the Hessian matrix for a scalar function f at the given parameter vector,
    using finite differences.
    """
    param_vector = np.array(param_vector, dtype=float)
    n = len(param_vector)
    hessian = np.zeros((n, n))
    h = eps_val * np.where(np.abs(param_vector) > 0, np.abs(param_vector), 1.0)
    for i in range(n):
        for j in range(n):
            p_pp = param_vector.copy(); p_pp[i] += h[i]; p_pp[j] += h[j]
            p_pm = param_vector.copy(); p_pm[i] += h[i]; p_pm[j] -= h[j]
            p_mp = param_vector.copy(); p_mp[i] -= h[i]; p_mp[j] += h[j]
            p_mm = param_vector.copy(); p_mm[i] -= h[i]; p_mm[j] -= h[j]
            
            f_pp = f(p_pp)
            f_pm = f(p_pm)
            f_mp = f(p_mp)
            f_mm = f(p_mm)
            
            hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * h[i] * h[j])
    return hessian

def check_symmetry(H, tol=1e-5):
    return np.allclose(H, H.T, atol=tol)

# Example: Hessian with respect to base parameters [k_off, k_on, P_in, P_out]
base_params = [params["k_off"], params["k_on"], params["P_in"], params["P_out"]]

def f_delay_N(param_vector, N_val=4):
    new_params = params.copy()
    new_params["k_off"], new_params["k_on"], new_params["P_in"], new_params["P_out"] = param_vector
    new_params = update_derived_params(new_params)
    delays, _, _ = calculate_baseline_output(new_params)
    return delays[N_val]

def f_rnap(param_vector):
    new_params = params.copy()
    new_params["k_off"], new_params["k_on"], new_params["P_in"], new_params["P_out"] = param_vector
    new_params = update_derived_params(new_params)
    _, rnap, _ = calculate_baseline_output(new_params)
    return rnap

def f_max_delay(param_vector):
    new_params = params.copy()
    new_params["k_off"], new_params["k_on"], new_params["P_in"], new_params["P_out"] = param_vector
    new_params = update_derived_params(new_params)
    _, _, max_delay = calculate_baseline_output(new_params)
    return max_delay

def f_ic50(param_vector):
    new_params = params.copy()
    new_params["k_off"], new_params["k_on"], new_params["P_in"], new_params["P_out"] = param_vector
    new_params = update_derived_params(new_params)
    return compute_IC50(new_params)

print("\n=== Hessian Analysis with respect to base parameters ===")
Hessians_delay = {}
eigenvalues_delay = {}
symmetry_delay = {}

for N_val in N_vec:
    H = compute_hessian(lambda p: f_delay_N(p, N_val), base_params)
    eigs = np.linalg.eigvals(H)
    symm = check_symmetry(H)
    
    Hessians_delay[N_val] = H
    eigenvalues_delay[N_val] = eigs
    symmetry_delay[N_val] = symm
    
    print(f"\n--- Hessian for delay (N = {N_val}) ---")
    print("Hessian:\n", H)
    print("Eigenvalues:", eigs)
    print("Approximately symmetric:", symm)

H_rnap = compute_hessian(f_rnap, base_params)
eigs_rnap = np.linalg.eigvals(H_rnap)
symmetry_rnap = check_symmetry(H_rnap)

print("\n--- Hessian for RNA/P (N=4) ---")
print("Hessian:\n", H_rnap)
print("Eigenvalues:", eigs_rnap)
print("Approximately symmetric:", symmetry_rnap)

H_max_delay = compute_hessian(f_max_delay, base_params)
eigs_max_delay = np.linalg.eigvals(H_max_delay)
symmetry_max_delay = check_symmetry(H_max_delay)

print("\n--- Hessian for Max Delay ---")
print("Hessian:\n", H_max_delay)
print("Eigenvalues:", eigs_max_delay)
print("Approximately symmetric:", symmetry_max_delay)

H_ic50 = compute_hessian(f_ic50, base_params)
eigs_ic50 = np.linalg.eigvals(H_ic50)
symmetry_ic50 = check_symmetry(H_ic50)

print("\n--- Hessian for IC50 ---")
print("Hessian:\n", H_ic50)
print("Eigenvalues:", eigs_ic50)
print("Approximately symmetric:", symmetry_ic50)

end = timer()
print("\nTotal runtime: {:.2f} minutes.".format((end - start) / 60))


# In[ ]:





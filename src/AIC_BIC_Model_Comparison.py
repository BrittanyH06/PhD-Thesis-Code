#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Model comparison script for three successive antibiotic response models:

1. Original Greulich-style model (ribosome-limited only)
2. Metabolic correction model (switching to kappa_n after the pulse)
3. Feedback model (ppGpp-like feedback via lambda_N - lambda_T)

The script:
- Encodes all three ODE models (glucose and glycerol)
- Predicts delay time, RNA/protein ratio, and inhibition curves
- Builds standardized residuals across all datasets
- Computes RSS, AIC, and BIC for each model
- Reports ΔAIC and ΔBIC relative to the best model

NOTE: This public version uses placeholder experimental data and
placeholder fitted parameter vectors. Replace the marked sections
with your actual data and optimal parameters for real analysis.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

###############################################################################
# GLOBAL CONSTANTS FOR CORRECTED (METABOLIC/FEEDBACK) MODELS
###############################################################################

# Growth-law / proteome constants for corrected models
r_min_corr   = 5.4
r_max_corr   = 54.4
delta_r_corr = r_max_corr - r_min_corr
kappa_t_corr = 0.058
lambda_max_corr = kappa_t_corr * delta_r_corr  # maximal possible growth rate

# Drug-free (raw) growth rates from experiments (placeholders here)
lambda_0_glu = 0.64   # h^-1, glucose  (TODO: replace if different)
lambda_0_gly = 0.40   # h^-1, glycerol (TODO: replace if different)

# Nutritional capacities for corrected models (from growth-law inversion)
kappa_n_glu = 1.0 / (delta_r_corr * ((1.0 / lambda_0_glu) - (1.0 / lambda_max_corr)))
kappa_n_gly = 1.0 / (delta_r_corr * ((1.0 / lambda_0_gly) - (1.0 / lambda_max_corr)))

###############################################################################
# ORIGINAL GREULICH CONSTANTS (AS GIVEN IN THE LITERATURE)
###############################################################################

r_min_orig = 19.3
r_max_orig = 65.8
delta_r_orig = r_max_orig - r_min_orig
kappa_t_orig = 6.1e-2
lambda_0_orig = 1.0           # Greulich's dimensionless reference λ0
IC50_orig = 14.3
P_in_orig = 2000.0
P_out_orig = 100.0
k_on_orig = 1000.0
k_off_orig = 1e5

# Nutritional capacity consistent with the original parameter set
kappa_n_orig = 1.0 / (delta_r_orig * ((1.0 / lambda_0_orig) - (1.0 / (kappa_t_orig * delta_r_orig))))

###############################################################################
# EXPERIMENTAL DATA (PLACEHOLDERS)
###############################################################################
# Replace this whole block with your actual experimental datasets.
# The shapes and variable names are kept so the rest of the script works.

# --- Delay time data (placeholders) ---
# Pulse lengths for glucose and glycerol
N_values_glu = [2, 4, 6, 8, 10]     # TODO: adjust as needed
N_values_gly = [4, 6, 8, 10]        # TODO: adjust as needed

# Delay times (hours); here zeros are placeholders
exp_delay_times_glu = {N: 0.0 for N in N_values_glu}  # TODO: fill with experimental Δt(N)
exp_delay_times_gly = {N: 0.0 for N in N_values_gly}  # TODO: fill with experimental Δt(N)

# Experimental standard deviations for delay (for weighting)
std_glu = {N: 1.0 for N in N_values_glu}  # TODO: replace with measured std devs
std_gly = {N: 1.0 for N in N_values_gly}  # TODO: replace with measured std devs

# Convert dicts into ordered arrays aligned with N_values_*
N_delay_glu = np.array(N_values_glu, dtype=float)
delay_glu_obs = np.array([exp_delay_times_glu[N] for N in N_values_glu], dtype=float)
delay_glu_std = np.array([std_glu[N] for N in N_values_glu], dtype=float)

N_delay_gly = np.array(N_values_gly, dtype=float)
delay_gly_obs = np.array([exp_delay_times_gly[N] for N in N_values_gly], dtype=float)
delay_gly_std = np.array([std_gly[N] for N in N_values_gly], dtype=float)

# --- RNA/P data (placeholders) ---
# Example: one pulse length (N=4 h) with 9 time points
# Replace with your measured RNA/P ratios and standard deviations.
lab_RNA_P_glu = {
    4: [1.0] * 9  # TODO: replace with experimental RNA/P data for glucose
}
lab_RNA_P_glu_std = {
    4: [1.0] * 9  # TODO: replace with experimental std devs
}

lab_RNA_P_gly = {
    4: [1.0] * 9  # TODO: replace with experimental RNA/P data for glycerol
}
lab_RNA_P_gly_std = {
    4: [1.0] * 9  # TODO: replace with experimental std devs
}

# Time points at which RNA/P was sampled (hours)
time_points_4H = [0.5, 1.75, 2.5, 3.25, 4.0, 4.75, 5.5, 6.0, 6.5]  # TODO: adjust if needed
time_points = np.array(time_points_4H, dtype=float)

rnap_glu_obs = np.array(lab_RNA_P_glu[4], dtype=float)
rnap_glu_std = np.array(lab_RNA_P_glu_std[4], dtype=float)

rnap_gly_obs = np.array(lab_RNA_P_gly[4], dtype=float)
rnap_gly_std = np.array(lab_RNA_P_gly_std[4], dtype=float)

# --- Inhibition data (placeholders) ---
# External TET concentrations (µM)
TET_concentrations = np.array([0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=float)  # TODO: adjust if needed

# Normalized growth λ/λ0 vs TET; these are dummy values
growth_glucose  = np.array([1.0, 0.8, 0.6, 0.5, 0.3, 0.2], dtype=float)  # TODO: replace with data
stdev_glucose   = np.array([0.1] * 6, dtype=float)                        # TODO: replace with std devs

growth_glycerol = np.array([1.0, 0.8, 0.6, 0.5, 0.3, 0.2], dtype=float)   # TODO: replace with data
stdev_glycerol  = np.array([0.1] * 6, dtype=float)                        # TODO: replace with std devs


def inhibition_model_norm(TET, IC50):
    """
    Simple one-parameter inhibition curve:
        λ/λ0 = 1 / (1 + TET / IC50)
    used to fit experimental IC50 for each carbon source.
    """
    return 1.0 / (1.0 + (TET / IC50))


# Fit experimental IC50 values with weighted non-linear least squares.
# With placeholder data this will just give some dummy IC50; for real use,
# replace growth_* and stdev_* with your measurements.
popt_glu_norm, pcov_glu_norm = curve_fit(
    inhibition_model_norm,
    TET_concentrations,
    growth_glucose,
    sigma=stdev_glucose,
    absolute_sigma=True,
    p0=[1.0],
)
popt_gly_norm, pcov_gly_norm = curve_fit(
    inhibition_model_norm,
    TET_concentrations,
    growth_glycerol,
    sigma=stdev_glycerol,
    absolute_sigma=True,
    p0=[1.0],
)

IC50_exp_glu = float(popt_glu_norm[0])  # TODO: you may also set these manually
IC50_exp_gly = float(popt_gly_norm[0])

# Lab IC50 and pulse plateau prefactors used to set pulse doses in the models
IC50_MINE = {
    "glucose": IC50_exp_glu,
    "glycerol": IC50_exp_gly,
}

# Experimental pulse (placeholder): N=4 h at fixed [TET]
N_LAB = 4.0
LAB_PULSE_CONC = {
    "glucose": 1.0,   # µM used in N=4 pulse experiments (glucose)   # TODO
    "glycerol": 1.0,  # µM used in N=4 pulse experiments (glycerol)  # TODO
}
# Dimensionless "prefactor" ~ 4*IC50/N-like ratio used to set a_ex
PREFAC = {sub: N_LAB * LAB_PULSE_CONC[sub] / IC50_MINE[sub] for sub in IC50_MINE}
prefactor_glu = PREFAC["glucose"]
prefactor_gly = PREFAC["glycerol"]
lab_IC_50_glu = IC50_MINE["glucose"]
lab_IC_50_gly = IC50_MINE["glycerol"]

###############################################################################
# ODE SYSTEMS
###############################################################################

def model_equations_original(t, y, N, parameters, IC_50, z):
    """
    Original Greulich-style model with translational limitation only.
    Antibiotic pulses enter as step-wise a_ex(t).
    """
    a, r_u, r_b = y

    r_min = parameters["r_min"]
    kappa_t = parameters["kappa_t"]
    delta_r = parameters["delta_r"]

    k_on = parameters["k_on"]
    k_off = parameters["k_off"]
    P_in = parameters["P_in"]
    P_out = parameters["P_out"]
    lambda_0_model = parameters["lambda_0_model"]

    # Step-pulse antibiotic: from t=1 to t=N+1, external concentration is non-zero
    if 1.0 <= t <= N + 1.0:
        a_ex = (4.0 * IC_50) / N
    else:
        a_ex = 0.0

    x = (r_u - r_min) * kappa_t

    s = x * (r_max_orig - ((x * delta_r) * ((1.0 / lambda_0_model) - (1.0 / (kappa_t * delta_r)))))

    F = (k_on * a * (r_u - r_min)) - (k_off * r_b)

    da_dt  = -F - (x * a) + (P_in * a_ex) - (P_out * a)
    dr_u_dt = -F - (x * r_u) + s
    dr_b_dt =  F - (x * r_b)

    return [da_dt, dr_u_dt, dr_b_dt]


def model_equations_metabolic(t, y, N, parameters, IC_50, z):
    """
    Metabolic correction model for one substrate (glucose or glycerol).

    The model:
    - Uses translational limitation before and during the pulse (kappa_t branch).
    - Switches to metabolic limitation after the pulse via kappa_n.
    """
    a, r_u, r_b = y

    r_min   = parameters["r_min"]
    r_max   = parameters["r_max"]
    delta_r = parameters["delta_r"]
    kappa_t = parameters["kappa_t"]
    lambda_0 = parameters["lambda_0"]
    kappa_n = parameters["kappa_n"]

    k_on = parameters["k_on"]
    k_off = parameters["k_off"]
    P_in = parameters["P_in"]
    P_out = parameters["P_out"]

    prefactor = parameters["prefactor"]
    lab_IC50  = parameters["IC50_lab"]

    if 0.0 <= t <= 1.0:
        # Pre-pulse: no external antibiotic, translational branch
        a_ex = 0.0
        x = (r_u - r_min) * kappa_t
        s = x * (r_max - ((x * delta_r) * ((1.0 / lambda_0) - (1.0 / (kappa_t * delta_r)))))
    elif 1.0 <= t <= N + 1.0:
        # Pulse: external antibiotic added, still translational branch
        a_ex = (prefactor * lab_IC50) / N
        x = (r_u - r_min) * kappa_t
        s = x * (r_max - ((x * delta_r) * ((1.0 / lambda_0) - (1.0 / (kappa_t * delta_r)))))
    else:
        # Post-pulse: no external antibiotic, metabolic branch
        a_ex = 0.0
        x = (r_max - r_u - r_b) * kappa_n
        s = x * (r_min + (x / kappa_t))

    F = (k_on * a * (r_u - r_min)) - (k_off * r_b)

    da_dt  = -F - (x * a) + (P_in * a_ex) - (P_out * a)
    dr_u_dt = -F - (x * r_u) + s
    dr_b_dt =  F - (x * r_b)

    return [da_dt, dr_u_dt, dr_b_dt]


def model_equations_feedback(t, y, N, parameters, IC_50, z):
    """
    Feedback model for one substrate (glucose or glycerol).

    Key features:
    - Uses min(λ_T, λ_N) = min(kappa_t*(r_u - r_min), kappa_n*(r_max - r_u - r_b)).
    - Implements feedback in synthesis rate via (λ_N - λ_T) ~ (x_dil - x_syn).
    - Distinguishes pre-pulse, pulse, and post-pulse synthesis regimes.
    """
    a, r_u, r_b = y

    r_min   = parameters["r_min"]
    r_max   = parameters["r_max"]
    delta_r = parameters["delta_r"]
    kappa_t = parameters["kappa_t"]
    lambda_0 = parameters["lambda_0"]
    kappa_n = parameters["kappa_n"]
    lambda_max = parameters["lambda_max"]

    k_on = parameters["k_on"]
    k_off = parameters["k_off"]
    P_in = parameters["P_in"]
    P_out = parameters["P_out"]

    prefactor = parameters["prefactor"]
    lab_IC50  = parameters["IC50_lab"]

    alpha_0 = parameters["alpha_0"]
    alpha_A = parameters["alpha_A"]

    t_on = 1.0
    t_off = t_on + N

    x_syn = (r_u - r_min) * kappa_t
    x_dil = (r_max - r_u - r_b) * kappa_n
    x     = min(x_syn, x_dil)

    if t < t_on:
        s_ss0 = lambda_0 * (r_max - (lambda_0 * delta_r * ((1.0 / lambda_0) - (1.0 / lambda_max))))
        s = s_ss0
        a_ex = 0.0
    elif t <= t_off:
        lambda_f = lambda_0 / (1.0 + (prefactor / N))
        s_ssA = lambda_f * (r_max - (lambda_f / kappa_n))
        s = s_ssA * (1.0 + (alpha_A * (x_dil - x_syn)))
        a_ex = (prefactor * lab_IC50) / N
    else:
        s_ss0 = lambda_0 * (r_max - (lambda_0 * delta_r * ((1.0 / lambda_0) - (1.0 / lambda_max))))
        s = s_ss0 * (1.0 + (alpha_0 * (x_dil - x_syn)))
        a_ex = 0.0

    F = (k_on * a * (r_u - r_min)) - (k_off * r_b)

    da_dt  = -F - (x * a) + (P_in * a_ex) - (P_out * a)
    dr_u_dt = -F - (x * r_u) + s
    dr_b_dt =  F - (x * r_b)

    return [da_dt, dr_u_dt, dr_b_dt]

###############################################################################
# DELAY-TIME FUNCTIONS
###############################################################################

def calculate_delay_time_feedback(parameters, N_values, IC_50, z, model_equations_func):
    lambda_0 = parameters['lambda_0']
    kappa_n = parameters['kappa_n']
    results = []
    for N in N_values:
        tMax = (N + 1) + 4 + (30.0 / lambda_0)
        time = np.linspace(0.0, tMax, int(200 * tMax))

        y0 = [0.0, parameters["r_min"] + (lambda_0 / parameters["kappa_t"]), 0.0]
        sol = solve_ivp(
            lambda t, y: model_equations_func(t, y, N, parameters, IC_50, z),
            [time[0], time[-1]],
            y0,
            method='BDF',
            t_eval=time
        )

        gr1 = (sol.y[1] - parameters["r_min"]) * parameters["kappa_t"]
        gr2 = (parameters["r_max"] - sol.y[1] - sol.y[2]) * kappa_n
        growth_rate_sum = np.minimum(gr1, gr2)

        OD = np.cumsum(growth_rate_sum) * (time[1] - time[0])
        delay_time = ((OD[1] - OD[-1]) / lambda_0) + (time[-1] - time[1])
        results.append((N, delay_time))
    return results


def calculate_delay_time_metabolic_wrapper(parameters, N_values, IC_50, z):
    results = []
    for N in N_values:
        lambda_0 = parameters['lambda_0']
        tMax = (N + 1) + 4 + (30.0 / lambda_0)
        time = np.linspace(0.0, tMax, int(200 * tMax))

        y0 = [0.0, parameters["r_min"] + (lambda_0 / parameters["kappa_t"]), 0.0]
        sol = solve_ivp(
            lambda t, y: model_equations_metabolic(t, y, N, parameters, IC_50, z),
            [time[0], time[-1]],
            y0,
            method='BDF',
            t_eval=time,
            rtol=1e-6,
            atol=1e-9
        )

        gr1 = np.zeros_like(time)
        gr2 = np.zeros_like(time)
        for i, t_val in enumerate(time):
            if 0.0 < t_val < N + 1.0:
                gr1[i] = (sol.y[1, i] - parameters["r_min"]) * parameters["kappa_t"]
            else:
                gr2[i] = (parameters["r_max"] - sol.y[1, i] - sol.y[2, i]) * parameters["kappa_n"]

        growth_rate_sum = gr1 + gr2
        OD = np.cumsum(growth_rate_sum) * (time[1] - time[0])
        delay_time = ((OD[1] - OD[-1]) / lambda_0) + (time[-1] - time[1])
        results.append((N, delay_time))
    return results


def calculate_delay_time_original_wrapper(parameters, N_values, IC_50, z):
    lambda_0_data = parameters["lambda_0_data"]
    results = []
    for N in N_values:
        tMax = (N + 1) + 4.0 + (30.0 / lambda_0_data)
        time = np.linspace(0.0, tMax, int(200 * tMax))

        y0 = [0.0, parameters["r_min"] + (parameters["lambda_0_model"] / parameters["kappa_t"]), 0.0]
        sol = solve_ivp(
            lambda t, y: model_equations_original(t, y, N, parameters, IC_50, z),
            [time[0], time[-1]],
            y0,
            method='BDF',
            t_eval=time
        )

        growth_rate = (sol.y[1] - parameters["r_min"]) * parameters["kappa_t"]
        OD = np.cumsum(growth_rate) * (time[1] - time[0])
        delay_time = ((OD[1] - OD[-1]) / lambda_0_data) + (time[-1] - time[1])
        results.append((N, delay_time))
    return results

###############################################################################
# RNA/P FUNCTION
###############################################################################

def calculate_r_total_normalized(lab_RNA_P, time_points, parameters, IC_50, z, model_equations_func):
    if 'lambda_0' in parameters:
        lambda_0 = parameters['lambda_0']
    else:
        lambda_0 = parameters['lambda_0_data']

    y0 = [0.0, parameters["r_min"] + (lambda_0 / parameters["kappa_t"]), 0.0]
    simulated_r_total = []
    for N in lab_RNA_P:
        tMax = (N + 1.0) + 4.0 + (30.0 / lambda_0)
        time = np.linspace(0.0, tMax, int(200 * tMax))
        sol_rtot = solve_ivp(
            lambda t, y: model_equations_func(t, y, N, parameters, IC_50, z),
            [time[0], time[-1]],
            y0,
            method='BDF',
            t_eval=time
        )
        r_total = sol_rtot.y[1] + sol_rtot.y[2]

        idx_ref = np.abs(sol_rtot.t - 0.5).argmin()
        denom = r_total[idx_ref] if r_total[idx_ref] != 0 else 1e-12
        r_total_norm = r_total / denom

        for t_pt in time_points:
            idx = np.abs(sol_rtot.t - t_pt).argmin()
            simulated_r_total.append(r_total_norm[idx])
    return simulated_r_total

###############################################################################
# INHIBITION: GREULICH CUBIC
###############################################################################

def solve_lambda_cubic(a_ex, lambda_0, k_off, k_on, P_in, P_out, kappa_t_val, delta_r_val):
    """
    Cardano solution of Greulich cubic for normalized growth x = λ/λ0.
    """
    if abs(a_ex) < 1e-12:
        return 1.0

    lambda0_star = 2.0 * np.sqrt(P_out * kappa_t_val * (k_off / k_on))
    IC50_star = (lambda0_star * delta_r_val) / (2.0 * P_in)

    a = 0.25 * (lambda0_star / lambda_0)**2
    b = (a_ex / (2.0 * IC50_star)) * (lambda0_star / lambda_0)

    A = -1.0
    B = a + b
    C = -a

    shift = 1.0/3.0
    p = B - (A**2)/3.0
    q = (2.0*(A**3))/27.0 - (A*B)/3.0 + C

    Delta = (q/2.0)**2 + (p/3.0)**3

    def cbrt(z):
        return np.sign(z) * abs(z)**(1.0/3.0)

    if Delta >= 0:
        sqrt_Delta = np.sqrt(Delta)
        y = cbrt(-q/2.0 + sqrt_Delta) + cbrt(-q/2.0 - sqrt_Delta)
        x = y + shift
    else:
        r_val = 2.0 * np.sqrt(-p/3.0)
        theta = np.arccos(-q / (2.0 * np.sqrt((-p/3.0)**3)))
        y1 = r_val * np.cos(theta/3.0)
        y2 = r_val * np.cos((theta + 2*np.pi)/3.0)
        y3 = r_val * np.cos((theta + 4*np.pi)/3.0)
        x1 = y1 + shift
        x2 = y2 + shift
        x3 = y3 + shift
        roots = [r for r in [x1, x2, x3] if r > 0]
        if not roots:
            return np.nan
        x = max(roots)
    return x

def simulate_inhibition_curve_cubic(params4, substrate, normalize, kappa_t_val, delta_r_val):
    k_off, k_on, P_in, P_out = params4
    lambda0 = lambda_0_glu if substrate.lower() == "glucose" else lambda_0_gly
    predicted = []
    for TET in TET_concentrations:
        x = solve_lambda_cubic(TET, lambda0, k_off, k_on, P_in, P_out, kappa_t_val, delta_r_val)
        if x is None or x < 0:
            predicted.append(0.0)
        else:
            predicted.append(x if normalize else x * lambda0)
    return np.array(predicted)

###############################################################################
# PREDICTION WRAPPERS
###############################################################################

def predict_delay(model_name, parameters, N_values, IC50, z):
    if model_name == "metabolic":
        res = calculate_delay_time_metabolic_wrapper(parameters, N_values, IC50, z)
    elif model_name == "feedback":
        res = calculate_delay_time_feedback(parameters, N_values, IC50, z, model_equations_feedback)
    elif model_name == "original":
        res = calculate_delay_time_original_wrapper(parameters, N_values, IC50, z)
    else:
        raise ValueError(model_name)
    return np.array([dt for (_, dt) in res], dtype=float)

def predict_rnap(model_name, parameters, lab_RNA_P_dict, time_points, IC50, z):
    if model_name == "metabolic":
        model_eq = model_equations_metabolic
    elif model_name == "feedback":
        model_eq = model_equations_feedback
    elif model_name == "original":
        model_eq = model_equations_original
    else:
        raise ValueError(model_name)
    rnap_pred = calculate_r_total_normalized(
        lab_RNA_P=lab_RNA_P_dict,
        time_points=time_points,
        parameters=parameters,
        IC_50=IC50,
        z=z,
        model_equations_func=model_eq,
    )
    return np.asarray(rnap_pred, dtype=float)

def predict_inhibition(model_name, z_params, substrate):
    if model_name == "metabolic":
        return simulate_inhibition_curve_cubic(z_params, substrate, True, kappa_t_corr, delta_r_corr)
    elif model_name == "feedback":
        return simulate_inhibition_curve_cubic(z_params[:4], substrate, True, kappa_t_corr, delta_r_corr)
    elif model_name == "original":
        return simulate_inhibition_curve_cubic(z_params, substrate, True, kappa_t_orig, delta_r_orig)
    else:
        raise ValueError(model_name)

###############################################################################
# RESIDUAL BUILDER
###############################################################################

def get_all_residuals_for_model(model_name, params_glu, params_gly):
    # Delay
    if model_name == "original":
        delay_glu_pred = predict_delay(model_name, params_glu, N_delay_glu, IC50_orig, params_glu["z"])
        delay_gly_pred = predict_delay(model_name, params_gly, N_delay_gly, IC50_orig, params_gly["z"])
    else:
        delay_glu_pred = predict_delay(model_name, params_glu, N_delay_glu, IC50_exp_glu, params_glu["z"])
        delay_gly_pred = predict_delay(model_name, params_gly, N_delay_gly, IC50_exp_gly, params_gly["z"])

    delay_glu_std_safe = np.where(delay_glu_std == 0.0, 1e-12, delay_glu_std)
    delay_gly_std_safe = np.where(delay_gly_std == 0.0, 1e-12, delay_gly_std)

    delay_resid_glu = (delay_glu_obs - delay_glu_pred) / delay_glu_std_safe
    delay_resid_gly = (delay_gly_obs - delay_gly_pred) / delay_gly_std_safe

    # RNA/P
    if model_name == "original":
        rnap_glu_pred = predict_rnap(model_name, params_glu, lab_RNA_P_glu, time_points, IC50_orig, params_glu["z"])
        rnap_gly_pred = predict_rnap(model_name, params_gly, lab_RNA_P_gly, time_points, IC50_orig, params_gly["z"])
    else:
        rnap_glu_pred = predict_rnap(model_name, params_glu, lab_RNA_P_glu, time_points, IC50_exp_glu, params_glu["z"])
        rnap_gly_pred = predict_rnap(model_name, params_gly, lab_RNA_P_gly, time_points, IC50_exp_gly, params_gly["z"])

    rnap_glu_std_safe = np.where(rnap_glu_std == 0.0, 1e-12, rnap_glu_std)
    rnap_gly_std_safe = np.where(rnap_gly_std == 0.0, 1e-12, rnap_gly_std)

    rnap_resid_glu = (rnap_glu_obs - rnap_glu_pred) / rnap_glu_std_safe
    rnap_resid_gly = (rnap_gly_obs - rnap_gly_pred) / rnap_gly_std_safe

    # Inhibition
    inhib_glu_pred = predict_inhibition(model_name, params_glu["z"], "glucose")
    inhib_gly_pred = predict_inhibition(model_name, params_gly["z"], "glycerol")

    inhib_glu_std_safe = np.where(stdev_glucose == 0.0, 1e-12, stdev_glucose)
    inhib_gly_std_safe = np.where(stdev_glycerol == 0.0, 1e-12, stdev_glycerol)

    inhib_resid_glu = (growth_glucose  - inhib_glu_pred) / inhib_glu_std_safe
    inhib_resid_gly = (growth_glycerol - inhib_gly_pred) / inhib_gly_std_safe

    residuals = np.concatenate([
        delay_resid_glu, delay_resid_gly,
        rnap_resid_glu,  rnap_resid_gly,
        inhib_resid_glu, inhib_resid_gly,
    ])
    return residuals

###############################################################################
# PARAMETER VECTORS FOR EACH MODEL (PLACEHOLDERS)
###############################################################################
# Replace these with your actual fitted parameter vectors.

# Metabolic model: z contains inhibition params [k_off, k_on, P_in, P_out]
z_metabolic_glu = np.array([1.0, 1.0, 1.0, 1.0])  # TODO: replace with fitted params (glucose)
z_metabolic_gly = np.array([1.0, 1.0, 1.0, 1.0])  # TODO: replace with fitted params (glycerol)

# Feedback model: z contains [k_off, k_on, P_in, P_out, alpha_0, alpha_A]
z_feedback_glu = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])  # TODO: replace with fitted params (glucose)
z_feedback_gly = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])  # TODO: replace with fitted params (glycerol)

# Original Greulich inhibition parameters (same for both substrates; literature values)
z_original = np.array([k_off_orig, k_on_orig, P_in_orig, P_out_orig])

# Parameter dictionaries

params_metabolic_glu = {
    "lambda_0": lambda_0_glu,
    "kappa_n": kappa_n_glu,
    "r_min": r_min_corr,
    "r_max": r_max_corr,
    "delta_r": delta_r_corr,
    "kappa_t": kappa_t_corr,
    "lambda_max": lambda_max_corr,
    "k_on": z_metabolic_glu[1],
    "k_off": z_metabolic_glu[0],
    "P_in": z_metabolic_glu[2],
    "P_out": z_metabolic_glu[3],
    "prefactor": prefactor_glu,
    "IC50_lab": lab_IC_50_glu,
    "z": z_metabolic_glu,
}
params_metabolic_gly = {
    "lambda_0": lambda_0_gly,
    "kappa_n": kappa_n_gly,
    "r_min": r_min_corr,
    "r_max": r_max_corr,
    "delta_r": delta_r_corr,
    "kappa_t": kappa_t_corr,
    "lambda_max": lambda_max_corr,
    "k_on": z_metabolic_gly[1],
    "k_off": z_metabolic_gly[0],
    "P_in": z_metabolic_gly[2],
    "P_out": z_metabolic_gly[3],
    "prefactor": prefactor_gly,
    "IC50_lab": lab_IC_50_gly,
    "z": z_metabolic_gly,
}
params_feedback_glu = {
    "lambda_0": lambda_0_glu,
    "kappa_n": kappa_n_glu,
    "r_min": r_min_corr,
    "r_max": r_max_corr,
    "delta_r": delta_r_corr,
    "kappa_t": kappa_t_corr,
    "lambda_max": lambda_max_corr,
    "k_on": z_feedback_glu[1],
    "k_off": z_feedback_glu[0],
    "P_in": z_feedback_glu[2],
    "P_out": z_feedback_glu[3],
    "prefactor": prefactor_glu,
    "IC50_lab": lab_IC_50_glu,
    "alpha_0": z_feedback_glu[4],
    "alpha_A": z_feedback_glu[5],
    "z": z_feedback_glu,
}
params_feedback_gly = {
    "lambda_0": lambda_0_gly,
    "kappa_n": kappa_n_gly,
    "r_min": r_min_corr,
    "r_max": r_max_corr,
    "delta_r": delta_r_corr,
    "kappa_t": kappa_t_corr,
    "lambda_max": lambda_max_corr,
    "k_on": z_feedback_gly[1],
    "k_off": z_feedback_gly[0],
    "P_in": z_feedback_gly[2],
    "P_out": z_feedback_gly[3],
    "prefactor": prefactor_gly,
    "IC50_lab": lab_IC_50_gly,
    "alpha_0": z_feedback_gly[4],
    "alpha_A": z_feedback_gly[5],
    "z": z_feedback_gly,
}
params_original_glu = {
    "lambda_0_model": lambda_0_orig,   # Greulich λ0 (dimensionless)
    "lambda_0_data": lambda_0_glu,     # experimental λ0 for glucose
    "kappa_n": kappa_n_orig,
    "r_min": r_min_orig,
    "r_max": r_max_orig,
    "delta_r": delta_r_orig,
    "kappa_t": kappa_t_orig,
    "k_on": k_on_orig,
    "k_off": k_off_orig,
    "P_in": P_in_orig,
    "P_out": P_out_orig,
    "z": z_original,
}
params_original_gly = {
    "lambda_0_model": lambda_0_orig,   # Greulich λ0 (dimensionless)
    "lambda_0_data": lambda_0_gly,     # experimental λ0 for glycerol
    "kappa_n": kappa_n_orig,
    "r_min": r_min_orig,
    "r_max": r_max_orig,
    "delta_r": delta_r_orig,
    "kappa_t": kappa_t_orig,
    "k_on": k_on_orig,
    "k_off": k_off_orig,
    "P_in": P_in_orig,
    "P_out": P_out_orig,
    "z": z_original,
}

###############################################################################
# PARAMETER COUNTS
###############################################################################

def count_unique_inhib_params(z_glu, z_gly, shared_indices=None):
    if shared_indices is None:
        vals = np.concatenate([z_glu, z_gly])
        return len(np.unique(np.round(vals, 10)))
    shared = z_glu[shared_indices]
    mask = np.ones_like(z_glu, dtype=bool)
    mask[shared_indices] = False
    rest = np.concatenate([z_glu[mask], z_gly[mask]])
    unique_rest = np.unique(np.round(rest, 10))
    return len(shared) + len(unique_rest)

# Metabolic: 4-element z, k_off/k_on shared
k_meta_z = count_unique_inhib_params(z_metabolic_glu, z_metabolic_gly, shared_indices=[0, 1])
# Feedback: 6-element z, k_off/k_on shared
k_fb_z = count_unique_inhib_params(z_feedback_glu, z_feedback_gly, shared_indices=[0, 1])
# Original: 4-element z, k_off/k_on shared
k_orig_z = count_unique_inhib_params(z_original, z_original, shared_indices=[0, 1])

extra_k_metabolic = 0
extra_k_feedback  = 0
extra_k_original  = 0

k_metabolic = k_meta_z + extra_k_metabolic
k_feedback  = k_fb_z   + extra_k_feedback
k_original  = k_orig_z + extra_k_original

###############################################################################
# AIC/BIC
###############################################################################

def aic_bic_from_rss(rss, n, k):
    if rss <= 0:
        raise ValueError("RSS must be positive for AIC/BIC.")
    if n <= k:
        raise ValueError("Need n > k for AIC/BIC.")
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    return aic, bic

models = {
    "original": {
        "name": "Original Greulich-style",
        "params_glu": params_original_glu,
        "params_gly": params_original_gly,
        "k": k_original,
    },
    "metabolic": {
        "name": "Metabolic correction",
        "params_glu": params_metabolic_glu,
        "params_gly": params_metabolic_gly,
        "k": k_metabolic,
    },
    "feedback": {
        "name": "Feedback model",
        "params_glu": params_feedback_glu,
        "params_gly": params_feedback_gly,
        "k": k_feedback,
    },
}

def run_aic_bic_comparison():
    for key, info in models.items():
        residuals = get_all_residuals_for_model(
            model_name=key,
            params_glu=info["params_glu"],
            params_gly=info["params_gly"],
        )
        rss = float(np.sum(residuals**2))
        n = residuals.size
        k = int(info["k"])
        aic, bic = aic_bic_from_rss(rss, n, k)
        info["RSS"] = rss
        info["n"] = n
        info["AIC"] = aic
        info["BIC"] = bic

    min_aic = min(m["AIC"] for m in models.values())
    min_bic = min(m["BIC"] for m in models.values())

    print("Model comparison (delay + RNA/P + inhibition; standardized residuals):\n")
    for key, info in models.items():
        dAIC = info["AIC"] - min_aic
        dBIC = info["BIC"] - min_bic
        print(
            f"{info['name']:22s} | "
            f"k={info['k']:2d}, n={info['n']:2d}, "
            f"RSS={info['RSS']:.3f}, "
            f"AIC={info['AIC']:.2f}, ΔAIC={dAIC:.2f}, "
            f"BIC={info['BIC']:.2f}, ΔBIC={dBIC:.2f}"
        )

def rss_breakdown(model_name, params_glu, params_gly):
    if model_name == "original":
        delay_glu_pred = predict_delay(model_name, params_glu, N_delay_glu, IC50_orig, params_glu["z"])
        delay_gly_pred = predict_delay(model_name, params_gly, N_delay_gly, IC50_orig, params_gly["z"])
    else:
        delay_glu_pred = predict_delay(model_name, params_glu, N_delay_glu, IC50_exp_glu, params_glu["z"])
        delay_gly_pred = predict_delay(model_name, params_gly, N_delay_gly, IC50_exp_gly, params_gly["z"])

    delay_glu_std_safe = np.where(delay_glu_std == 0.0, 1e-12, delay_glu_std)
    delay_gly_std_safe = np.where(delay_gly_std == 0.0, 1e-12, delay_gly_std)

    r_delay_glu = (delay_glu_obs - delay_glu_pred) / delay_glu_std_safe
    r_delay_gly = (delay_gly_obs - delay_gly_pred) / delay_gly_std_safe

    if model_name == "original":
        rnap_glu_pred = predict_rnap(model_name, params_glu, lab_RNA_P_glu, time_points, IC50_orig, params_glu["z"])
        rnap_gly_pred = predict_rnap(model_name, params_gly, lab_RNA_P_gly, time_points, IC50_orig, params_gly["z"])
    else:
        rnap_glu_pred = predict_rnap(model_name, params_glu, lab_RNA_P_glu, time_points, IC50_exp_glu, params_glu["z"])
        rnap_gly_pred = predict_rnap(model_name, params_gly, lab_RNA_P_gly, time_points, IC50_exp_gly, params_gly["z"])

    rnap_glu_std_safe = np.where(rnap_glu_std == 0.0, 1e-12, rnap_glu_std)
    rnap_gly_std_safe = np.where(rnap_gly_std == 0.0, 1e-12, rnap_gly_std)

    r_rnap_glu = (rnap_glu_obs - rnap_glu_pred) / rnap_glu_std_safe
    r_rnap_gly = (rnap_gly_obs - rnap_gly_pred) / rnap_gly_std_safe

    inhib_glu_pred = predict_inhibition(model_name, params_glu["z"], "glucose")
    inhib_gly_pred = predict_inhibition(model_name, params_gly["z"], "glycerol")

    inhib_glu_std_safe = np.where(stdev_glucose == 0.0, 1e-12, stdev_glucose)
    inhib_gly_std_safe = np.where(stdev_glycerol == 0.0, 1e-12, stdev_glycerol)

    r_inhib_glu = (growth_glucose  - inhib_glu_pred) / inhib_glu_std_safe
    r_inhib_gly = (growth_glycerol - inhib_gly_pred) / inhib_gly_std_safe

    rss_delay = np.sum(r_delay_glu**2) + np.sum(r_delay_gly**2)
    rss_rnap  = np.sum(r_rnap_glu**2) + np.sum(r_rnap_gly**2)
    rss_inhib = np.sum(r_inhib_glu**2) + np.sum(r_inhib_gly**2)

    return rss_delay, rss_rnap, rss_inhib

###############################################################################
# AKAKE & BAYESIAN MODEL WEIGHTS
###############################################################################

def compute_akaike_weights(models_dict):
    keys = list(models_dict.keys())
    aics = np.array([models_dict[k]["AIC"] for k in keys], dtype=float)
    min_aic = np.min(aics)
    dAIC = aics - min_aic
    rel_lik = np.exp(-0.5 * dAIC)
    weights = rel_lik / np.sum(rel_lik)

    print("\nAkaike weights (relative model probabilities):")
    for k, w, da in zip(keys, weights, dAIC):
        models_dict[k]["AIC_weight"] = float(w)
        print(f"{models_dict[k]['name']:22s} | ΔAIC={da:7.2f}, weight={w: .4e}")

    return {k: models_dict[k]["AIC_weight"] for k in keys}


def compute_bayes_weights(models_dict):
    keys = list(models_dict.keys())
    bics = np.array([models_dict[k]["BIC"] for k in keys], dtype=float)
    min_bic = np.min(bics)
    dBIC = bics - min_bic
    rel_lik = np.exp(-0.5 * dBIC)
    weights = rel_lik / np.sum(rel_lik)

    print("\nBIC-based model weights (approx. posterior probabilities):")
    for k, w, db in zip(keys, weights, dBIC):
        models_dict[k]["BIC_weight"] = float(w)
        print(f"{models_dict[k]['name']:22s} | ΔBIC={db:7.2f}, weight={w: .4e}")

    return {k: models_dict[k]["BIC_weight"] for k in keys}

if __name__ == "__main__":
    run_aic_bic_comparison()
    akaike_w = compute_akaike_weights(models)
    bic_w    = compute_bayes_weights(models)

    # Optional: RSS breakdown
    for key, info in models.items():
        rd, rr, ri = rss_breakdown(key, info["params_glu"], info["params_gly"])
        print(f"{info['name']}: RSS_delay={rd:.3e}, RSS_RNA/P={rr:.3e}, RSS_inhib={ri:.3e}")


# In[ ]:





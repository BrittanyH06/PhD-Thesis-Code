#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
ProportionalResponseModel_Optimization.py

Purpose
-------
Runs parameter optimization for glucose/glycerol growth under an antibiotic pulse, with dynamic proportional response (final model version).

Usage
-----
1) Provide your experimental values in the DATA PLACEHOLDERS section below.
2) Run from a terminal or Jupyter cell:
       python ProportionalResponseModel_Optimization.py
   or
       %run ProportionalResponseModel_Optimization.py

Data policy
-----------
This file contains NO lab data. Users must insert their own numbers locally.
See thesis Materials and Methods for experimental details and how the values
were obtained.

Outputs
-------
- Printed optimization summary
- Figures saved to ./figures (SVG/PDF)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import math
import random
from scipy.optimize import minimize, curve_fit
from scipy.integrate import cumulative_trapezoid as cumtrapz
from timeit import default_timer as timer
import matplotlib as mpl
from pathlib import Path

start = timer()

# Editable fonts in vector outputs
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

def save_vector(fig, save_name, outdir="figures", formats=("svg", "pdf"), transparent=True):
    """Save a Matplotlib figure in multiple vector formats."""
    Path(outdir).mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(
            Path(outdir) / f"{save_name}.{ext}",
            bbox_inches="tight",
            transparent=transparent
        )

# Define the Vibrant colors
tol_vibrant = [
    "#EE7733",  # orange
    "#0077BB",  # blue
    "#003C5E",  # blue-black
    "#33BBEE",  # cyan
    "#009988",  # teal
    "#004C44",  # teal-black
    "#CC3311",  # red
    "#661A08",  # red-black
    "#EE3377",  # pink
    "#BBBBBB",  # grey
    "#AA4499",  # purple/magenta
    "#44AA99",  # green-teal
    "#FEC44F",  # yellow
    "#7FBC41",  # lime-ish green
    "#8C510A",  # brown
    "#AAAA00",  # olive
]

# Function to blend two colors
def blend_colors(hex1, hex2, ratio=0.5):
    h1 = [int(hex1[i:i+2], 16) for i in (1, 3, 5)]
    h2 = [int(hex2[i:i+2], 16) for i in (1, 3, 5)]
    blended = [round((1 - ratio) * c1 + ratio * c2) for c1, c2 in zip(h1, h2)]
    return '#' + ''.join(f'{c:02X}' for c in blended)

# Blend two vibrant colors (for example: blue + red)
mixed_color = blend_colors(tol_vibrant[3], tol_vibrant[8])  # blue + red

# New custom palette
custom_palette = [
    tol_vibrant[0],  # orange
    tol_vibrant[1],  # blue
    tol_vibrant[2],  # blue-black
    tol_vibrant[3],  # cyan
    tol_vibrant[4],  # teal
    tol_vibrant[5],  # teal-black
    tol_vibrant[6],  # red
    tol_vibrant[7],  # red-black
    tol_vibrant[8],  # pink
    tol_vibrant[9],  # grey
    tol_vibrant[10], # purple/magenta
    tol_vibrant[11], # green-teal
    tol_vibrant[12], # yellow
    tol_vibrant[13], # lime-ish green
    tol_vibrant[14], # brown
    tol_vibrant[15], # olive
    mixed_color      # blended color (cyan + pink = purple)
]

###############################################################################
#                            Data and Model Setup                             #
###############################################################################
r_min = 5.4          # minimal ribosome mass fraction
r_max = 54.4         # maximal ribosome mass fraction
delta_r = r_max - r_min
kappa_t = 0.058      # translational efficiency (h^-1)
lambda_max = kappa_t * delta_r   # maximal growth rate (h^-1)

# Drug-free reference growth rates (h^-1)
lambda_0_glu = 0.64   # for glucose
lambda_0_gly = 0.40   # for glycerol

# Nominal IC50 values (used only if no inhibition curve is supplied)
IC_50_glu = 1.75      # µM
IC_50_gly = 1.45      # µM

# kappa_n values for delay-time calculations
kappa_n_glu = 1 / (delta_r * ((1 / lambda_0_glu) - (1 / lambda_max)))
kappa_n_gly = 1 / (delta_r * ((1 / lambda_0_gly) - (1 / lambda_max)))

###############################################################################
#                       DATA PLACEHOLDERS (FILL IN LOCALLY)                   #
###############################################################################
# The following variables originally contained experimental measurements.
# They are left empty here for public sharing.
# Users should supply their own data in these same formats.

# Pulse lengths (hours)
N_values_glu = []                  # e.g. [2, 4, 5, 6, 8, 10]
N_values_gly = []                  # e.g. [4, 6, 8, 10]

# Delay times (hours) by pulse length
exp_delay_times_glu = {}           # {N_hours: delay_time}
exp_delay_times_gly = {}           # {N_hours: delay_time}

# Standard deviations for delay times (hours)
std_glu = {}                        # {N_hours: std_dev}
std_gly = {}                        # {N_hours: std_dev}

# RNA/Protein ratio time-series for N = 4-h pulse
lab_RNA_P_glu = {}                  # {4: [values at time_points_4H]}
lab_RNA_P_glu_std = {}              # {4: [std_dev at same timepoints]}
lab_RNA_P_gly = {}                  # {4: [values at time_points_4H]}
lab_RNA_P_gly_std = {}              # {4: [std_dev at same timepoints]}

# Time points (hours) corresponding to RNA/P lists (from experiment start)
time_points_4H = []                 # e.g. [0.5, 1.75, 2.5, ...]

# Optional: IC50 values fitted from experimental inhibition curves
# (leave as None unless fitting from provided inhibition data)
lab_IC50_glu = None
lab_IC50_gly = None

###############################################################################
#                        INHIBITION DATA PLACEHOLDERS                         #
###############################################################################
# These were originally filled with experimental inhibition-curve measurements.
# Leave them empty in the public version.  
# Users should supply their own data locally in the same formats.

TET_concentrations = np.array([])      # µM, e.g. [0, 0.4, 0.8, ...]
growth_glucose     = np.array([])      # normalized growth, same length as TET_concentrations
stdev_glucose      = np.array([])      # std deviation of growth_glucose
growth_glycerol    = np.array([])      # normalized growth, same length as TET_concentrations
stdev_glycerol     = np.array([])      # std deviation of growth_glycerol

###############################################################################
#                         Curve-Fit to the Lab Data                           #
###############################################################################
# REQUIREMENT: TET_concentrations, growth_glucose/glycerol (and optional stdevs)
# must be provided above. If you are publishing without inhibition data, leave
# these empty here and fill them locally when reproducing.

def inhibition_model_norm(TET, IC50):
    return 1.0 / (1.0 + (TET / IC50))

initial_guess_norm = [1.0]  # Single parameter initial guess

popt_glu_norm, pcov_glu_norm = curve_fit(
    inhibition_model_norm,
    TET_concentrations,
    growth_glucose,
    sigma=stdev_glucose,  # Using updated stdevs directly
    absolute_sigma=True,
    p0=initial_guess_norm
)
popt_gly_norm, pcov_gly_norm = curve_fit(
    inhibition_model_norm,
    TET_concentrations,
    growth_glycerol,
    sigma=stdev_glycerol,  # Using updated stdevs directly
    absolute_sigma=True,
    p0=initial_guess_norm
)

IC50_exp_glu = popt_glu_norm[0]
IC50_exp_gly = popt_gly_norm[0]

print(f"Experimental IC50 for Glucose: {IC50_exp_glu:.2f}")
print(f"Experimental IC50 for Glycerol: {IC50_exp_gly:.2f}")

# from your curve fit above:
IC50_MINE = {
    "glucose": float(IC50_exp_glu),   # use the fitted IC50s from THIS inhibition curve
    "glycerol": float(IC50_exp_gly),
}

# your pulse experiment settings (same as before)
N_LAB = 4.0
LAB_PULSE_CONC = {
    "glucose": 1.75,   # µM actually used during the N=4 h pulse
    "glycerol": 1.45,
}

# plateau (Greulich) you want to encode:
PREFAC = {sub: N_LAB * LAB_PULSE_CONC[sub] / IC50_MINE[sub] for sub in IC50_MINE}

###############################################################################
#    Helper Function for [k_off, k_on, P_in, P_out] extraction from vector    #
###############################################################################
def extract_inhibition_params(params, substrate):
    L = len(params)
    if L == 10:
        k_off, k_on = params[0], params[1]
        if substrate.lower() == "glucose":
            P_in  = params[2]
            P_out = params[4]
        elif substrate.lower() == "glycerol":
            P_in  = params[3]
            P_out = params[5]
        else:
            raise ValueError("substrate must be 'glucose' or 'glycerol'")
        return [k_off, k_on, P_in, P_out]
    elif L == 6:
        return params[0:4]
    elif L == 4:
        return params
    else:
        raise ValueError(f"Parameter vector must have length 4, 6 or 10, got {L}.")

###############################################################################
#            Functions for Delay Times, RNA/P, and the Model MSE              #
###############################################################################
def calculate_parameters(lambda_0):
    kappa_n = 1 / (delta_r * ((1 / lambda_0) - (1 / (kappa_t * delta_r))))
    return {'lambda_0': lambda_0, 'kappa_n': kappa_n}

parameters_glu = calculate_parameters(lambda_0_glu)
parameters_gly = calculate_parameters(lambda_0_gly)

def model_equations(t, y, N, parameters, IC_50, z):
    """
    ODE system for intracellular antibiotic (a), unbound ribosomes (r_u),
    and bound ribosomes (r_b) under an N-hour pulse. Switches between
    synthesis-limited and dilution-limited regimes; includes binding/unbinding.
    z is the parameter vector [k_off, k_on, P_in, P_out, ...] as used here.
    """
    k_off, k_on, P_in, P_out, alpha_0, alpha_A = z
    lambda_0 = parameters['lambda_0']
    kappa_n = parameters['kappa_n']
    a, r_u, r_b = y

    t_on = 1
    t_off = t_on+N

    x_syn = (r_u - r_min) * kappa_t
    x_dil = (r_max - r_u - r_b) * kappa_n 
    x     = min(x_syn, x_dil)

    if t < t_on:
        s_ss0 = lambda_0 * (r_max - (lambda_0 * delta_r * ((1 / lambda_0) - (1 / lambda_max))))
        s = s_ss0
        a_ex = 0
    elif t <= t_off:
        #lambda_f = lambda_0 / (1 + (4 / N))
        if np.isclose(parameters['lambda_0'], lambda_0_glu):
            sub = "glucose"
        else:
            sub = "glycerol"
        
        pref = PREFAC[sub]
        ic50_mine = IC50_MINE[sub]
        
        # use YOUR IC50 fit: a_ex = (prefactor * IC50_mine)/N
        a_ex = (pref * ic50_mine) / N
        
        # and the corresponding λ_f drop
        lambda_f = lambda_0 / (1.0 + pref / N)
        s_ssA = lambda_f * (r_max - (lambda_f / kappa_n))
        s = s_ssA * (1 + (alpha_A * (x_dil-x_syn)))
        #a_ex = (4.0 * LAB_PULSE_CONC[substrate.lower()]) / N
    else:
        s_ss0 = lambda_0 * (r_max - (lambda_0 * delta_r * ((1 / lambda_0) - (1 / lambda_max))))
        s = s_ss0 * (1 + (alpha_0 * (x_dil-x_syn)))
        a_ex = 0

    F = (k_on * a * (r_u - r_min)) - (k_off * r_b)
    
    da_dt   = -F - x * a + P_in * a_ex - P_out * a
    dr_u_dt = -F - x * r_u + s
    dr_b_dt = F - x * r_b

    return [da_dt, dr_u_dt, dr_b_dt]

def calculate_delay_time(parameters, N_values, IC_50, z):
    """
    For each pulse length N, integrates the ODEs and computes Δt (delay time)
    from cumulative growth vs time. Returns a list of (N, Δt).
    """
    lambda_0 = parameters['lambda_0']
    kappa_n = parameters['kappa_n']
    results = []
    for N in N_values:
        tMax = (N + 1) + 4 + (30 / lambda_0)
        time = np.linspace(0, tMax, int(200 * tMax))
        
        y0 = [0, r_min + (lambda_0 / kappa_t), 0]
        sol = solve_ivp(lambda t, y: model_equations(t, y, N, parameters, IC_50, z),
                        [time[0], time[-1]], y0, method='BDF', t_eval=time)
        gr1 = (sol.y[1] - r_min) * kappa_t
        gr2 = (r_max - sol.y[1] - sol.y[2]) * kappa_n
        growth_rate_sum = np.minimum(gr1, gr2)

        OD = np.cumsum(growth_rate_sum) * (time[1] - time[0])
        delay_time = ((OD[1] - OD[-1]) / lambda_0) + (time[-1] - time[1])
        results.append((N, delay_time))
    return results

def calculate_mse(delay_times_opt, exp_delay_times, stds):
    length = len(delay_times_opt)
    sum_sq = 0
    for (N, dt) in delay_times_opt:
        if N not in exp_delay_times:
            continue
        denom = max(0.05, stds.get(N, 0.05))
        sum_sq += ((exp_delay_times[N] - dt)**2) / (denom**2)
    return sum_sq / length

def calculate_r_total_normalized(lab_RNA_P, time_points, parameters, IC_50, z):
    """
    Simulates total ribosome signal r_total over time for each N in lab_RNA_P,
    normalizes by the value near t=0.5 h, and returns the series aligned to
    the provided time_points. Requires lab_RNA_P and _std structures to be filled.
    """
    lambda_0 = parameters['lambda_0']
    y0 = [0, r_min + (lambda_0 / kappa_t), 0]
    simulated_r_total = []
    for N in lab_RNA_P:
        tMax = (N + 1) + 4 + (30 / lambda_0)
        time = np.linspace(0, tMax, int(200 * tMax))
        sol_rtot = solve_ivp(lambda t, y: model_equations(t, y, N, parameters, IC_50, z),
                             [time[0], time[-1]], y0, method='BDF', t_eval=time)
        r_total = sol_rtot.y[1] + sol_rtot.y[2]
        idx_ref = np.abs(sol_rtot.t - 0.5).argmin()
        denom = r_total[idx_ref] if r_total[idx_ref] != 0 else 1e-12
        r_total_norm = r_total / denom
        for t_pt in time_points:
            idx = np.abs(sol_rtot.t - t_pt).argmin()
            simulated_r_total.append(r_total_norm[idx])
    return simulated_r_total

def mse_RNA_P(lab_data, lab_stds, simulated_r_total):
    sum_sq = 0
    for i in range(len(lab_data)):
        denom = lab_stds[i] if lab_stds[i] > 0 else 0.05
        sum_sq += ((lab_data[i] - simulated_r_total[i])**2) / (denom**2)
    return sum_sq / len(lab_data)

###############################################################################
#           Inhibition Curve: Model Predictions via Cubic Equation            #
###############################################################################
def solve_lambda_cubic(a_ex, lambda_0, k_off, k_on, P_in, P_out):
    """
    Solve the cubic equation for the normalized growth rate, x = λ/λ₀, using Cardano's formula.
    
    The cubic is assumed to be:
        x³ - x² + (α + β)x - α = 0,
    where:
        α = 0.25 * (λ₀* / λ₀)²,
        β = (a_ex / (2*IC50*)) * (λ₀* / λ₀),
    with:
        λ₀* = 2 * sqrt(P_out * κₜ * (k_off/k_on))
        IC50* = (λ₀* * δ_r) / (2 * P_in)
    
    For a_ex near zero, return x = 1.
    """
    if abs(a_ex) < 1e-12:
        return 1.0  # normalized growth equals 1 when no drug is present

    lambda0_star = 2.0 * np.sqrt(P_out * kappa_t * (k_off / k_on))
    IC50_star = (lambda0_star * delta_r) / (2.0 * P_in)

    a = 0.25 * (lambda0_star / lambda_0)**2
    b = (a_ex / (2.0 * IC50_star)) * (lambda0_star / lambda_0)

    # Cubic: x³ - x² + (a+b)x - a = 0
    A = -1.0
    B = a + b
    C = -a

    shift = 1.0/3.0
    p = B - (A**2)/3.0    # p = (a+b) - 1/3
    q = (2.0*(A**3))/27.0 - (A*B)/3.0 + C  # for A = -1, q = (-2/27) + ((a+b)/3) - a

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

def simulate_inhibition_curve_cubic(full_params, substrate, normalize=False):
    # Extract only the inhibition parameters.
    inhib_params = extract_inhibition_params(full_params, substrate)
    k_off, k_on, P_in, P_out = inhib_params
    lambda0 = lambda_0_glu if substrate.lower() == "glucose" else lambda_0_gly
    predicted = []
    for TET in TET_concentrations:
        x = solve_lambda_cubic(TET, lambda0, k_off, k_on, P_in, P_out)
        if x is None or x < 0:
            predicted.append(0.0)
        else:
            predicted.append(x if normalize else x * lambda0)
    return np.array(predicted)

###############################################################################
#       Helper Functions for Model-Predicted IC50 using Newton's Method       #
###############################################################################
def model_normalized_growth(TET, full_params, substrate):
    inhib_params = extract_inhibition_params(full_params, substrate)
    k_off, k_on, P_in, P_out = inhib_params
    lambda0 = lambda_0_glu if substrate.lower() == "glucose" else lambda_0_gly
    x = solve_lambda_cubic(TET, lambda0, k_off, k_on, P_in, P_out)
    return x if (x is not None and x >= 0) else 0.0

def get_model_IC50(full_params, substrate):
    """
    Analytic IC₅₀: no inner root‐finding, just the closed‐form formula.
    full_params is your 8‐element vector:
      [k_off, k_on, P_in, P_out, alpha_0, alpha_A]
    """
    # extract only the inhibition params
    k_off, k_on, P_in, P_out = full_params[0], full_params[1], full_params[2], full_params[3]

    # pick the right drug‐free growth rate
    if substrate.lower() == "glucose":
        lam_0 = lambda_0_glu
    else:
        lam_0 = lambda_0_gly

    # derived quantities
    K_D = k_off / k_on
    lambda_0_star = 2.0 * np.sqrt(P_out * kappa_t * K_D)
    IC_50_star = (delta_r * lambda_0_star) / (2.0 * P_in)

    # final closed‐form
    return 0.5 * IC_50_star * ((lam_0 / lambda_0_star) + (lambda_0_star / lam_0))

###############################################################################
#             Inhibition MSE for the Optimization Objective Function          #
###############################################################################
def inhib_curve_mse(full_params, substrate):
    """
    Mean squared error between model-predicted normalized growth and the
    provided inhibition curve for the given substrate. Requires the
    TET_concentrations/growth_* arrays to be filled.
    """
    pred = simulate_inhibition_curve_cubic(full_params, substrate, normalize=True)
    if substrate.lower() == "glucose":
        obs = growth_glucose
        stdev_arr = stdev_glucose
    elif substrate.lower() == "glycerol":
        obs = growth_glycerol
        stdev_arr = stdev_glycerol
    else:
        raise ValueError("Unknown substrate")
    sum_sq = 0.0
    n = len(TET_concentrations)
    for i in range(n):
        sum_sq += ((obs[i] - pred[i])**2) / (stdev_arr[i]**2)
    return sum_sq / n

def objective(params):
    """
    Combined cost:
      delay (glucose) + RNA/P (glucose)
    + delay (glycerol) + RNA/P (glycerol)
    + weighted inhibition-curve error.
    Uses the data structures defined in the DATA PLACEHOLDERS section.
    """
    (k_off, k_on,
     P_in_GLU,  P_in_GLY,
     P_out_GLU, P_out_GLY,
     alpha_0_GLU, alpha_A_GLU,
     alpha_0_GLY, alpha_A_GLY) = params

    current_z_glu = [k_off, k_on, P_in_GLU, P_out_GLU, alpha_0_GLU, alpha_A_GLU]
    current_z_gly = [k_off, k_on, P_in_GLY, P_out_GLY, alpha_0_GLY, alpha_A_GLY]

    model_IC_50_glu = get_model_IC50(current_z_glu, "glucose")
    model_IC_50_gly = get_model_IC50(current_z_gly, "glycerol")
    
    delay_times_glu = calculate_delay_time(parameters_glu, N_values_glu, model_IC_50_glu, current_z_glu)
    delay_times_gly = calculate_delay_time(parameters_gly, N_values_gly, model_IC_50_gly, current_z_gly)
    
    mse_glu_delay = calculate_mse(delay_times_glu, exp_delay_times_glu, std_glu)
    mse_gly_delay = calculate_mse(delay_times_gly, exp_delay_times_gly, std_gly)
    
    sim_rna_glu = calculate_r_total_normalized(lab_RNA_P_glu, time_points_4H, parameters_glu, model_IC_50_glu, current_z_glu)
    sim_rna_gly = calculate_r_total_normalized(lab_RNA_P_gly, time_points_4H, parameters_gly, model_IC_50_gly, current_z_gly)
    
    mse_glu_rnap = mse_RNA_P(lab_RNA_P_glu[4], lab_RNA_P_glu_std[4], sim_rna_glu)
    mse_gly_rnap = mse_RNA_P(lab_RNA_P_gly[4], lab_RNA_P_gly_std[4], sim_rna_gly)
    
    mse_inhib = inhib_curve_mse(current_z_glu, "glucose") + inhib_curve_mse(current_z_gly, "glycerol")
    
    combined = ((25*mse_glu_delay) + mse_glu_rnap) + ((50*mse_gly_delay) + (mse_gly_rnap*25)) + (mse_inhib*50)
    
    return combined

###############################################################################
#                               Run Optimization                              #
###############################################################################
# NOTE:
# The optimization below assumes you have populated the DATA PLACEHOLDERS.
# If any of those structures are empty, fill them locally before running.
# This public file intentionally contains no lab values.

bounds = [
    (1e2, 1e7),    # k_off
    (1e2, 1e7),    # k_on
    (1e1, 1e4),    # P_in_GLU
    (1e1, 1e4),    # P_in_GLY
    (1e1, 1e4),    # P_out_GLU
    (1e1, 1e4),    # P_out_GLY
    (0.05, 15.0),     # alpha_0_GLU
    (0.05, 15.0),     # alpha_A_GLU
    (0.05, 20.0),     # alpha_0_GLY
    (0.05, 20.0)      # alpha_A_GLY
]

initial_params = [17761.16797884993, 1844.5633655257093, 1991.6679966679635, 2183.466400139361, 46.80349238571554, 40.75594409939771, 1.0, 0.5, 1.0, 0.5]
#Optimals from Mar2025 + geometric mean of bounds for synthesis parameters

print("\nOriginal Initial Guess:", initial_params)

percent_shift = 0.05
shifted_guess = [val + random.gauss(0, percent_shift*val) for val in initial_params]
print(f"Random-Shifted Initial Guess (percent shift: {percent_shift*100:.0f}%):", shifted_guess)

#result = minimize(objective, initial_params, method='Nelder-Mead', bounds=bounds)
result = minimize(objective, shifted_guess, method='Nelder-Mead', bounds=bounds)
print("\n================= Minimization Result =================")
print(result)
print("======================================================\n")
optimals_glu = [result.x[0], result.x[1], result.x[2], result.x[4]]
optimals_gly = [result.x[0], result.x[1], result.x[3], result.x[5]]
print("Optimal Parameters Glucose (k_off, k_on, P_in, P_out):", optimals_glu)
print("Optimal Parameters Glycerol (k_off, k_on, P_in, P_out):", optimals_gly)
print("Smallest Combined MSE:", result.fun)
print()

# ─── after minimize(...) ─────────────────────────────────────────────────────

# grab optimized parameters
lambda_0_glu = parameters_glu['lambda_0']
kappa_n_glu  = parameters_glu['kappa_n']
lambda_0_gly = parameters_gly['lambda_0']
kappa_n_gly  = parameters_gly['kappa_n']

# pre‐pulse anchor (same for all N)
s_ss0_glu = lambda_0_glu * (r_max - lambda_0_glu*delta_r*((1/lambda_0_glu)-(1/(kappa_t*delta_r))))
s_ss0_gly = lambda_0_gly * (r_max - lambda_0_gly*delta_r*((1/lambda_0_gly)-(1/(kappa_t*delta_r))))

print("\nGlucose synthesis‐rate anchors:")
for N in N_values_glu:
    λ_f   = lambda_0_glu / (1 + 4.0/N)
    s_ssA = λ_f * (r_max - λ_f/kappa_n_glu)
    print(f"  N={N:>2} h → s_ss0 = {s_ss0_glu:.3g},   s_ssA = {s_ssA:.3g}")

print("\nGlycerol synthesis‐rate anchors:")
for N in N_values_gly:
    λ_f   = lambda_0_gly / (1 + 4.0/N)
    s_ssA = λ_f * (r_max - λ_f/kappa_n_gly)
    print(f"  N={N:>2} h → s_ss0 = {s_ss0_gly:.3g},   s_ssA = {s_ssA:.3g}")

# now grab your two new α’s
alpha0_GLU, alphaA_GLU = result.x[6], result.x[7]
alpha0_GLY, alphaA_GLY = result.x[8], result.x[9]

print("\nOptimized Glucose synthesis‐strategy:")
print(f"  • During pulse: s = s_ssA(N) * (1 + {alphaA_GLU:.3g}·(λ_N–λ_T))")
print(f"  • Post‐pulse:   s = s_ss0 *   (1 + {alpha0_GLU:.3g}·(λ_N–λ_T))")

print("\nOptimized Glycerol synthesis‐strategy:")
print(f"  • During pulse: s = s_ssA(N) * (1 + {alphaA_GLY:.3g}·(λ_N–λ_T))")
print(f"  • Post‐pulse:   s = s_ss0 *   (1 + {alpha0_GLY:.3g}·(λ_N–λ_T))")
print()

###############################################################################
#     Functions to compute & print delay comparisons and other differences    #
###############################################################################
def get_model_delay_at_N(parameters, IC_50, z, N):
    dt_list = calculate_delay_time(parameters, [N], IC_50, z)
    return dt_list[0][1] if dt_list else np.nan

def compute_experimental_delay_max(exp_delay_dict):
    """
    Returns (max_delay_value, N_at_max).
    """
    N_at_max = max(exp_delay_dict, key=exp_delay_dict.get)
    return exp_delay_dict[N_at_max], N_at_max

def get_max_delay(parameters, IC_50, z, N_list):
    # Returns (max_delay, N_at_max)
    delays = calculate_delay_time(parameters, N_list, IC_50, z)
    max_pair = max(delays, key=lambda x: x[1])
    return max_pair[1], max_pair[0]

def get_max_RNA_P(parameters, IC_50, z, lab_time_points):
    # Solve model for time_points but up to 10 hours if we want max
    # We'll just do an array from 0 -> 10 in small steps:
    times_for_max = np.linspace(0, 10, 200)  # up to 10 h
    # For the function's sake, we'll pass N=4 so it solves that scenario
    # (lab_RNA_P dict uses key=4)
    y0 = [0, r_min + (parameters['lambda_0'] / kappa_t), 0]
    sol = solve_ivp(
        lambda t, y: model_equations(t, y, 4, parameters, IC_50, z),
        [0, 10],
        y0,
        method='BDF',
        t_eval=times_for_max
    )
    r_total = sol.y[1] + sol.y[2]
    # normalize by the value at t=0.5 (if present)
    idx_ref = np.abs(times_for_max - 0.5).argmin()
    denom = r_total[idx_ref] if r_total[idx_ref] != 0 else 1e-12
    r_norm = r_total / denom
    
    return np.max(r_norm)

def get_max_RNA_P_experimental(lab_RNA_P_dict):
    # Only N=4 in the data. Just pick max across its times
    return max(lab_RNA_P_dict[4])

###############################################################################
#                          Print the Summary Table                            #
###############################################################################
# Comment out shifted_guess vectors if not using Gaussian random perturbation on initial guess
# (i.e. using original initial guess)

# original (pre-shift) guesses, 6 per substrate:
# orig_params_glu_full = [
#     initial_params[0],  # k_off
#     initial_params[1],  # k_on
#     initial_params[2],  # P_in_glu
#     initial_params[4],  # P_out_glu
#     initial_params[6],  # alpha_0_GLU
#     initial_params[7],  # alpha_A_GLU
# ]
# orig_params_gly_full = [
#     initial_params[0],  # k_off
#     initial_params[1],  # k_on
#     initial_params[3],  # P_in_gly
#     initial_params[5],  # P_out_gly
#     initial_params[8],  # alpha_0_GLY
#     initial_params[9],  # alpha_A_GLY
# ]

# if you’re using shifted_guess instead of initial_params:
orig_params_glu_full = [
    shifted_guess[0],
    shifted_guess[1],
    shifted_guess[2],
    shifted_guess[4],
    shifted_guess[6],
    shifted_guess[7],
]
orig_params_gly_full = [
    shifted_guess[0],
    shifted_guess[1],
    shifted_guess[3],
    shifted_guess[5],
    shifted_guess[8],
    shifted_guess[9],
]

# optimized result.x is length-10 in the order
# [k_off, k_on, P_in_GLU, P_in_GLY, P_out_GLU, P_out_GLY,
#  alpha_0_GLU, alpha_A_GLU, alpha_0_GLY, alpha_A_GLY]
optimals_glu_full = [
    result.x[0],  # k_off
    result.x[1],  # k_on
    result.x[2],  # P_in_GLU
    result.x[4],  # P_out_GLU
    result.x[6],  # alpha_0_GLU
    result.x[7],  # alpha_A_GLU
]
optimals_gly_full = [
    result.x[0],  # k_off
    result.x[1],  # k_on
    result.x[3],  # P_in_GLY
    result.x[5],  # P_out_GLY
    result.x[8],  # alpha_0_GLY
    result.x[9],  # alpha_A_GLY
]

# Now get IC50 values using the full vectors:
orig_model_IC50_glu = get_model_IC50(orig_params_glu_full, "glucose")
opt_model_IC50_glu  = get_model_IC50(optimals_glu_full, "glucose")
orig_model_IC50_gly = get_model_IC50(orig_params_gly_full, "glycerol")
opt_model_IC50_gly  = get_model_IC50(optimals_gly_full, "glycerol")

IC50m_glu = get_model_IC50(optimals_glu_full, "glucose")
IC50m_gly = get_model_IC50(optimals_gly_full, "glycerol")
print("IC50_model (glu) =", IC50m_glu, "=> expected plateau ≈", 4*LAB_PULSE_CONC["glucose"]/IC50m_glu, "h")
print("IC50_model (gly) =", IC50m_gly, "=> expected plateau ≈", 4*LAB_PULSE_CONC["glycerol"]/IC50m_gly, "h")

for sub in ["glucose", "glycerol"]:
    print(f"{sub}: IC50_mine={IC50_MINE[sub]:.3g} µM, "
          f"C_pulse={LAB_PULSE_CONC[sub]:.3g} µM, "
          f"expected plateau={PREFAC[sub]:.2f} h")

for sub, params, z in [("glucose", parameters_glu, optimals_glu_full),
                       ("glycerol", parameters_gly, optimals_gly_full)]:
    dt_long = calculate_delay_time(params, [100], IC50_MINE[sub], z)[0][1]
    print(f"[Sanity] {sub}: expected≈{PREFAC[sub]:.2f} h, simulated (N=100)≈{dt_long:.2f} h")

# (1) Delay: Original vs. Optimized
max_delay_glu_orig, N_glu_origmax = get_max_delay(parameters_glu, orig_params_glu_full, N_values_glu, "glucose")
max_delay_glu_opt,  N_glu_optmax  = get_max_delay(parameters_glu, optimals_glu_full, N_values_glu, "glucose")
max_delay_gly_orig, N_gly_origmax = get_max_delay(parameters_gly, orig_params_gly_full, N_values_gly, "glycerol")
max_delay_gly_opt,  N_gly_optmax  = get_max_delay(parameters_gly, optimals_gly_full, N_values_gly, "glycerol")

print("Delay Time (Glucose) - from initial guess vs. optimized:")
print(f"Original max model delay = {max_delay_glu_orig:.3f} (from initial guess), at N={N_glu_origmax}")
print(f"Optimal max model delay = {max_delay_glu_opt:.3f}, at N={N_glu_optmax}\n")

print("Delay Time (Glycerol) - from initial guess vs. optimized:")
print(f"Original max model delay = {max_delay_gly_orig:.3f} (from initial guess), at N={N_gly_origmax}")
print(f"Optimal max model delay = {max_delay_gly_opt:.3f}, at N={N_gly_optmax}\n")

# (1a) Per-N Delay Comparison for Glucose
print("---- Per-N Delay Time Comparison: Glucose ----")
for N in sorted(N_values_glu):
    dt_model = get_model_delay_at_N(parameters_glu, optimals_glu_full, N, "glucose")
    dt_exp = exp_delay_times_glu[N]
    diff = dt_exp - dt_model
    print(f"N={N}: Lab={dt_exp:.2f}, Model={dt_model:.2f}, (Lab-Model)={diff:.2f}")
print()

# (1b) Per-N Delay Comparison for Glycerol
print("---- Per-N Delay Time Comparison: Glycerol ----")
for N in sorted(N_values_gly):
    dt_model = get_model_delay_at_N(parameters_gly, optimals_gly_full, N, "glycerol")
    dt_exp = exp_delay_times_gly[N]
    diff = dt_exp - dt_model
    print(f"N={N}: Lab={dt_exp:.2f}, Model={dt_model:.2f}, (Lab-Model)={diff:.2f}")
print()

# (1c) Compare maximum delay at the lab maximum N
lab_max_delay_glu, lab_N_glu = compute_experimental_delay_max(exp_delay_times_glu)
mdl_delay_at_labmax_glu = get_model_delay_at_N(parameters_glu, optimals_glu_full, lab_N_glu, "glucose")

lab_max_delay_gly, lab_N_gly = compute_experimental_delay_max(exp_delay_times_gly)
mdl_delay_at_labmax_gly = get_model_delay_at_N(parameters_gly, optimals_gly_full, lab_N_gly, "glycerol")

print("At the N where Lab Delay is max for Glucose:")
print(f"Lab max delay = {lab_max_delay_glu:.2f} (N={lab_N_glu}), Model={mdl_delay_at_labmax_glu:.2f}, Diff={lab_max_delay_glu - mdl_delay_at_labmax_glu:.2f}")
print("At the N where Lab Delay is max for Glycerol:")
print(f"Lab max delay = {lab_max_delay_gly:.2f} (N={lab_N_gly}), Model={mdl_delay_at_labmax_gly:.2f}, Diff={lab_max_delay_gly - mdl_delay_at_labmax_gly:.2f}\n")

# (2) RNA/P Comparisons
max_RNA_P_glu_orig = get_max_RNA_P(parameters_glu, orig_params_glu_full, time_points_4H, "glucose")
max_RNA_P_glu_opt  = get_max_RNA_P(parameters_glu, optimals_glu_full, time_points_4H, "glucose")
max_RNA_P_gly_orig = get_max_RNA_P(parameters_gly, orig_params_gly_full, time_points_4H, "glycerol")
max_RNA_P_gly_opt  = get_max_RNA_P(parameters_gly, optimals_gly_full, time_points_4H, "glycerol")

print("RNA/P (Glucose) - from initial guess vs. optimized:")
print(f"Original max model = {max_RNA_P_glu_orig:.3f}, Optimal max model = {max_RNA_P_glu_opt:.3f}")
print("RNA/P (Glycerol) - from initial guess vs. optimized:")
print(f"Original max model = {max_RNA_P_gly_orig:.3f}, Optimal max model = {max_RNA_P_gly_opt:.3f}\n")

lab_max_rnap_glu = get_max_RNA_P_experimental(lab_RNA_P_glu)
lab_max_rnap_gly = get_max_RNA_P_experimental(lab_RNA_P_gly)
model_max_rnap_glu = get_max_RNA_P(parameters_glu, optimals_glu_full, time_points_4H, "glucose")
model_max_rnap_gly = get_max_RNA_P(parameters_gly, optimals_gly_full, time_points_4H, "glycerol")

print("Max RNA/P (Glucose) - lab vs. model at N=4 scenario:")
print(f"Lab max = {lab_max_rnap_glu:.3f}, Model max = {model_max_rnap_glu:.3f}, Diff={lab_max_rnap_glu - model_max_rnap_glu:.3f}")
print("Max RNA/P (Glycerol) - lab vs. model at N=4 scenario:")
print(f"Lab max = {lab_max_rnap_gly:.3f}, Model max = {model_max_rnap_gly:.3f}, Diff={lab_max_rnap_gly - model_max_rnap_gly:.3f}\n")

print("\n========== IC50 Comparisons ==========")
model_IC50_glu_orig = get_model_IC50(orig_params_glu_full, "glucose")
model_IC50_gly_orig = get_model_IC50(orig_params_gly_full, "glycerol")
model_IC50_glu_opt  = get_model_IC50(optimals_glu_full, "glucose")
model_IC50_gly_opt  = get_model_IC50(optimals_gly_full, "glycerol")

print()
print(f"Glucose Model IC_50 (with optimal parameters): {opt_model_IC50_glu}")
print(f"Glycerol Model IC_50 (with optimal parameters): {opt_model_IC50_gly}")

def dt_infty_at(parameters, z, IC50_model, N_big=100):
    return calculate_delay_time(parameters, [N_big], IC50_model, z)[0][1]

for sub, params, z in [
    ("glucose",  parameters_glu, optimals_glu_full),
    ("glycerol", parameters_gly, optimals_gly_full),
]:
    IC50_model = get_model_IC50(z, sub)
    aex1 = N_LAB * LAB_PULSE_CONC[sub]  # same definition as in your ODE
    dt_inf = dt_infty_at(params, z, IC50_model, N_big=100)

    rhs_model = aex1 / IC50_model
    rhs_lab   = aex1 / IC50_MINE[sub]   # this is your "sanity" plateau

    print(f"\n[{sub.upper()}]")
    print(f"IC50_model = {IC50_model:.6g} µM")
    print(f"Δt_∞(N=100)        ≈ {dt_inf:.6g} h")
    print(f"a_ex(1)/IC50_model = {rhs_model:.6g} h")
    print(f"a_ex(1)/IC50_lab   = {rhs_lab:.6g} h")

print("IC50 differences from Lab (Glucose):")
if not np.isnan(model_IC50_glu_orig):
    print(f"Original guess vs Lab: ({model_IC50_glu_orig:.2f} - {IC50_exp_glu:.2f}) = {model_IC50_glu_orig - IC50_exp_glu:.2f}")
if not np.isnan(model_IC50_glu_opt):
    print(f"Optimized vs Lab: ({model_IC50_glu_opt:.2f} - {IC50_exp_glu:.2f}) = {model_IC50_glu_opt - IC50_exp_glu:.2f}")

print("IC50 differences from Lab (Glycerol):")
if not np.isnan(model_IC50_gly_orig):
    print(f"Original guess vs Lab: ({model_IC50_gly_orig:.2f} - {IC50_exp_gly:.2f}) = {model_IC50_gly_orig - IC50_exp_gly:.2f}")
if not np.isnan(model_IC50_gly_opt):
    print(f"Optimized vs Lab: ({model_IC50_gly_opt:.2f} - {IC50_exp_gly:.2f}) = {model_IC50_gly_opt - IC50_exp_gly:.2f}")
print("=========================================================\n")

print("\n========== Normalized Growth Rates Comparisons ==========")
model_growth_glu = simulate_inhibition_curve_cubic(optimals_glu_full, "glucose", normalize=True)
model_growth_gly = simulate_inhibition_curve_cubic(optimals_gly_full, "glycerol", normalize=True)

print("Glucose:")
for tet, exp_val, mod_val in zip(TET_concentrations, growth_glucose, model_growth_glu):
    diff = exp_val - mod_val
    print(f"TET = {tet:.2f} μM: Experimental = {exp_val:.2f}, Model = {mod_val:.2f}, Diff = {diff:.2f}")
print("Glycerol:")
for tet, exp_val, mod_val in zip(TET_concentrations, growth_glycerol, model_growth_gly):
    diff = exp_val - mod_val
    print(f"TET = {tet:.2f} μM: Experimental = {exp_val:.2f}, Model = {mod_val:.2f}, Diff = {diff:.2f}")
print("=========================================================\n")

###############################################################################
#        Helper function to compute lambda0* for a given carbon source        #
###############################################################################
def compute_lambda0_star(params, substrate):
    # Here, assume params is a 4-element vector; use the helper to extract if needed.
    inhib_params = extract_inhibition_params(params, substrate)
    k_off, k_on, P_in, P_out = inhib_params
    return 2.0 * np.sqrt(P_out * kappa_t * (k_off / k_on))

###############################################################################
#                         Plotting / Summary Functions                        #
###############################################################################
def analyze_growth(lambda_0, N_values, parameters, optimals, substrate_name, save_name=None):
    # Compute the model IC50 using get_model_IC50 (for a 4-element vector)
    model_IC50 = get_model_IC50(optimals, substrate_name.lower())
    #model_IC50 = IC50_MINE[substrate_name.lower()]
    plt.figure(figsize=(10,6))
    time_end = max(N_values) + 1 + 4 + (30 / lambda_0)
    time = np.linspace(0, time_end, int(200 * time_end))
    for N in N_values:
        y0 = [0, r_min + (lambda_0 / kappa_t), 0]
        sol_plot = solve_ivp(lambda t, y: model_equations(t, y, N, parameters, model_IC50, optimals),
                             [time[0], time[-1]], y0, method='BDF', t_eval=time)
        gr1 = (sol_plot.y[1] - r_min) * kappa_t
        gr2 = (r_max - sol_plot.y[1] - sol_plot.y[2]) * parameters['kappa_n']
        growth_rate = np.minimum(gr1, gr2)
        plt.plot(time, growth_rate, label=f"T={N}")
    plt.xlabel("Time (h)", fontsize=14)
    plt.ylabel(r"Growth Rate $\lambda$ (h$^{-1}$)", fontsize=14, labelpad=20)
    plt.title(f"Time vs. Growth Rate for {substrate_name}", fontsize=16)
    #plt.axhline(y=lambda_0, color='black', linestyle='--', linewidth=1.2, label=r'$\lambda_0$')
    plt.xlim(0, 60)
    plt.ylim(0)
    plt.grid(True)
    plt.legend()

    fig = plt.gcf()
    if save_name:
        save_vector(fig, save_name)
        
    plt.show()

def plot_inhibition_curve_glu(params, save_name=None):
    TET_range = np.linspace(0, 2.0, 100)
    ic50_exp   = popt_glu_norm[0]
    lab_fit    = inhibition_model_norm(TET_range, ic50_exp)
    ic50_model = get_model_IC50(params, "glucose")

    # compute lambda0_star & ratio exactly as before
    lambda0_star = compute_lambda0_star(params, "glucose")
    ratio        = lambda0_star / lambda_0_glu

    print(f"Glucose: Experimental IC$_{{50}}$ = {ic50_exp:.2f}, Model IC$_{{50}}$ = {ic50_model:.2f}")
    print(f"lambda0* = {lambda0_star:.2f}, lambda0*/lambda0 = {ratio:.2f}")

    # your discrete predictions
    model_pred = simulate_inhibition_curve_cubic(params, "glucose", normalize=True)

    # ─── NEW: re-solve the cubic on the fine grid for a smooth curve ───
    k_off, k_on, P_in, P_out = extract_inhibition_params(params, "glucose")
    model_smooth = np.array([
        solve_lambda_cubic(t, lambda_0_glu, k_off, k_on, P_in, P_out)
        for t in TET_range
    ])

    plt.figure(figsize=(8,6))

    # experimental data + fit
    plt.errorbar(TET_concentrations, growth_glucose, yerr=stdev_glucose,
                 fmt='^', capsize=5, color=custom_palette[1], label="Glucose Experimental Data")
    plt.plot(TET_range, lab_fit, '-', color=custom_palette[1],
             label=f"Experimental IC$_{{50}}$ = {ic50_exp:.2f}")

    # ─── NEW: smooth dashed model curve + discrete markers ───
    plt.plot(TET_range, model_smooth,
             linestyle='--', color=custom_palette[2],
             label=f"Model IC$_{{50}}$ = {ic50_model:.2f}")
    plt.plot(TET_concentrations, model_pred,
             marker='x', linestyle='None', color=custom_palette[2])

    # rest unchanged
    plt.xlabel(r"Tetracycline Concentration ($\mu$M)", fontsize=14)
    plt.ylabel(r"Normalized Growth Rate $\frac{\lambda}{\lambda_0}$", fontsize=14, labelpad=20)
    plt.title("Inhibition Curve - Glucose", fontsize=16)
    plt.xticks(np.arange(0, 2.1, 0.4))
    plt.ylim(ymin=0)
    plt.legend()
    plt.grid(True)

    fig = plt.gcf()
    if save_name:
        save_vector(fig, save_name)
        
    plt.show()

def plot_inhibition_curve_gly(params, save_name=None):
    TET_range = np.linspace(0, 2.0, 100)
    ic50_exp   = popt_gly_norm[0]
    lab_fit    = inhibition_model_norm(TET_range, ic50_exp)
    ic50_model = get_model_IC50(params, "glycerol")

    # compute lambda0_star & ratio exactly as before
    lambda0_star = compute_lambda0_star(params, "glycerol")
    ratio = lambda0_star / lambda_0_gly

    print(f"Glycerol: Experimental IC$_{{50}}$ = {ic50_exp:.2f}, Model IC$_{{50}}$ = {ic50_model:.2f}")
    print(f"lambda0* = {lambda0_star:.2f}, lambda0*/lambda0 = {ratio:.2f}")

    # your discrete predictions
    model_pred = simulate_inhibition_curve_cubic(params, "glycerol", normalize=True)

    # ─── NEW: re-solve the cubic on the fine grid for a smooth curve ───
    k_off, k_on, P_in, P_out = extract_inhibition_params(params, "glycerol")
    model_smooth = np.array([
        solve_lambda_cubic(t, lambda_0_gly, k_off, k_on, P_in, P_out)
        for t in TET_range
    ])

    plt.figure(figsize=(8,6))

    # experimental data + fit
    plt.errorbar(TET_concentrations, growth_glycerol, yerr=stdev_glycerol,
                 fmt='^', capsize=5, color=custom_palette[6], label="Glycerol Experimental Data")
    plt.plot(TET_range, lab_fit, '-', color=custom_palette[6],
             label=f"Experimental IC$_{{50}}$ = {ic50_exp:.2f}")

    # ─── NEW: smooth dashed model curve + discrete markers ───
    plt.plot(TET_range, model_smooth,
             linestyle='--', color=custom_palette[7],
             label=f"Model IC$_{{50}}$ = {ic50_model:.2f}")
    plt.plot(TET_concentrations, model_pred,
             marker='x', linestyle='None', color=custom_palette[7])

    # rest unchanged
    plt.xlabel(r"Tetracycline Concentration ($\mu$M)", fontsize=14)
    plt.ylabel(r"Normalized Growth Rate $\frac{\lambda}{\lambda_0}$", fontsize=14, labelpad=20)
    plt.title("Inhibition Curve - Glycerol", fontsize=16)
    plt.xticks(np.arange(0, 2.1, 0.4))
    plt.ylim(ymin=0)
    plt.legend()
    plt.grid(True)

    fig = plt.gcf()
    if save_name:
        save_vector(fig, save_name)
        
    plt.show()

def plot_delay_times(parameters_glu, parameters_gly, optimals_glu, optimals_gly, save_name=None):
    plt.figure(figsize=(10,6))
    # Get the model IC50 using the full 8-element vectors:
    model_ic50_glu = get_model_IC50(optimals_glu, "glucose")
    model_ic50_gly = get_model_IC50(optimals_gly, "glycerol")
    # model_ic50_glu = IC50_MINE["glucose"]
    # model_ic50_gly = IC50_MINE["glycerol"]
    
    exp_Ns_glu = np.array(sorted(exp_delay_times_glu.keys()))
    exp_delays_glu = np.array([exp_delay_times_glu[n] for n in exp_Ns_glu])
    exp_stds_glu = np.array([std_glu[n] for n in exp_Ns_glu])
    plt.errorbar(exp_Ns_glu, exp_delays_glu, yerr=exp_stds_glu,
    fmt='s', color=custom_palette[1], capsize=5, label="Glucose Experimental Data")
    
    exp_Ns_gly = np.array(sorted(exp_delay_times_gly.keys()))
    exp_delays_gly = np.array([exp_delay_times_gly[n] for n in exp_Ns_gly])
    exp_stds_gly = np.array([std_gly[n] for n in exp_Ns_gly])
    plt.errorbar(exp_Ns_gly, exp_delays_gly, yerr=exp_stds_gly,
    fmt='s', color=custom_palette[6], capsize=5, label="Glycerol Experimental Data")
    
    N_range_glu = np.arange(0.1, 20.1, 0.1)
    model_delays_glu = []
    model_Ns_glu = []
    for N in N_range_glu:
        # Make sure to pass the full 8-element vector (optimals_glu).
        dt_list = calculate_delay_time(parameters_glu, [N], model_ic50_glu, optimals_glu)
        model_delays_glu.append(dt_list[0][1])
        model_Ns_glu.append(N)
        
    model_Ns_glu = [0.0] + model_Ns_glu
    model_delays_glu = [0.0] + model_delays_glu
    plt.plot(model_Ns_glu, model_delays_glu, color=custom_palette[2], label="Glucose Model")

    N_range_gly = np.arange(0.1, 20.1, 0.1)
    model_delays_gly = []
    model_Ns_gly = []
    for N in N_range_gly:
        dt_list = calculate_delay_time(parameters_gly, [N], model_ic50_gly, optimals_gly)
        model_delays_gly.append(dt_list[0][1])
        model_Ns_gly.append(N)
        
    model_Ns_gly = [0.0] + model_Ns_gly
    model_delays_gly = [0.0] + model_delays_gly

    plt.plot(model_Ns_gly, model_delays_gly, color=custom_palette[7], label="Glycerol Model")
    plt.xlabel("Pulse Length, T (h)", fontsize=14)
    plt.ylabel(r"Delay Time $\Delta t$ (h)", fontsize=14)
    plt.title("Delay Time vs. Pulse Length", fontsize=16)
    plt.xlim(xmin=0, xmax=20)
    plt.ylim(ymin=0)
    plt.grid(True)
    plt.legend()

    fig = plt.gcf()
    if save_name:
        save_vector(fig, save_name)
        
    plt.show()

def plot_RNA_P(parameters, IC_50, z, lab_RNA_P, lab_RNA_P_std, time_points, substrate, save_name=None):
    plt.figure(figsize=(10,6))
    if substrate.lower() == "glucose":
        color_exp = custom_palette[1]
        color_mod = custom_palette[2]
    else:
        color_exp = custom_palette[6]
        color_mod = custom_palette[7]
    plt.errorbar(time_points, lab_RNA_P[4], yerr=lab_RNA_P_std[4], 
                 fmt='o', capsize=5, color=color_exp, 
                 label=f"{substrate} Experimental Data")
    t_span = np.linspace(0, 10, 200)
    lambda_0_val = parameters['lambda_0']
    y0 = [0, r_min + (lambda_0_val / kappa_t), 0]
    # Pass the full 8-element vector z to model_equations.
    sol = solve_ivp(lambda t, y: model_equations(t, y, 4, parameters, IC_50, z),
    [0, 10], y0, method='BDF', t_eval=t_span)#, rtol=1e-8, atol=1e-10)
    
    r_total = sol.y[1] + sol.y[2]
    idx_ref = np.abs(t_span - 0.5).argmin()
    denom = r_total[idx_ref] if r_total[idx_ref] != 0 else 1e-12
    r_norm = r_total / denom
    plt.plot(t_span, r_norm, color=color_mod, label=f"{substrate} Model")
    plt.axvspan(1, 5, color=custom_palette[9], alpha=0.2, label='Pulse Duration')
    plt.xlabel("Time (h)", fontsize=16)
    plt.ylabel('Normalized Total Ribosome Concentration $r_{total}$', fontsize=14)
    plt.title(f'Normalized $r_{{total}}$ and Experimental Data for {substrate}', fontsize=16)
    plt.xlim(0, 10)
    plt.ylim(0)
    plt.grid(True)
    plt.legend()
    
    fig = plt.gcf()
    if save_name:
        save_vector(fig, save_name)
        
    plt.show()

def plot_growth_laws(lambda_0, kappa_n, N_values, parameters, z, substrate, IC50):
    """Plot gr₁ and gr₂ vs time for each N."""
    alpha0, alphaA = z[4], z[5]
    for which, label in [( (r_min, r_max), substrate )]:
        plt.figure(figsize=(10,6))
        for N in N_values:
            t_on, t_off = 1.0, 1.0+N
            tMax = (N+1)+4+(30/lambda_0)
            time = np.linspace(0, tMax, int(200*tMax))
            sol = solve_ivp(lambda t,y: model_equations(t,y,N,parameters, IC50, z),
                            [time[0],time[-1]], [0, r_min+(lambda_0/kappa_t),0],
                            method='BDF', t_eval=time)
            ru, rb = sol.y[1], sol.y[2]
            gr1 = (ru - r_min)*kappa_t
            gr2 = (r_max - ru - rb)*kappa_n
            plt.plot(time, gr1,  label=f"$\\lambda_T$ (T={N})", alpha=0.6)
            plt.plot(time, gr2,  '--', label=f"$\\lambda_N$ (T={N})", alpha=0.6)
        plt.title(f"{substrate}: Growth Laws vs Time")
        plt.xlabel("Time (h)", fontsize=16); plt.ylabel(r"$\lambda_T$, $\lambda_N$ ($h^{-1}$)", fontsize=16)
        plt.legend(fontsize='small', loc='lower right'); plt.grid(True)
        plt.show()

def plot_synthesis_rate(lambda_0, kappa_n, N_values, parameters, z, substrate, IC50):
    """Plot the pulse‐strategy synthesis rate s(t) for each N."""
    alpha0, alphaA = z[4], z[5]
    s_ss0 = lambda_0*(r_max - lambda_0*delta_r*((1/lambda_0)-(1/(kappa_t*delta_r))))
    plt.figure(figsize=(10,6))
    for N in N_values:
        t_on, t_off = 1.0, 1.0+N
        lam_f = lambda_0/(1+4/N)
        s_ssA = lam_f*(r_max - lam_f/kappa_n)
        tMax = (N+1)+4+(30/lambda_0)
        time = np.linspace(0, tMax, int(200*tMax))
        sol = solve_ivp(lambda t,y: model_equations(t,y,N,parameters, IC50, z),
                        [time[0],time[-1]], [0, r_min+(lambda_0/kappa_t),0],
                        method='BDF', t_eval=time)
        ru, rb = sol.y[1], sol.y[2]
        gr1 = (ru - r_min)*kappa_t
        gr2 = (r_max - ru - rb)*kappa_n
        s = np.zeros_like(time)
        for i,t in enumerate(time):
            if t < t_on:
                s[i] = s_ss0
            elif t <= t_off:
                s[i] = s_ssA*(1 + alphaA*(gr2[i]-gr1[i]))
            else:
                s[i] = s_ss0*(1 + alpha0*(gr2[i]-gr1[i]))
        plt.plot(time, s, label=f"T={N}")
    plt.title(f"{substrate}: Time vs Synthesis Rate")
    plt.xlabel("Time (h)", fontsize=16); plt.ylabel(r"s(t) ($\mu$Mh)$^{-1}$", fontsize=16)
    plt.legend(fontsize='small'); plt.grid(True)
    plt.show()

def plot_r_total_and_time(lambda_0, kappa_n, N_values, parameters, z, substrate, IC50):
    """Plot total ribosomes rₜₒₜ vs time."""
    plt.figure(figsize=(10,6))
    for N in N_values:
        tMax = (N+1)+4+(30/lambda_0)
        time = np.linspace(0, tMax, int(200*tMax))
        sol = solve_ivp(lambda t,y: model_equations(t,y,N,parameters, IC50, z),
                        [time[0],time[-1]], [0, r_min+(lambda_0/kappa_t),0],
                        method='BDF', t_eval=time)
        r_tot = sol.y[1] + sol.y[2]
        plt.plot(time, r_tot, label=f"T={N}")
    plt.title(f"{substrate}: Time vs Total Ribosome Concentration")
    plt.xlabel("Time (h)", fontsize=16); plt.ylabel(r"$r_{total}$ ($\mu$M)", fontsize=16)
    plt.legend(fontsize='small'); plt.grid(True)
    plt.show()

def plot_r_total_vs_growth(lambda_0, kappa_n, N_values, parameters, z, substrate, IC50):
    """
    Plot r_total vs λ for each pulse length N, plus the two boundary lines
      r = r_max - λ/κ_n   and   r = r_min + λ/κ_t
    """
    plt.figure(figsize=(10,6))

    # 1) for each N, solve the ODEs and collect (λ(t), r_total(t))
    for N in N_values:
        t_on, t_off = 1.0, 1.0 + N
        tMax = (N + 1) + 4 + (30 / lambda_0)
        time = np.linspace(0, tMax, int(200 * tMax))

        sol = solve_ivp(
            lambda t, y: model_equations(t,y,N,parameters, IC50, z),
            [time[0], time[-1]],
            [0, r_min + (lambda_0 / kappa_t), 0],
            method='BDF',
            t_eval=time
        )

        ru = sol.y[1]
        rb = sol.y[2]
        gr1 = (ru - r_min) * kappa_t
        gr2 = (r_max - ru - rb) * kappa_n
        gr  = np.minimum(gr1, gr2)
        r_tot = ru + rb

        plt.plot(gr, r_tot, label=f"T={N}")

    # 2) add the two straight‐line boundaries over the same λ‐range
    λ_vals = np.linspace(0, 1.0, 200)
    plt.plot(λ_vals, r_max - λ_vals / kappa_n,
             'k--', label=r'$r_{\max} - \lambda/\kappa_n$')
    plt.plot(λ_vals, r_min   + λ_vals / kappa_t,
             'k-.', label=r'$r_{\min} + \lambda/\kappa_t$')

    plt.title(f"{substrate}: Growth Rate vs $r_{{total}}$")
    plt.xlabel(r"$\lambda(t)$ (h$^{-1}$)", fontsize=16)
    plt.ylabel(r"$r_{\mathrm{total}}$ ($\mu$M)", fontsize=16)
    plt.xlim(0, 1.0)
    plt.ylim(r_min * 0.9, r_max * 1.05)
    plt.legend(fontsize='small')
    plt.grid(True)
    plt.show()

###############################################################################
#                       Calls to Plot After Optimization                      #
###############################################################################
# --- only save plots if converged ---
if result.success:
    print("Optimization converged! Saving plots...")

    # Growth rate plots
    analyze_growth(lambda_0_glu, N_values_glu, parameters_glu, optimals_glu_full, "Glucose", save_name="August_growthrates_glucose")
    analyze_growth(lambda_0_gly, N_values_gly, parameters_gly, optimals_gly_full, "Glycerol", save_name="August_growthrates_glycerol")

    # Inhibition curves
    plot_inhibition_curve_glu(optimals_glu, save_name="August_inhibition_glucose")
    plot_inhibition_curve_gly(optimals_gly, save_name="August_inhibition_glycerol")

    # Delay time plot
    plot_delay_times(parameters_glu, parameters_gly, optimals_glu_full, optimals_gly_full, save_name="August_delay_times")

    # RNA/P plots
    plot_RNA_P(parameters_glu, opt_model_IC50_glu, optimals_glu_full, lab_RNA_P_glu, lab_RNA_P_glu_std, time_points_4H, "Glucose", save_name="August_RNAP_glucose")
    plot_RNA_P(parameters_gly, opt_model_IC50_gly, optimals_gly_full, lab_RNA_P_gly, lab_RNA_P_gly_std, time_points_4H, "Glycerol", save_name="August_RNAP_glycerol")

else:
    print("Optimization did NOT converge — no plots saved.")

# Growth laws
plot_growth_laws(lambda_0_glu, kappa_n_glu, N_values_glu, parameters_glu, optimals_glu_full, "Glucose", opt_model_IC50_glu)
plot_synthesis_rate(lambda_0_glu, kappa_n_glu, N_values_glu, parameters_glu, optimals_glu_full, "Glucose", opt_model_IC50_glu)
plot_r_total_and_time(lambda_0_glu, kappa_n_glu, N_values_glu, parameters_glu, optimals_glu_full, "Glucose", opt_model_IC50_glu)
plot_r_total_vs_growth(lambda_0_glu, kappa_n_glu, N_values_glu, parameters_glu, optimals_glu_full, "Glucose", opt_model_IC50_glu)

plot_growth_laws(lambda_0_gly, kappa_n_gly, N_values_gly, parameters_gly, optimals_gly_full, "Glycerol", opt_model_IC50_gly)
plot_synthesis_rate(lambda_0_gly, kappa_n_gly, N_values_gly, parameters_gly, optimals_gly_full, "Glycerol", opt_model_IC50_gly)
plot_r_total_and_time(lambda_0_gly, kappa_n_gly, N_values_gly, parameters_gly, optimals_gly_full, "Glycerol", opt_model_IC50_gly)
plot_r_total_vs_growth(lambda_0_gly, kappa_n_gly, N_values_gly, parameters_gly, optimals_gly_full, "Glycerol", opt_model_IC50_gly)

end = timer()
print("\nTotal runtime: {:.2f} minutes.\n".format((end - start)/60))


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Purpose: Simulate antibiotic-pulse dynamics (glucose) and plot growth, OD, delay times, r_total vs "growth laws", r_total and synthesis rate.

Usage:
- Fill the USER placeholders below (IC50 from your fit, optional experimental overlays, optimal parameters).
- Run in Jupyter or as a script.

Data policy:
- This public version contains NO lab-measured data. See thesis Materials and Methods for how to obtain/format inputs.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import warnings
warnings.filterwarnings("ignore", message=".*Intel MKL.*")

# =========================
# Define parameter values
# =========================
r_min = 5.4
r_max = 54.4
delta_r = r_max - r_min
kappa_t = 0.058
lambda_max = kappa_t * delta_r

# USER: (k_off, k_on, P_in, P_out, alpha_A, alpha_0) are optimal parameters (Proportional Response Optimization). Set before running.
k_off_glu = None
k_on_glu = None
P_in_glu = None
P_out_glu = None
alpha_A_glu = None
alpha_0_glu = None

lambda_0_glu = 0.64
Greu_IC50_glu = 1.75   # nominal/reference IC50 (µM) used in prefactor

# Derived quantities
K_D = k_off_glu / k_on_glu
lambda_0_star  = 2.0 * np.sqrt(P_out_glu * kappa_t * K_D)
IC_50_star = (delta_r * lambda_0_star) / (2.0 * P_in_glu)
IC_50_glu = 0.5 * IC_50_star * ((lambda_0_glu / lambda_0_star) + (lambda_0_star / lambda_0_glu))

# USER: IC50 from your inhibition-curve fit (µM). Set before running.
lab_IC_50_glu = None

# Prefactor for pulse (uses Greu_IC50_glu and your lab_IC_50_glu)
prefactor_glu = 4 * Greu_IC50_glu / lab_IC_50_glu  # USER: set lab_IC_50_glu above

kappa_n_glu = 1/((delta_r * ((1 / lambda_0_glu) - (1 / (kappa_t * delta_r)))))

N_values_glu = [2,4,5,6,8,10]
N_values = range(1,16)

# ==========================================
# EXPERIMENTAL DATA PLACEHOLDERS (EMPTY)
# Formats (no example numbers in public repo):
#   exp_delay_times_glu : dict[int, float]        -> {N_hours: delay_time_hours}
#   std_glu             : dict[int, float]        -> {N_hours: std_dev_hours}
#   lab_RNA_P_glu       : dict[int, list[float]]  -> {4: [...] } if used
#   lab_RNA_P_glu_std   : dict[int, list[float]]  -> {4: [...] } if used
#   time_points_4H      : list[float]             -> hours matching lists above
# ==========================================
exp_delay_times_glu = {}
std_glu = {}
lab_RNA_P_glu = {}
lab_RNA_P_glu_std = {}
time_points_4H = []

# =========================
# ODE model
# =========================
def model_equations(t, y):
    a  = y[0]  # intracellular antibiotic
    r_u = y[1] # unbound ribosomes
    r_b = y[2] # bound ribosomes

    t_on  = 1
    t_off = t_on + N

    # Two candidate dilution/synthesis terms
    x_syn = (r_u - r_min) * kappa_t
    x_dil = (r_max - r_u - r_b) * kappa_n_glu
    x     = min(x_syn, x_dil)

    # Synthesis rate s(t) by time region
    if t < t_on:
        s_ss0 = lambda_0_glu * (r_max - (lambda_0_glu * delta_r * ((1 / lambda_0_glu) - (1 / lambda_max))))
        s = s_ss0
        a_ex = 0
    elif t <= t_off:
        lambda_f = lambda_0_glu / (1 + (prefactor_glu / N))
        s_ssA = lambda_f * (r_max - (lambda_f / kappa_n_glu))
        s = s_ssA * (1 + (alpha_A_glu * (x_dil - x_syn)))
        a_ex = (prefactor_glu * lab_IC_50_glu) / N   # USER: requires lab_IC_50_glu set above
    else:
        s_ss0 = lambda_0_glu * (r_max - (lambda_0_glu * delta_r * ((1 / lambda_0_glu) - (1 / lambda_max))))
        s = s_ss0 * (1 + (alpha_0_glu * (x_dil - x_syn)))
        a_ex = 0
    
    # Binding/unbinding
    F = (k_on_glu * a * (r_u - r_min)) - (k_off_glu * r_b)

    # ODEs
    da_dt  = -F - (x * a) + (P_in_glu * a_ex) - (P_out_glu * a)
    dr_u_dt = -F - (x * r_u) + s
    dr_b_dt =  F - (x * r_b)

    return [da_dt, dr_u_dt, dr_b_dt]

# Initial conditions
y0 = [0, r_min + (lambda_0_glu / kappa_t), 0]

# Global time grid for plots
tMax = max(N_values) + 4 + (30 / lambda_0_glu)
time = np.linspace(0, tMax, int(200 * tMax))

growth_rates_list = []
ratios = []

# Solve the ODEs for each value of N
for N in N_values:
    solution = solve_ivp(model_equations, [time[0], time[-1]], y0, method='BDF', t_eval=time)

    # Compute both laws everywhere; growth is min(gr1, gr2)
    gr1_values = (solution.y[1] - r_min) * kappa_t
    gr2_values = (r_max - solution.y[1] - solution.y[2]) * kappa_n_glu
    growth_rate = np.minimum(gr1_values, gr2_values)
    growth_rates_list.append(growth_rate)

# Plot growth rates for all N values on the same plot
plt.figure(figsize=(10, 6))
for i, N in enumerate(N_values):
    plt.plot(time, growth_rates_list[i], label=f"N = {N}")
plt.xlabel('Time (h)', fontsize=12)
plt.ylabel(r'$\lambda$', fontsize=12, rotation=0, labelpad=20)
plt.grid(True)
plt.ylim(ymin=0, ymax=1.5)
plt.xlim(xmin=0, xmax=30)
plt.title(r'Antibiotic Pulse $\lambda$ Plot for Glucose ($\lambda_0=0.64$)')
plt.axvspan(1, N + 1, color='gray', alpha=0.2, label='Pulse Duration')
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.)
plt.show()

# =========================
# OD and delay calculation
# =========================
ODs_and_delays = []
data_list = []

for N, growth_rate in zip(N_values, growth_rates_list):
    OD = np.cumsum(growth_rate) * (time[1] - time[0])
    x_in_pre  = time[1]
    x_in_post = time[-1]

    OD_in_pre  = OD[1]
    OD_in_post = OD[-1]
    
    y_int_pre  = OD_in_pre  - (lambda_0_glu * x_in_pre)
    y_int_post = OD_in_post - (lambda_0_glu * x_in_post)
    
    y_values_pre  = (lambda_0_glu * time) + y_int_pre
    y_values_post = (lambda_0_glu * time) + y_int_post

    delay = ((OD_in_pre - OD_in_post) / lambda_0_glu) + (x_in_post - x_in_pre)

    ODs_and_delays.append((N, OD, delay))
    data_list.append((N, delay))

# Plot ODs and reference lines
plt.figure(figsize=(10, 6))
for N, OD, delay in ODs_and_delays:
    plt.plot(time, OD, label=f'OD - T={N}')
    # For display, reuse the last computed y_values_pre/post from this loop if desired,
    # or recompute inside if you need per-N lines drawn distinctly.
plt.xlabel('Time', fontsize=12)
plt.ylabel(r'$OD_{600}$', fontsize=12)
plt.grid(True)
plt.title(r'$OD_{600}$ vs Time')
plt.show()

# Scatter of delays and plateau line
plt.figure(figsize=(10, 6))
for N, OD, delay in ODs_and_delays:
    plt.scatter(N, delay, color='blue', marker='o', label=f'T = {N}')
plt.xlabel('Length of Pulse (T=N)', fontsize=12)
plt.ylabel(r'$\Delta$t', fontsize=12, rotation=0, labelpad=20)
plt.title(r'Length of Pulse vs $\Delta$t')
plt.xlim(xmin=0)
plt.ylim(ymin=0)
plt.grid(True)
plt.show()

# =========================
# r_total vs λ(t) plotting
# =========================
x_values_1 = np.linspace(0, 1, 100)
y_values_1 = r_max - (x_values_1 / kappa_n_glu)
x_values_2 = np.linspace(0, 2, 100)
y_values_2 = r_min + (x_values_2 / kappa_t)

r_total_and_growth_rates = []
for N, growth_rate in zip(N_values, growth_rates_list):
    solution = solve_ivp(model_equations, [time[0], time[-1]], y0, method='BDF', t_eval=time)
    r_total = solution.y[1] + solution.y[2]
    r_total_and_growth_rates.append((N, growth_rate, r_total))

plt.figure(figsize=(10, 6))
for N, growth_rate, r_total in r_total_and_growth_rates:
    plt.plot(growth_rate, r_total, label=f"T = {N}")
plt.plot(x_values_1, y_values_1, label=r'$r_{max} - \frac{\lambda}{\kappa_n}$')
plt.plot(x_values_2, y_values_2, label=r'$r_{min} + \frac{\lambda}{\kappa_t}$')
plt.xlabel(r'$\lambda$(t)', fontsize=12)
plt.ylabel(r'$r_{total}$', fontsize=12)
plt.xlim(xmin=0, xmax=1)
plt.ylim(ymin=5, ymax=40)
plt.show()

# =========================
# Optional RNA/P overlay (public version disabled)
# Provide your own arrays locally to enable this block:
#   lab_RNA_P_glu: {4: [ ... ]}
#   lab_RNA_P_glu_std: {4: [ ... ]}
#   time_points_4H: [ ... ]
# =========================
simulated_r_total = []
r_total_data = []
max_rnap = {}
max_times = {}

for N in lab_RNA_P_glu.keys():
    sol_rtot = solve_ivp(lambda t, y: model_equations(t, y),
                         [time[0], time[-1]], y0, method='BDF', t_eval=time)
    desired_time = 0.5
    r_total = sol_rtot.y[1] + sol_rtot.y[2]
    nearest_index = np.abs(sol_rtot.t - desired_time).argmin()
    r_total_normalized = r_total / r_total[nearest_index]

    simulated_r_total_plot = []
    for t in time:
        idx = np.abs(sol_rtot.t - t).argmin()
        simulated_r_total_plot.append(r_total_normalized[idx])
        r_total_data.append(r_total_normalized[idx])

    peak_idx = np.argmax(r_total_normalized)
    max_rnap[N] = r_total_normalized[peak_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(time, simulated_r_total_plot, label=f'Theoretical Data (N={N})')
    # If you add your data locally, the next line can overlay it:
    # plt.errorbar(time_points_4H, lab_RNA_P_glu[N], yerr=lab_RNA_P_glu_std.get(N, None), capsize=5, linestyle='None', fmt='o', label=f'Lab Data (N={N})', color='red')
    plt.axvline(x=1, color='darkgreen', linestyle='--', label='Pulse Start (t=1)')
    plt.axvline(x=N + 1, color='darkred', linestyle='--', label=f'Pulse End (t={N})')

if lab_RNA_P_glu:
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Normalized $r_{total}$', fontsize=12)
    plt.title('Normalized $r_{total}$ and Lab Data (Glucose)')
    plt.grid(True)
    plt.xlim(xmin=0, xmax=10)
    plt.ylim(ymin=0)
    plt.legend(loc='best')
    plt.show()

# =========================
# Synthesis rate s(t) plot
# =========================
alpha0 = alpha_0_glu
alphaA = alpha_A_glu

s_ss0 = lambda_0_glu * (
    r_max
    - lambda_0_glu * delta_r * ((1 / lambda_0_glu) - (1 / (kappa_t * delta_r)))
)

plt.figure(figsize=(10,6))
for N in N_values_glu:
    t_on, t_off = 1.0, 1.0 + N
    lam_f  = lambda_0_glu / (1 + prefactor_glu / N)
    s_ssA  = lam_f * (r_max - lam_f / kappa_n_glu)

    sol = solve_ivp(lambda t,y: model_equations(t, y),
                    [time[0], time[-1]], y0, method='BDF', t_eval=time)
    ru, rb = sol.y[1], sol.y[2]

    gr1 = (ru - r_min) * kappa_t
    gr2 = (r_max - ru - rb) * kappa_n_glu

    s = np.zeros_like(time)
    for i, t_i in enumerate(time):
        if t_i < t_on:
            s[i] = s_ss0
        elif t_i <= t_off:
            s[i] = s_ssA * (1 + alphaA * (gr2[i] - gr1[i]))
        else:
            s[i] = s_ss0 * (1 + alpha0 * (gr2[i] - gr1[i]))

    plt.plot(time, s, label=f"N = {N}")

plt.title("Glucose: Time vs Synthesis Rate")
plt.xlabel("Time (h)", fontsize=14)
plt.ylabel("s(t)", fontsize=14)
plt.legend(title="Pulse length", fontsize=12)
plt.grid(True)
plt.show()


# In[ ]:





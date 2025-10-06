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

# Define parameter values
r_min = 5.4
r_max = 54.4
delta_r = r_max - r_min
kappa_t = 0.058
lambda_max = kappa_t * delta_r

# USER: (k_off, k_on, P_in, P_out) are optimal parameters (Steady State Optimization). Set before running.
k_off_glu = None
k_on_glu = None
P_in_glu = None
P_out_glu = None

lambda_0_glu = 0.64
Greu_IC50_glu = 1.75  # nominal/reference IC50 (µM) used in prefactor

# Derived quantities from parameters above
K_D = k_off_glu / k_on_glu
lambda_0_star  = 2.0 * np.sqrt(P_out_glu * kappa_t * K_D)
IC_50_star = (delta_r * lambda_0_star) / (2.0 * P_in_glu)
IC_50_glu = 0.5 * IC_50_star * ((lambda_0_glu / lambda_0_star) + (lambda_0_star / lambda_0_glu))

# USER: IC50 from your inhibition-curve fit (µM). Set before running.
lab_IC_50_glu = None

# Pulse prefactor (uses Greu_IC50_glu and your lab_IC_50_glu)
prefactor_glu = 4 * Greu_IC50_glu / lab_IC_50_glu  # USER: requires lab_IC_50_glu to be set

# kappa_n for delay model
kappa_n_glu = 1 / ((delta_r * ((1 / lambda_0_glu) - (1 / (kappa_t * delta_r)))))

N_values_glu = [2,4,5,6,8,10]
N_values = range(1,16)

# ────────────────────────────────────────────────────────────────────────────
# EXPERIMENTAL DATA PLACEHOLDERS (public version: leave empty)
# Formats (no example numbers):
#   exp_delay_times_glu : dict[int, float]      -> {N_hours: delay_time_hours}
#   std_glu             : dict[int, float]      -> {N_hours: std_dev_hours}
#   lab_RNA_P_glu       : dict[int, list[float]]  with key 4
#   lab_RNA_P_glu_std   : dict[int, list[float]]  with key 4
#   time_points_4H      : list[float]           -> hours matching the lists above
# ────────────────────────────────────────────────────────────────────────────
exp_delay_times_glu = {}
std_glu = {}
lab_RNA_P_glu = {}
lab_RNA_P_glu_std = {}
time_points_4H = []

# Define model equations
def model_equations(t, y):
    a = y[0] # intracellular antibiotic
    r_u = y[1] # unbound ribosomes
    r_b = y[2] # bound ribosomes

    # Iterate step-pulse addition of antibiotics
    if 0<=t<=1:
        a_ex = 0
        x = (r_u-r_min)*kappa_t
        s = x * (r_max - ((x * delta_r) * ((1 / lambda_0_glu) - (1 / (kappa_t*delta_r)))))
    elif 1<=t<=N+1:
        a_ex = (prefactor_glu*lab_IC_50_glu) / N  # USER: requires lab_IC_50_glu
        x = (r_u-r_min)*kappa_t
        s = x * (r_max - ((x * delta_r) * ((1 / lambda_0_glu) - (1 / (kappa_t*delta_r)))))
    else:
        a_ex = 0
        x = (r_max-r_u-r_b)*kappa_n_glu
        s = x*(r_min+(x/kappa_t))
    
    # Calculate F - binding/unbinding
    F = (k_on_glu * a * (r_u - r_min)) - (k_off_glu * r_b)

    # Calculate the rate of change of a, r_u, r_b
    da_dt = -F - (x * a) + (P_in_glu * a_ex) - (P_out_glu * a)
    dr_u_dt = -F - (x * r_u) + s
    dr_b_dt = F - (x * r_b)

    # Return the rate of change of a, r_u, r_b
    return[da_dt, dr_u_dt, dr_b_dt]
    pass

# Set initial conditions
y0 = [0, r_min+(lambda_0_glu/kappa_t), 0]

# Define the time range
for N in N_values:
    tMax=(N+1)+4+(30/lambda_0_glu)
    time=np.linspace(0,tMax,int(10*tMax))
    
growth_rates_list = []
ratios = []

# Solve the ODEs for each value of N
for N in N_values:
    solution = solve_ivp(model_equations, [time[0], time[-1]], y0, method='BDF', t_eval=time, rtol=1e-6, atol=1e-9)

    gr1_values = np.zeros_like(time)
    gr2_values = np.zeros_like(time)

    # Calculate gr1 and gr2 for all time points
    for i, t in enumerate(time):
        if 0 < t < N+1:
            gr1_values[i] = ((solution.y[1, i] - r_min) * kappa_t)
        else:
            gr2_values[i] = ((r_max - solution.y[1, i] - solution.y[2, i]) * kappa_n_glu)

    # Calculate the total growth rate by element-wise addition
    growth_rate = gr1_values + gr2_values
    growth_rates_list.append(growth_rate)

# Plot growth rates for all N values on the same plot
plt.figure(figsize=(10, 6))
for i, N in enumerate(N_values):
    plt.plot(time, growth_rates_list[i], label=f"N = {N}")

plt.xlabel('Time (h)', fontsize=14)
plt.ylabel(r'$\lambda$', fontsize=16, rotation=0, labelpad=20)
plt.grid(True)
plt.ylim(ymin=0, ymax=1.5)
plt.xlim(xmin=0, xmax=20)
plt.title(r'Antibiotic Pulse $\lambda$ Plot for Glucose ($\lambda_0=0.64$)', fontsize=16)
plt.axvspan(1, N+1, color='gray', alpha=0.2, label='Pulse Duration')
plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.)
plt.legend(loc='best', fontsize=14)
plt.show()

# Updated loop
ODs_and_delays = []
data_list = []

for N, growth_rate in zip(N_values, growth_rates_list):
    OD = np.cumsum(growth_rate)*(time[1]-time[0])
    
    x_in_pre = time[1]
    x_in_post = time[-1]

    OD_in_pre = OD[1]
    OD_in_post = OD[-1]
    
    y_int_pre = OD_in_pre - (lambda_0_glu * x_in_pre)
    y_int_post = OD_in_post - (lambda_0_glu * x_in_post)
    
    y_values_pre = (lambda_0_glu * time) + y_int_pre
    y_values_post = (lambda_0_glu * time) + y_int_post

    delay = ((OD_in_pre - OD_in_post) / lambda_0_glu) + (x_in_post - x_in_pre)

    ODs_and_delays.append((N, OD, delay))
    data_list.append((N, delay))

# Plot ODs and x-intercepts for each N
plt.figure(figsize=(10, 6))
for N, OD, delay in ODs_and_delays:
    plt.plot(time, OD, label=f'$OD_{{600}}$ - N = {N}')

    x_in_pre = time[1]
    x_in_post = time[-1]

    OD_in_pre = OD[1]
    OD_in_post = OD[-1]

    # Plot the line corresponding to t=[0,1] for each N
    plt.plot(time, y_values_pre, label=f'Pre-Antibiotic $\lambda_0$ - N = {N}', linestyle='--', color='darkgreen')

    # Plot the line extending to the end of the time span for each N
    plt.plot(time, y_values_post, label=f'Post-Antibiotic $\lambda_0$ - N = {N}', linestyle='--', color='darkred')
    
plt.xlabel('Time', fontsize=14)
plt.ylabel(r'$OD_{600}$', fontsize=16)
plt.grid(True)
plt.xlim(0,20)
plt.ylim(0,12.5)
plt.title(r'$OD_{600}$ vs Time', fontsize=16)
plt.legend(fontsize=14)
plt.show()

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

# Calculate x and y values for the r_max and r_min lines
x_values_1 = np.linspace(0, 1, 100)
y_values_1 = r_max - (x_values_1 / kappa_n_glu)
x_values_2 = np.linspace(0, 2, 100)
y_values_2 = r_min + (x_values_2 / kappa_t)

# Initialize a list to store r_total and growth rate data for each N
r_total_and_growth_rates = []

for N, growth_rate in zip(N_values, growth_rates_list):
    solution = solve_ivp(model_equations, [time[0], time[-1]], y0, method='BDF', t_eval=time, rtol=1e-6, atol=1e-9)
    r_total = solution.y[1] + solution.y[2]
    r_total_and_growth_rates.append((N, growth_rate, r_total))

# Plot r_total and lines for each N
plt.figure(figsize=(10, 6))
for N, growth_rate, r_total in r_total_and_growth_rates:
    plt.plot(growth_rate, r_total, label=f"N = {N}")
plt.plot(x_values_1, y_values_1, label=r'$r_{max} - \frac{\lambda}{\kappa_n}$', color='#6A204C')
plt.plot(x_values_2, y_values_2, label=r'$r_{min} + \frac{\lambda}{\kappa_t}$', color='#A9561E')
plt.xlabel(r'$\lambda$(t)', fontsize=16)
plt.ylabel(r'$r_{total}$', fontsize=18)
plt.legend(fontsize=14)
plt.xlim(xmin=0, xmax=2)
plt.ylim(ymin=0, ymax=40)
plt.show()

simulated_r_total = []
r_total_data = []

# Overlay block (optional): only runs if user provides RNA/P data & times for N=4
for N in lab_RNA_P_glu.keys():
    tMax = (N + 1) + 4 + (30 / lambda_0_glu)
    time = np.linspace(0, tMax, int(10 * tMax))

    # Solve the ODE system
    sol_rtot = solve_ivp(lambda t, y: model_equations(t, y), [time[0], time[-1]], y0, method='BDF', t_eval=time)

    # Compute r_total and normalize to value at t=0.5 h
    desired_time = 0.5
    r_total = sol_rtot.y[1] + sol_rtot.y[2]
    nearest_index = np.abs(sol_rtot.t - desired_time).argmin()
    r_total_normalized = r_total / r_total[nearest_index]

    # Store the simulated values (if you want to save/compare)
    for t in time:
        idx = np.abs(sol_rtot.t - t).argmin()
        simulated_r_total.append(r_total_normalized[idx])
        r_total_data.append(r_total_normalized[idx])

    # Plot model + (optional) lab data
    plt.figure(figsize=(10, 6))
    plt.plot(time, r_total_normalized, label=f'Theoretical Data (N={N})')
    if N in lab_RNA_P_glu:
        plt.errorbar(time_points_4H, lab_RNA_P_glu[N],
                     yerr=lab_RNA_P_glu_std.get(N, None),
                     capsize=5, linestyle='None', fmt='o',
                     label=f'Lab Data (N={N})', color='red')
    plt.axvspan(1, N + 1, color='gray', alpha=0.2, label='Pulse Duration')

# Final plot settings for overlay block
if lab_RNA_P_glu:
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Normalized $r_{total}$', fontsize=12)
    plt.title('Normalized $r_{total}$ and Lab Data for Glucose')
    plt.grid(True)
    plt.xlim(xmin=0, xmax=10)
    plt.ylim(ymin=0)
    plt.legend(loc='best')
    plt.show()

# Synthesis rate s(t) demo
s_values_glu = []
N_synth = 4
solution_demo = solve_ivp(model_equations, [time[0], time[-1]], y0, method='BDF', t_eval=time)
for i, t in enumerate(time):
    r_u = solution_demo.y[1, i]
    r_b = solution_demo.y[2, i]
    if 0 <= t < 1:  
        x = (r_u - r_min) * kappa_t
        s = x * (r_max - (x * delta_r * ((1 / lambda_0_glu) - (1 / lambda_max))))
    elif 1 <= t <= N_synth + 1:
        x = (r_u - r_min) * kappa_t
        s = x * (r_max - (x * delta_r * ((1 / lambda_0_glu) - (1 / lambda_max))))
    else:
        x = (r_max - r_u - r_b) * kappa_n_glu
        s = x * (r_min + (x / kappa_t))
    s_values_glu.append(s)

plt.figure(figsize=(10, 6))
plt.plot(time, s_values_glu, label=r'$s(t)$')
plt.xlabel('Time')
plt.ylabel(r'Synthesis Rate $s(\lambda)$')
plt.title(r'Synthesis Rate $s(\lambda)$ vs Time')
plt.axvline(x=1, color='darkgreen', linestyle='--', label='Pulse Start (t=1)')
plt.axvline(x=N_synth + 1, color='darkred', linestyle='--', label=f'Pulse End (t={N_synth + 1})')
plt.grid(True)
plt.xlim(0, 10)
plt.legend()
plt.show()


# In[ ]:





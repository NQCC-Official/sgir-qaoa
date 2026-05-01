# -*- coding: utf-8 -*-
"""

Grover eigenvalue spectrum plot and RC QAA schedule


"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# User's preferred style settings
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 25})

# Parameters
N = 64  # Hilbert space size, N = 2^n
s_steps = 400
s_vals = np.linspace(0, 1, s_steps)

# Define H0 and HP
psi0 = np.ones((N, 1)) / np.sqrt(N)
H0 = np.eye(N) - np.dot(psi0, psi0.T) 

m = np.zeros((N, 1))
m[0] = 1
HP = np.eye(N) - np.dot(m, m.T) 

# Calculate Eigenvalues
E0, E1, E2 = [], [], []
for s in s_vals:
    H_s = (1 - s) * H0 + s * HP
    evals = eigh(H_s, eigvals_only=True, subset_by_index=[0, 2])
    E0.append(evals[0])
    E1.append(evals[1])
    E2.append(evals[2] if len(evals) > 2 else 1.0)

# Plotting
colors = plt.cm.inferno(np.linspace(0.1, 0.75, 3))

plt.figure(figsize=(10, 7))
plt.plot(s_vals, E0, color=colors[0], label='$E_0$ ', linewidth=2)
plt.plot(s_vals, E1, color=colors[1], label='$E_1$', linewidth=2)
plt.plot(s_vals, E2, color=colors[2], label='$E_2$', linewidth=2)

plt.xlabel('$s$')
plt.ylabel('Energy')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=24)
plt.savefig(f"plots/grover_eigenval_spectrum.pdf")
plt.show()


# Using the gaps calculated from the spectrum above
gaps = np.array(E1) - np.array(E0)

# Adiabatic condition: df/ds = (epsilon / gap^2)
epsilon = 0.05 
df = (1.0 / (gaps**2)) * (s_vals[1] - s_vals[0]) * epsilon
t_vals = np.cumsum(df)
t_vals = t_vals - t_vals[0] 

# Plotting the Schedule
plt.figure(figsize=(10, 7))
plt.plot(t_vals, s_vals, color=plt.cm.plasma(0.5), linewidth=2)

plt.xlabel('$s$')
plt.ylabel('$f$')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(f"plots/grover_optimal_schedule.pdf")
plt.show()
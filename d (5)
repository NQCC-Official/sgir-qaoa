# -*- coding: utf-8 -*-
"""

Grover eigenvalue spectrum plot and SGIR-QAOA schedule


"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from datetime import datetime
from qiskit.quantum_info import SparsePauliOp
from itertools import product

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 25})

timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

n = 4
p = 10


def grover_hamiltonian_ising(target_state: str):
    """
    Build the true Grover oracle Hamiltonian in Ising (Pauli-Z) form:
        H = I - |target><target|
    Returns a SparsePauliOp.
    """
    n = len(target_state)

    # Initialize dictionary for Pauli terms
    pauli_dict = {"I" * n: 1.0}  # Start with the identity (I)

    # Expand product ∏_i (I + (-1)^t_i Z_i)/2
    coeff = 1 / (2**n)
    for bits in product([0, 1], repeat=n):
        # bits indicate which Z_i are included
        pauli_label = list("I" * n)
        sign = 1.0
        for i, b in enumerate(bits):
            if b == 1:
                pauli_label[i] = "Z"
                sign *= (-1) ** int(target_state[i])
        pauli_dict["".join(pauli_label)] = pauli_dict.get("".join(pauli_label), 0) - coeff * sign

    # Remove zero terms
    pauli_dict = {p: c for p, c in pauli_dict.items() if abs(c) > 1e-12}

    # Build SparsePauliOp
    H = SparsePauliOp(list(pauli_dict.keys()), list(pauli_dict.values()))
    return H


# target_state = ''.join(random.choice(['0', '1']) for _ in range(n))
target_state = '0' * n
print(target_state)

H_oracle = grover_hamiltonian_ising(target_state)
    
    
hamiltonian = {
tuple([i for i, p in enumerate(label) if p != "I"]): coeff
for label, coeff in zip(H_oracle.paulis.to_labels(), H_oracle.coeffs)
}
    
# Normalization: scale all weights by the maximum absolute value (as is done in the LR-QAOA paper)
max_weight = max(abs(w) for w in hamiltonian.values()) if hamiltonian else 1.0

hamiltonian = {
    term: w / max_weight
    for term, w in hamiltonian.items()
    if abs(w) > 1e-9
}


def tensor_kron(ops):
    out = ops[0]
    for A in ops[1:]:
        out = np.kron(out, A)
    return out

# Pauli matrices
X = np.array([[0,1],[1,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)
I = np.eye(2, dtype=complex)

def build_projector_driver(n):
    dim = 2**n
    psi0 = np.ones(dim) / np.sqrt(dim)           # |s>
    P = np.outer(psi0, psi0)
    return np.eye(dim, dtype=complex) - P       # H0 = I - |s><s|

def build_transverse_field(n):
    dim = 2**n
    H = np.zeros((dim,dim), dtype=complex)
    for i in range(n):
        ops = [I]*n
        ops[i] = X
        H += tensor_kron(ops)
    # convention: we want ground state |s> -> choose negative sign
    return -H

def Z_i(n, i):
    ops = [I]*n
    ops[i] = Z
    return tensor_kron(ops)

def ZZ_ij(n, i, j):
    ops = [I]*n
    ops[i] = Z
    ops[j] = Z
    return tensor_kron(ops)

def I_all(n):
    """n-qubit Identity matrix."""
    out = I
    for _ in range(n-1):
        out = np.kron(out, I)
    return out

def build_cost_from_edges(n, hamiltonian):
    """
    Build H_cost from edge-based Hamiltonian dict:
    hamiltonian = {(i,j): weight, ...}
    """
    dim = 2**n
    H_cost = np.zeros((dim, dim), dtype=complex)
    
    for (i, j), w in hamiltonian.items():
        H_cost += w * (I_all(n) - ZZ_ij(n, i, j)) / 2.0
        # (I - Z_i Z_j)/2 is the MaxCut term
    
    return -H_cost

def lowest_k_eigs(H, k=6):
    # dense small-matrix eigensolver; returns ascending eigenvalues
    vals = eigh(H, eigvals_only=True)
    return np.sort(vals)[:k]


def build_grover_oracle(n, target_state):
    dim = 2**n
    H = np.eye(dim, dtype=complex)
    
    # Index of the marked (target) state
    idx = int(target_state, 2)
    
    # Subtract projector onto marked state
    H[idx, idx] -= 1.0
    return H

s_vals = np.linspace(0, 1, p+1)

H0_proj = build_projector_driver(n)
H_X = build_transverse_field(n)

H_cost = build_grover_oracle(n, target_state)


# Store eigenvalues
eig_proj = np.zeros((len(s_vals), 6))
eig_X    = np.zeros((len(s_vals), 6))
gaps_qaoa = np.zeros(len(s_vals))

for j, s in enumerate(s_vals):
    Hinst = (1-s)*H_X + s*H_cost
    HX_s = Hinst
    
    eig_X[j,:]    = lowest_k_eigs(HX_s, k=6)
    
    gaps_qaoa[j] = eig_X[j, 1] - eig_X[j, 0]
    


print('E0, E1, E2 are:', eig_X[-1][:3])




colors = plt.cm.inferno(np.linspace(0.1, 0.75, 3))


# # Plotting only the transverse-field driver (QAOA style)
plt.figure(figsize=(10,7))

for k in range(3): # make 3 to see e2
    plt.plot(s_vals, eig_X[:,k], label=f'E{k}' if k<3 else None, color=colors[k])

plt.xlabel("s")
plt.ylabel("Energy")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(f"plots/grover_eig_spec_{target_state}.pdf")
plt.show()


# Now calculate the SGIR schedule

def make_f_from_gap_shifted(s_vals, gap_vals, power=2.0, start=0.0, end=1.0):
    s_vals = np.asarray(s_vals, dtype=float)
    gap_vals = np.asarray(gap_vals, dtype=float)
    
    # symmetric weight function
    integrand = np.maximum(gap_vals - np.min(gap_vals), 0.0)**power
    
    # cumulative integral with trapezoidal rule
    cum = np.zeros_like(s_vals)
    cum[1:] = np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(s_vals))
    
    # normalize to [0,1]
    cum = (cum - cum[0]) / (cum[-1] - cum[0])
    
    # scale to [start,end]
    return start + cum * (end - start)


log_delta_beta_range = [-1.5,0.5]
log_delta_gamma_range = [-1,1]

log_delta_beta = log_delta_beta_range[0]
log_delta_gamma = log_delta_gamma_range[0]

beta_start = 10**(log_delta_beta)
gamma_end = 10**(log_delta_gamma)

gamma_start = (1/p) * gamma_end
beta_end = (1/p) * beta_start   

# Compute s_approx(t) values using the global control parameters
gammas = make_f_from_gap_shifted(s_vals, gaps_qaoa, power=2, start=gamma_start, end=gamma_end)
betas = make_f_from_gap_shifted(s_vals, gaps_qaoa, power=2, start=beta_start, end=beta_end)



# Plot
colors = plt.cm.viridis(np.linspace(0.1, 0.75, 8))

plt.figure(figsize=(8,6))

plt.plot(s_vals,gammas,color=colors[2],label=r'$\gamma$')
plt.plot(s_vals,betas,color=colors[-2],label=r'$\beta$')
plt.xlabel("s")
plt.ylabel("Angle (radians)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(f"plots/grover_sched_{target_state}.pdf")
plt.show()
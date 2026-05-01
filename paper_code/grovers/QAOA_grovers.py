# -*- coding: utf-8 -*-
"""

Solving Grovers problem with SGIR, RC, LR and Random QAOA

"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from itertools import product
from qiskit_aer import AerSimulator
import itertools
import pandas as pd
import os
import timeit
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh_tridiagonal
import random

random.seed(0)

repeats = 1   

shots = 10000 

optimization_level = 3 # compilation effort

p = 10

log_delta_beta_range = [-1.5,0.5]
log_delta_gamma_range = [-1,1]

# Parameter grid eg 11x11
disc = 2

method = 'SGIR-QAOA-0k-sym' # SGIR-QAOA-0k-sym-extrap random LR-QAOA RC-QAOA

min_n = 2
max_n = 16

power = 3
tanh_rate = 10 # (for RC-QAOA)

ms_name = 'Rand'


sim_method = "statevector" 
backend = AerSimulator()

t_vals = np.linspace(0, p, p)
s_vals = np.linspace(0, 1, p)

# Pauli matrices
X = np.array([[0,1],[1,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)
I = np.eye(2, dtype=complex)
I2 = np.eye(2, dtype=complex)


def grover_energy(x, target_state):
    """
    Energy of the true Grover oracle Hamiltonian:
    E(x) = 0 if x matches the marked state, else 1.
    """
    x_str = ''.join(str(int(b)) for b in x)
    return 0.0 if x_str == target_state else 1.0


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


def qaoa_circ(hamiltonian, gammas, betas, n_qubits):
    p = len(gammas)
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    for ii in range(p):
        for qbits, value in hamiltonian.items():
            if len(qbits) == 1:
                qc.rz(2*gammas[ii]*float(value), qbits[0])
        for qbits, value in hamiltonian.items():
            if len(qbits) == 2:
                qc.rzz(2*gammas[ii]*float(value), *qbits)
        qc.rx(-2*betas[ii], range(n_qubits)) # mixer
    qc = qc.reverse_bits()
    qc.measure_all()
    return qc

def tensor_kron(ops):
    out = ops[0]
    for A in ops[1:]:
        out = np.kron(out, A)
    return out


def build_transverse_field(n):
    dim = 2**n
    H = np.zeros((dim,dim), dtype=complex)
    for i in range(n):
        ops = [I2]*n
        ops[i] = X
        H += tensor_kron(ops)
    # convention: we want ground state |s> -> choose negative sign
    return -H


def Z_i(n, i):
    ops = [I2]*n
    ops[i] = Z
    return tensor_kron(ops)

def ZZ_ij(n, i, j):
    ops = [I2]*n
    ops[i] = Z
    ops[j] = Z
    return tensor_kron(ops)

def lowest_k_eigs(H, k=3):
    # dense small-matrix eigensolver; returns ascending eigenvalues
    # vals = eigh(H, eigvals_only=True)
    vals,_ = eigsh(H,k=k)
    return np.sort(vals)[:k]


# Modified function
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


def build_grover_oracle(n, target_state):
    dim = 2**n
    H = np.eye(dim, dtype=complex)
    
    # Index of the marked (target) state
    idx = int(target_state, 2)
    
    # Subtract projector onto marked state
    H[idx, idx] -= 1.0
    return H


def build_gap_symmetric_subspace(n, s_vals):
    """
    Compute spectral gaps for Grover + transverse-field mixer
    using the (n+1)-dimensional symmetric subspace.
    
    Args:
        n (int): number of qubits
        s_vals (array): values of s in [0,1]
    
    Returns:
        gaps (array): spectral gap at each s
    """
    gaps = np.zeros(len(s_vals))
    
    for j, s in enumerate(s_vals):
        # --- Diagonal: cost Hamiltonian ---
        # Cost = 0 for marked state (m=0), 1 for others
        diag = np.ones(n+1) * s
        diag[0] = 0
        
        # --- Off-diagonal: transverse-field Hamiltonian ---
        # Connects m <-> m+1
        m = np.arange(1, n+1)
        off_diag = (1-s) * np.sqrt(m * (n - m + 1))
        
        # --- Compute eigenvalues efficiently ---
        eigvals = eigh_tridiagonal(diag, off_diag, eigvals_only=True)
        eigvals.sort()
        
        # --- Gap between ground and first excited ---
        gaps[j] = eigvals[1] - eigvals[0]
    
    return gaps


# # create angle lists
log_delta_beta_range_ls = np.linspace(log_delta_beta_range[0],log_delta_beta_range[1],disc)
log_delta_gamma_range_ls = np.linspace(log_delta_gamma_range[0],log_delta_gamma_range[1],disc)


start_time = timeit.default_timer()  # Start timing



output_csv = f'results/big_{method}_{p}_b{log_delta_beta_range}_g{log_delta_gamma_range}_d{disc}_v{ms_name}.csv'
    

for nn in range(min_n,max_n+1,1):
    
    for ii in range(10):

        
        target_state = ''.join(random.choice(['0', '1']) for _ in range(nn))
        print(target_state)
        n = len(target_state)
        
        H_oracle = grover_hamiltonian_ising(target_state)
        
        hamiltonian = {
        tuple([i for i, p in enumerate(label) if p != "I"]): coeff
        for label, coeff in zip(H_oracle.paulis.to_labels(), H_oracle.coeffs)
    }
        
        # Normalization: scale all weights by the maximum absolute value (as in LR-QAOA paper)
        max_weight = max(abs(w) for w in hamiltonian.values()) if hamiltonian else 1.0
        
        hamiltonian = {
            term: w / max_weight
            for term, w in hamiltonian.items()
            if abs(w) > 1e-9
        }
        

        if method == 'SGIR-QAOA-0k':
            
            H_X = build_transverse_field(n)
            H_cost = build_grover_oracle(n, target_state)
            
            eig_X    = np.zeros((len(s_vals), 4)) # changed from 6 to 3 based off k
            gaps_qaoa = np.zeros(len(s_vals))

            for j, s in enumerate(s_vals):
                # Hproj_s = (1-s)*H0_proj + s*H_cost
                # HX_s    = (1-s)*H0_proj    + s*H_cost
                Hinst = (1-s)*H_X + s*H_cost
                
                HX_s = Hinst

                eig_X[j,:]    = lowest_k_eigs(HX_s, k=4)
                gaps_qaoa[j] = eig_X[j, 1] - eig_X[j, 0]
            
            # Index of minimum gap
            min_gap_idx = np.argmin(gaps_qaoa)
            # Minimum gap value
            min_gap = gaps_qaoa[min_gap_idx]
            # corresponding s value
            s_at_min_gap = s_vals[min_gap_idx]
            
            
        if method == 'RC-QAOA-anal':
            
            N = 2**n
            gaps_qaoa = np.zeros(len(s_vals))
            
            for j, s in enumerate(s_vals):
                gaps_qaoa[j] = np.sqrt(1 - 4*(1 - 1/N)*s*(1-s))
            
            # index of minimum gap
            min_gap_idx = np.argmin(gaps_qaoa)
            # minimum gap value
            min_gap = gaps_qaoa[min_gap_idx]
            # corresponding s value
            s_at_min_gap = s_vals[min_gap_idx]
            

        elif method == 'SGIR-QAOA-0k-sym':
            
            gaps_qaoa = build_gap_symmetric_subspace(n, s_vals)

            # Find minimum gap and its s
            min_gap_idx = np.argmin(gaps_qaoa)
            min_gap = gaps_qaoa[min_gap_idx]
            s_at_min_gap = s_vals[min_gap_idx]
            
        
        elif method == 'SGIR-QAOA-0k-sym-extrap':
            
            if n < 10:
                gaps_qaoa = build_gap_symmetric_subspace(n, s_vals)
                
            if n>9:
                gaps_qaoa = build_gap_symmetric_subspace(n, s_vals)
                gaps_qaoa[-1] = 0
                gaps_qaoa[-2] = 0

            # Find minimum gap and its s
            min_gap_idx = np.argmin(gaps_qaoa)
            min_gap = gaps_qaoa[min_gap_idx]
            s_at_min_gap = s_vals[min_gap_idx]
        
        log_delta_beta_ls = []
        log_delta_gamma_ls =[]
        x0_ls =[]
        Ps_ls = []
        AR_ls = []
        
        track = 0
        for log_delta_beta, log_delta_gamma in itertools.product(log_delta_beta_range_ls, log_delta_gamma_range_ls):
            track += 1
            
            if method == 'LR-QAOA':
                delta_beta = 10**(log_delta_beta)
                delta_gamma = 10**(log_delta_gamma)
                
                gammas = np.arange(1, p+1) * delta_gamma/p
                betas = np.arange(1, p+1)[::-1] * delta_beta/p
                
            
            elif method == 'SGIR-QAOA-0k-sym' or method == 'RC-QAOA' or method == 'RC-QAOA-anal' or method == 'SGIR-QAOA-0k-sym-extrap':
                
                beta_start = 10**(log_delta_beta)
                gamma_end = 10**(log_delta_gamma)
                
                gamma_start = (1/p) * gamma_end
                beta_end = (1/p) * beta_start   
                
                # Compute s_approx(t) values using the global control parameters
                gammas = make_f_from_gap_shifted(s_vals, gaps_qaoa, power=2, start=gamma_start, end=gamma_end)
                betas = make_f_from_gap_shifted(s_vals, gaps_qaoa, power=2, start=beta_start, end=beta_end)
                

            elif method == 'random':
                gammas = np.random.uniform(0, 2 * np.pi, p)
                betas = np.random.uniform(0, 2 * np.pi, p)
                  
            x0 = [val for pair in zip(gammas, betas) for val in pair]
            
            qc = qaoa_circ(hamiltonian, gammas, betas, n)   
            qc = transpile(qc, backend=backend)
        
            samples = backend.run(qc, shots=shots).result().get_counts()
            
            bitstrings = list(samples.keys())
            counts = list(samples.values())
            
            # # Calculate energies
            fvals = []
            for i in range(len(bitstrings)):
                # Convert bitstring to a 1D NumPy array
                x = np.array([int(bit) for bit in bitstrings[i]])
                xT = np.transpose(x)
                # Calculate energy = x^T * Q * x
                fvals.append(grover_energy(x, target_state)) 
            
            # Create DataFrame
            df = pd.DataFrame({
                'x': bitstrings,
                'prob': [c / shots for c in counts],
                'fval': fvals
            })
            
            # Sort descending by count
            df = df.sort_values(by='prob', ascending=False).reset_index(drop=True)
            
            E_avg = (df["prob"] * df["fval"]).sum()
            AR = E_avg # !!! patched, alter for AR here
        
            df = df.sort_values(by='fval', ascending=True).reset_index(drop=True)
            # print(df)
            
            # Calculate Ps (post-selection)
            good_df = df[df['fval'] == 0]
            Ps = good_df['prob'].sum()
            # print('Ps =',Ps)
            
            Ps_ls.append(Ps)
            AR_ls.append(AR)
            log_delta_beta_ls.append(log_delta_beta)
            log_delta_gamma_ls.append(log_delta_gamma)
            x0_ls.append(x0)
            
            print(f'Completed {track} out of {disc**2}')
            
        
        
        if method == 'RC-QAOA' or method == 'RC-QAOA-anal' or method == 'SGIR-QAOA-0k-sym' or method == 'SGIR-QAOA-0k-sym-extrap':
            
            print('Saving here')
            
            print('Min gap', min_gap,
            'S min', s_at_min_gap)
            
            df_angle_set = pd.DataFrame({
                'log_delta_beta': log_delta_beta_ls,
                'log_delta_gamma': log_delta_gamma_ls,
                'x0':x0_ls,
                'Ps_avg': Ps_ls,
                'AR': AR_ls,
                'Min gap': min_gap,
                'S min': s_at_min_gap
                # could add AR
            })
            
        else:
            df_angle_set = pd.DataFrame({
                'log_delta_beta': log_delta_beta_ls,
                'log_delta_gamma': log_delta_gamma_ls,
                'x0':x0_ls,
                'Ps_avg': Ps_ls,
                'AR': AR_ls
            })
        

        if method == 'GR-QAOA':
            path = f'results/raw_results/{target_state}__{method}_pow{power}_b{log_delta_beta_range}_g{log_delta_gamma_range}_d{disc}_p{p}.csv'
        if method == 'GR-QAOA_inv':
            path = f'results/raw_results/{target_state}_{method}_rate{tanh_rate}_b{log_delta_beta_range}_g{log_delta_gamma_range}_d{disc}_p{p}.csv'   
            
        else:
            path = f'results/raw_results/{target_state}_{method}_b{log_delta_beta_range}_g{log_delta_gamma_range}_d{disc}_p{p}.csv'
            

            
        # Save the DataFrame to a CSV file
        df_angle_set.to_csv(path, index=False, float_format='%.15g')
        
        
        # Now append to overall CSV
        
        # Sort df_angle_set by Ps_avg (descending)
        df_angle_set_sorted = df_angle_set.sort_values(by="Ps_avg", ascending=False)
        
        # Extract the best row (highest Ps_avg)
        best_row = df_angle_set_sorted.iloc[0].copy()
        best_row["n"] = nn
        best_row["marked"] = target_state
        

        
        # Convert to DataFrame for appending
        best_df = pd.DataFrame([best_row])
        
        
        if os.path.exists(output_csv):
            best_df.to_csv(output_csv, mode='a', header=False, index=False)   # append
        else:
            best_df.to_csv(output_csv, index=False)   # create new

end_time = timeit.default_timer()
Runtime = end_time - start_time
print('Runtime', Runtime)
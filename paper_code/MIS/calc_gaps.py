# -*- coding: utf-8 -*-
"""

MIS

"""

import numpy as np
from docplex.mp.model import Model
import pickle
import pandas as pd
import os
import timeit
import re
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import LinearOperator

# from qiskit_aer.noise import (
#     NoiseModel,
#     QuantumError,
#     ReadoutError,
#     depolarizing_error,
#     pauli_error,
#     thermal_relaxation_error,
# )

# import qiskit as qiskit

# print("Qiskit:", qiskit.__version__)

repeats = 1   

n_min = 2
n_max = 22


shots = 10000 

optimization_level = 3 # compilation effort

p = 10

log_delta_beta_range = [-1.5,0.5]
log_delta_gamma_range = [-1,1]

# beta_range = [0.1,1.2]
# gamma_range = [0.1,1.2]

degen_check = 'Y'

method = 'SGIR-QAOA-0k' # GR-QAOA random LR-QAOA SGIR-QAOA-0k
eig_meth = 'dense'
power = 3
tanh_rate = 10

disc = 11 # grid eg 11x11
penalty = 2000


sim_method = "statevector" #'density_matrix',"statevector","matrix_product_state",extended_stabilizer

# Create an empty noise model
# noise_model = NoiseModel()

# p_noise = 0.1
# error_1q = depolarizing_error(p_noise, 1)
# error_2q = depolarizing_error(p_noise, 2)
# # # Apply to ALL 1-qubit gates
# noise_model.add_all_qubit_quantum_error(error_1q, ['u', 'u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'sx'])
# # Apply to ALL 2-qubit gates
# noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz', 'ecr'])
 

# if sim_method == "statevector":
#     backend = AerSimulator()
# elif sim_method == "noisy":
#     backend = AerSimulator(noise_model=noise_model)



use_openqaoa = 'N'

weighted_inst = 'weighted' # unweighted


# folder containing instances
# folder containing instances
p_e = '04'
folder = "instances"


t_vals = np.linspace(0, p, p)
s_vals = np.linspace(0, 1, p)


# Pauli matrices
X = np.array([[0,1],[1,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)
I = np.eye(2, dtype=complex)
I2 = np.eye(2, dtype=complex)

def I_all(n):
    """n-qubit Identity matrix."""
    out = I
    for _ in range(n-1):
        out = np.kron(out, I)
    return out



def tensor_kron(ops):
    out = ops[0]
    for A in ops[1:]:
        out = np.kron(out, A)
    return out

def build_projector_driver(n):
    dim = 2**n
    psi0 = np.ones(dim) / np.sqrt(dim)           # |s>
    P = np.outer(psi0, psi0)
    return np.eye(dim, dtype=complex) - P       # H0 = I - |s><s|

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

def lowest_k_eigs(H, k=6):
    # dense small-matrix eigensolver; returns ascending eigenvalues
    vals = eigh(H, eigvals_only=True)
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


def make_Hs_operator(n, cost_diag, s):
    dim = 2**n
    idx = np.arange(dim)

    def H_mv(v):
        out = np.zeros_like(v)

        # cost term (diagonal)
        out += s * cost_diag * v

        # transverse-field mixer: H_X = -sum_i X_i
        if s != 1.0:
            mix = np.zeros_like(v)
            for i in range(n):
                mix += v[idx ^ (1 << i)]
            out += (1 - s) * (-mix)

        return out

    return LinearOperator(
        shape=(dim, dim),
        matvec=H_mv,
        dtype=np.float64
    )

def lowest_k_eigs_sparse(H, k=6):
    vals = eigsh(
        H,
        k=k,
        which="SA",                # smallest algebraic
        return_eigenvectors=False,
        tol=1e-10
    )
    return np.sort(vals)



def MIS_QUBO(G, penalty=2):
    # MIS model as a QUBO problem
    mdl = Model('MIS')
    num_vertices = G.number_of_nodes()

    x = {i: mdl.binary_var(name=f"x_{i}") for i in range(num_vertices)}
    mdl.minimize(-mdl.sum(x) + penalty * mdl.sum(
        x[i] * x[j] for (i, j) in G.edges
    ))
    return mdl

def cost(x, G):
    obj = 0
    for i, j in G.edges():
        if x[i] + x[j] == "11":
            obj += 2
    return - x.count("1") + obj

def is_independent_set(bitstring, G):
    # Map bitstring indices to graph nodes
    node_list = list(sorted(G.nodes))  # ensure consistent order
    selected_nodes = [node_list[i] for i, bit in enumerate(bitstring) if bit == '1']
    return G.subgraph(selected_nodes).number_of_edges() == 0

def post_select_mis(df, G):
    return df[df['x'].apply(lambda bs: is_independent_set(bs, G))].copy()


def mis_hamiltonian_from_graph(G, A=2.0, normalize=True):
    """
    Build MIS Ising Hamiltonian directly from an unweighted graph.

    Returns:
        hamiltonian: dict mapping tuples to coefficients
                     (i,)   -> local field h_i
                     (i,j)  -> coupling J_ij
    """
    hamiltonian = {}

    # Local fields
    for i in G.nodes():
        deg_i = G.degree(i)
        h_i = 0.5 - (A / 4.0) * deg_i
        if abs(h_i) > 1e-9:
            hamiltonian[(i,)] = h_i

    # Couplings
    for i, j in G.edges():
        J_ij = A / 4.0
        if abs(J_ij) > 1e-9:
            hamiltonian[(i, j)] = J_ij

    # Optional normalization (like your MaxCut code)
    if normalize:
        max_coeff = max(abs(w) for w in hamiltonian.values())
        hamiltonian = {k: v / max_coeff for k, v in hamiltonian.items()}

    return hamiltonian


def build_mis_cost_from_hamiltonian(n, hamiltonian):
    """
    Build MIS cost Hamiltonian from a QUBO-style Hamiltonian dict:
    hamiltonian = {(i,): h_i, (i,j): J_ij}
    """

    dim = 2**n
    H_cost = np.zeros((dim, dim), dtype=complex)

    for term, w in hamiltonian.items():

        # Linear term: x_i = (1 - Z_i)/2
        if len(term) == 1:
            i = term[0]
            H_cost += w * (I_all(n) - Z_i(n, i)) / 2.0

        # Quadratic term: x_i x_j = (1 - Z_i)(1 - Z_j)/4
        elif len(term) == 2:
            i, j = term
            H_cost += w * (
                I_all(n)
                - Z_i(n, i)
                - Z_i(n, j)
                + ZZ_ij(n, i, j)
            ) / 4.0

        else:
            raise ValueError("Only linear and quadratic terms supported")

    return H_cost


def precompute_mis_cost_diagonal(n, hamiltonian):
    """
    Diagonal entries of MIS H_cost in computational basis.
    hamiltonian = {(i,): h_i, (i,j): J_ij}
    """

    dim = 2**n
    E = np.zeros(dim, dtype=np.float64)

    for k in range(dim):
        e = 0.0

        for term, w in hamiltonian.items():

            # Compute Z_i values on the fly
            if len(term) == 1:
                i = term[0]
                zi = 1 if ((k >> i) & 1) == 0 else -1
                e += w * (1 - zi) / 2.0

            elif len(term) == 2:
                i, j = term
                zi = 1 if ((k >> i) & 1) == 0 else -1
                zj = 1 if ((k >> j) & 1) == 0 else -1
                e += w * (1 - zi - zj + zi * zj) / 4.0

            else:
                raise ValueError("Only linear and quadratic terms supported")

        E[k] = e

    return E


# # create angle lists
log_delta_beta_range_ls = np.linspace(log_delta_beta_range[0],log_delta_beta_range[1],disc)
log_delta_gamma_range_ls = np.linspace(log_delta_gamma_range[0],log_delta_gamma_range[1],disc)




start_time = timeit.default_timer()  # Start timing




if method == 'GR-QAOA':
    output_csv = f'results/{folder}_{sim_method}_no_renorm_pen{penalty}_{method}_2026_{p}_pow{power}_b{log_delta_beta_range}_g{log_delta_gamma_range}_d{disc}.csv'
elif method == 'SGIR-QAOA-0k':
    output_csv = f'results/{folder}_{sim_method}_no_renorm_pen{penalty}_{method}_{eig_meth}_2026_{p}_b{log_delta_beta_range}_g{log_delta_gamma_range}_d{disc}.csv'
else:
    output_csv = f'results/{folder}_{sim_method}_no_renorm_pen{penalty}_{method}_2026_{p}_b{log_delta_beta_range}_g{log_delta_gamma_range}_d{disc}.csv'
    
instances = sorted(
    [
        f for f in os.listdir(folder)
        if (match := re.search(r"n(\d+)", f))
        and int(match.group(1)) <= n_max
    ],
    key=lambda name: int(re.search(r"n(\d+)", name).group(1))
)


inst_ls = []
e3_e0_ls = []
e2_e0_ls = []
e1_e0_ls = []
gap_sel_ls = []
min_gap_ls = []
s_at_min_gap_ls = []


for instance in instances:
    file_path = os.path.join(folder, instance)
    
    print("-------------------------------")
    print(f"Processing instance: {instance}")
    print("-------------------------------")

    with open(file_path, "rb") as f:
        G = pickle.load(f)
    
    # extract instance number (the last digits before .gpickle)
    inst_match = re.search(r"_(\d+)\.gpickle$", instance)
    if inst_match:
        inst_num = int(inst_match.group(1))
    else:
        raise ValueError(f"No inst value found in CSV for {instance}")
        
    
    n = G.number_of_nodes()
    
    if n<n_min:
        continue
    
    if n>n_max:
        continue
    
    
    hamiltonian = mis_hamiltonian_from_graph(G, A=penalty)


    # Load solutions CSV
    df = pd.read_csv(f'{folder}/mis_solutions_gurobi.csv')

    # Normalise filename if necessary
    match_df = df[df['filename'] == instance]
    if match_df.empty:
        # Try matching without extension
        base_name = os.path.splitext(instance)[0]
        match_df = df[df['filename'] == base_name]
    if match_df.empty:
        raise ValueError(f"No solution found in CSV for {instance}")
        

    soln_str = match_df['solution_nodes'].iloc[0]
    soln_list = list(map(int, soln_str.split()))
    min_energy = len(soln_list) * -1
    
    
    # # Now add to lists - n and inst, gap e2-e0, gap e1-e0, gap selected, min gap, s_at_min_gap
    
    
    
    if method == 'SGIR-QAOA-0k':
        
        counter = 0
        
        if eig_meth == 'dense':
            H_X = build_transverse_field(n)
            H_cost = build_mis_cost_from_hamiltonian(n,hamiltonian)
            
            
            eig_X    = np.zeros((len(s_vals), 4))
            gaps_qaoa_1 = np.zeros(len(s_vals))
            gaps_qaoa_2 = np.zeros(len(s_vals))
            gaps_qaoa_3 = np.zeros(len(s_vals))
            
            start_time_eig = timeit.default_timer()
    
            for j, s in enumerate(s_vals):

                # Hproj_s = (1-s)*H0_proj + s*H_cost
                # HX_s    = (1-s)*H0_proj    + s*H_cost
                Hinst = (1-s)*H_X + s*H_cost
                
                HX_s = Hinst
    
                eig_X[j,:]    = lowest_k_eigs(HX_s, k=4) # !!! was 6 previously
                
                # !!! could be smarter here and check for a degeneracy between e0 and e1
                # compute gap between E2 and E0
                gaps_qaoa_3[j] = eig_X[j, 3] - eig_X[j, 0]
                gaps_qaoa_2[j] = eig_X[j, 2] - eig_X[j, 0]
                gaps_qaoa_1[j] = eig_X[j, 1] - eig_X[j, 0]
                
                
            if degen_check == 'Y':
                if abs(eig_X[-1][0] - eig_X[-1][1]) < 1e-4 and abs(eig_X[-1][0] - eig_X[-1][2]) < 1e-4:
                    gaps_qaoa = gaps_qaoa_3
                    gap_sel = 'e3-e0'
                
                elif abs(eig_X[-1][0] - eig_X[-1][1]) < 1e-4:
                    gaps_qaoa = gaps_qaoa_2
                    gap_sel = 'e2-e0'
                
                else:
                    gaps_qaoa = gaps_qaoa_1
                    gap_sel = 'e1-e0'
                
            else:
                gaps_qaoa = gaps_qaoa_2
                gap_sel = 'e2-e0'
            
            # find minimum gap and its s
            min_gap_idx = np.argmin(gaps_qaoa)
            min_gap = gaps_qaoa[min_gap_idx]
            s_at_min_gap = s_vals[min_gap_idx]
            
            end_time_eig = timeit.default_timer()
            
            Runtime_eig = end_time_eig - start_time_eig
            print("Time taken to find gap for all s:", Runtime_eig)
            print("Eigenvalues are:", eig_X[-1])
            
            # # Now add to lists - n and inst, gap e2-e0, gap e1-e0, gap selected, min gap, s_at_min_gap
            inst_ls.append(instance)
            e3_e0_ls.append(gaps_qaoa_3)
            e2_e0_ls.append(gaps_qaoa_2)
            e1_e0_ls.append(gaps_qaoa_1)
            gap_sel_ls.append(gap_sel)
            min_gap_ls.append(min_gap)
            s_at_min_gap_ls.append(s_at_min_gap)
            
            
        
        
        elif eig_meth == 'matrix_free':
            eig_X = np.zeros((len(s_vals), 4))
            gaps_qaoa_1 = np.zeros(len(s_vals))
            gaps_qaoa_2 = np.zeros(len(s_vals))
            gaps_qaoa_3 = np.zeros(len(s_vals))

            cost_diag = precompute_mis_cost_diagonal(n, hamiltonian)
            
            start_time_eig = timeit.default_timer()
            
            for j, s in enumerate(s_vals):
                Hs = make_Hs_operator(n, cost_diag, s)
                eig_X[j, :] = lowest_k_eigs_sparse(Hs, k=4)
                gaps_qaoa_3[j] = eig_X[j, 3] - eig_X[j, 0]
                gaps_qaoa_2[j] = eig_X[j, 2] - eig_X[j, 0]
                gaps_qaoa_1[j] = eig_X[j, 1] - eig_X[j, 0]
                
            if degen_check == 'Y':
                if abs(eig_X[-1][0] - eig_X[-1][1]) < 1e-4 and abs(eig_X[-1][0] - eig_X[-1][2]) < 1e-4:
                    gaps_qaoa = gaps_qaoa_3
                    gap_sel = 'e3-e0'
                
                elif abs(eig_X[-1][0] - eig_X[-1][1]) < 1e-4:
                    gaps_qaoa = gaps_qaoa_2
                    gap_sel = 'e2-e0'
                
                else:
                    gaps_qaoa = gaps_qaoa_1
                    gap_sel = 'e1-e0'
                
            else:
                gaps_qaoa = gaps_qaoa_2
                gap_sel = 'e2-e0'
            
            # find minimum gap and its s
            min_gap_idx = np.argmin(gaps_qaoa)
            min_gap = gaps_qaoa[min_gap_idx]
            s_at_min_gap = s_vals[min_gap_idx]
            
            end_time_eig = timeit.default_timer()
            
            Runtime_eig = end_time_eig - start_time_eig
            print("Time taken to find gap for all s:", Runtime_eig)
            print("Eigenvalues are:", eig_X[-1])
            
            # # Now add to lists - n and inst, gap e2-e0, gap e1-e0, gap selected, min gap, s_at_min_gap
            inst_ls.append(instance)
            e3_e0_ls.append(gaps_qaoa_3)
            e2_e0_ls.append(gaps_qaoa_2)
            e1_e0_ls.append(gaps_qaoa_1)
            gap_sel_ls.append(gap_sel)
            min_gap_ls.append(min_gap)
            s_at_min_gap_ls.append(s_at_min_gap)
            
            
            
           
    # Now save a df - n and inst, gap e2-e0, gap e1-e0, gap selected
    df = pd.DataFrame({
    'instance': inst_ls,
    'gap_e3_e0': e3_e0_ls,
    'gap_e2_e0': e2_e0_ls,
    'gap_e1_e0': e1_e0_ls,
    'gap_selected': gap_sel_ls,
    'min_gap': min_gap_ls,
    's_at_min_gap': s_at_min_gap_ls,
})



    path = f'results/MIS_gaps_p{p}_degChck{degen_check}_{eig_meth}.csv'
        

    # Save the DataFrame to a CSV file
    df.to_csv(path, index=False, float_format='%.15g')
    
        
        
        
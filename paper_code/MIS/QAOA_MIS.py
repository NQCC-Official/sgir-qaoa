# -*- coding: utf-8 -*-
"""

MIS - at constant depth p

"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import itertools
import pickle
import pandas as pd
import os
import timeit
import re
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import LinearOperator

import qiskit as qiskit

print("Qiskit:", qiskit.__version__)

repeats = 1   

n_min = 24
n_max = 42

shots = 10000 

optimization_level = 3 # compilation effort

p = 10

log_delta_beta_range = [-1.5,0.5]
log_delta_gamma_range = [-1,1]


method = 'SGIR-QAOA-0k' # random LR-QAOA SGIR-QAOA-0k
eig_meth = 'matrix_free_extrap_s0_gap4all_s1_gap0' #
power = 3
tanh_rate = 10
degen_check = 'N' # accNO

disc = 11 # grid eg 11x11
penalty = 2000

sim_method = "statevector" #"matrix_product_state"

matrix_product_state_max_bond_dimension=70 # was 50 ,
matrix_product_state_truncation_threshold=1e-6


if sim_method == "statevector":
    backend = AerSimulator()
elif sim_method =='matrix_product_state':
    backend = AerSimulator(method=sim_method,
    matrix_product_state_max_bond_dimension=matrix_product_state_max_bond_dimension, # was 50 ,
    matrix_product_state_truncation_threshold=matrix_product_state_truncation_threshold# was 
    # mps_log_data=True,
)
else:
    backend = AerSimulator(method=sim_method)
use_openqaoa = 'N'


weighted_inst = 'weighted' # unweighted


# folder containing instances
p_e = 'd3'
folder = "instances_d3"


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
        qc.rx(-2*betas[ii], range(n_qubits))
    qc = qc.reverse_bits()
    qc.measure_all()
    return qc



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

if sim_method == "matrix_product_state":
    if method == 'SGIR-QAOA-0k':
        output_csv = f'results/{folder}_{sim_method}_degenCheck{degen_check}_no_renorm_pen{penalty}_{method}_{eig_meth}_2026_{p}_b{log_delta_beta_range}_g{log_delta_gamma_range}_d{disc}_bdim{matrix_product_state_max_bond_dimension}_thres{matrix_product_state_truncation_threshold}.csv'
    else:
        output_csv = f'results/{folder}_{sim_method}_degenCheck{degen_check}_no_renorm_pen{penalty}_{method}_2026_{p}_b{log_delta_beta_range}_g{log_delta_gamma_range}_d{disc}_bdim{matrix_product_state_max_bond_dimension}_thres{matrix_product_state_truncation_threshold}.csv'


else:
    if method == 'SGIR-QAOA-0k':
        output_csv = f'results/{folder}_{sim_method}_degenCheck{degen_check}_no_renorm_pen{penalty}_{method}_{eig_meth}_2026_{p}_b{log_delta_beta_range}_g{log_delta_gamma_range}_d{disc}.csv'
    else:
        output_csv = f'results/{folder}_{sim_method}_degenCheck{degen_check}_no_renorm_pen{penalty}_{method}_2026_{p}_b{log_delta_beta_range}_g{log_delta_gamma_range}_d{disc}.csv'
    
# Collect and sort by n
instances = sorted(
    [f for f in os.listdir(folder) if re.search(r"n(\d+)", f)],
    key=lambda name: int(re.search(r"n(\d+)", name).group(1))
)

counter = 0

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
    
    min_gap_ls = []
    s_at_min_gap_ls = []
    e5_e0_ls = []
    e4_e0_ls = []
    e3_e0_ls = []
    e2_e0_ls = []
    e1_e0_ls = []
    gap_sel_ls = []
    
    if method == 'SGIR-QAOA-0k':
        
        
        if eig_meth == 'dense':
            H_X = build_transverse_field(n)
            H_cost = build_mis_cost_from_hamiltonian(n,hamiltonian)
            
            
            eig_X    = np.zeros((len(s_vals), 3))
            gaps_qaoa = np.zeros(len(s_vals))
            
            start_time_eig = timeit.default_timer()
    
            for j, s in enumerate(s_vals):

                # Hproj_s = (1-s)*H0_proj + s*H_cost
                # HX_s    = (1-s)*H0_proj    + s*H_cost
                Hinst = (1-s)*H_X + s*H_cost
                
                HX_s = Hinst
    
                eig_X[j,:]    = lowest_k_eigs(HX_s, k=3) 
                
                # compute gap between E2 and E0
                gaps_qaoa[j] = eig_X[j, 2] - eig_X[j, 0]
                
             # find minimum gap and its s
            min_gap_idx = np.argmin(gaps_qaoa)
            min_gap = gaps_qaoa[min_gap_idx]
            s_at_min_gap = s_vals[min_gap_idx]

            
            end_time_eig = timeit.default_timer()
            Runtime_eig = end_time_eig - start_time_eig
            print('Time taken to find gap for all s', Runtime_eig)
        
        
        elif eig_meth == 'matrix_free':
            eig_X = np.zeros((len(s_vals), 6))
            gaps_qaoa_1 = np.zeros(len(s_vals))
            gaps_qaoa_2 = np.zeros(len(s_vals))
            gaps_qaoa_3 = np.zeros(len(s_vals))
            gaps_qaoa_4 = np.zeros(len(s_vals))
            gaps_qaoa_5 = np.zeros(len(s_vals))
            gaps_qaoa = np.zeros(len(s_vals))

            cost_diag = precompute_mis_cost_diagonal(n, hamiltonian)
            
            start_time_eig = timeit.default_timer()
            
            for j, s in enumerate(s_vals):
                Hs = make_Hs_operator(n, cost_diag, s)
                eig_X[j, :] = lowest_k_eigs_sparse(Hs, k=6)

                if degen_check =='Y':
                    gaps_qaoa_5[j] = eig_X[j, 5] - eig_X[j, 0]
                    gaps_qaoa_4[j] = eig_X[j, 4] - eig_X[j, 0]
                    gaps_qaoa_3[j] = eig_X[j, 3] - eig_X[j, 0]
                    gaps_qaoa_2[j] = eig_X[j, 2] - eig_X[j, 0]
                    gaps_qaoa_1[j] = eig_X[j, 1] - eig_X[j, 0]
                else:
                    gaps_qaoa[j] = eig_X[j, 2] - eig_X[j, 0]


            if degen_check == 'Y':
                if abs(eig_X[-1][0] - eig_X[-1][1]) < 1e-4 and abs(eig_X[-1][0] - eig_X[-1][2]) < 1e-4 and abs(eig_X[-1][0] - eig_X[-1][3]) < 1e-4 and abs(eig_X[-1][0] - eig_X[-1][4]) < 1e-4:
                    gaps_qaoa = gaps_qaoa_5
                    gap_sel = 'e5-e0'
                
                if abs(eig_X[-1][0] - eig_X[-1][1]) < 1e-4 and abs(eig_X[-1][0] - eig_X[-1][2]) < 1e-4 and abs(eig_X[-1][0] - eig_X[-1][3]) < 1e-4:
                    gaps_qaoa = gaps_qaoa_4
                    gap_sel = 'e4-e0'
                
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
                gaps_qaoa = gaps_qaoa
                gap_sel = 'e2-e0'
            
            # find minimum gap and its s
            min_gap_idx = np.argmin(gaps_qaoa)
            min_gap = gaps_qaoa[min_gap_idx]
            s_at_min_gap = s_vals[min_gap_idx]
            
            end_time_eig = timeit.default_timer()
            
            Runtime_eig = end_time_eig - start_time_eig
            print("Time taken to find gap for all s:", Runtime_eig)
            print("Eigenvalues are:", eig_X[-1])

            e5_e0_ls.append(gaps_qaoa_5)
            e4_e0_ls.append(gaps_qaoa_4)
            e3_e0_ls.append(gaps_qaoa_3)
            e2_e0_ls.append(gaps_qaoa_2)
            e1_e0_ls.append(gaps_qaoa_1)
            gap_sel_ls.append(gap_sel)
            min_gap_ls.append(min_gap)


        elif eig_meth == 'matrix_free_extrap':
            # ---- Load average gap vs s ----
            df_avg_gap = pd.read_csv("results/avg_gap_vs_s_degChckN.csv")  # columns: s, avg_gap
            s_vals = df_avg_gap["s"].values
            avg_gaps = df_avg_gap["avg_gap"].values
        
            if n <= 12:
                # ---- Load MIS gap arrays for small n ----
                df_mis = pd.read_csv("results/MIS_gaps_p10_degChckY_matrix-free.csv")
                
                # Parse gap arrays
                def parse_array(arr_str):
                    arr_str = arr_str.replace("\n", " ").strip().strip('"').strip("'").strip("[]")
                    return np.fromstring(arr_str, sep=' ')
        
                def select_gap_array(row):
                    if row["gap_selected"] == "e1-e0":
                        return parse_array(row["gap_e1_e0"])
                    elif row["gap_selected"] == "e2-e0":
                        return parse_array(row["gap_e2_e0"])
                    elif row["gap_selected"] == "e3-e0":
                        return parse_array(row["gap_e3_e0"])
                    else:
                        return None
        
                df_mis["gap_array"] = df_mis.apply(select_gap_array, axis=1)
                # df_mis["n"] = df_mis["instance"].str.extract(r"mis_n(\d+)_", expand=False).astype(int)
                df_mis["n"] = df_mis["instance"].str.extract(r"mis_n(\d+)_", expand=False)
                df_mis = df_mis.dropna(subset=["n"])   # remove rows where regex failed
                df_mis["n"] = df_mis["n"].astype(int)
                
                # Take first instance for the target n
                gap_row = df_mis[df_mis["n"] == n].iloc[0]
                gaps_qaoa = gap_row["gap_array"].copy()
            else:
                # ---- Load extrapolated minimum gap ----
                df_extrap = pd.read_csv("results/extrapolated_min_gaps_degChckN.csv")  # columns: n, gap
                extrap_min_gap = df_extrap.loc[df_extrap["n"] == n, "gap"].values[0]
        
                # ---- Replace last value (s=1) with extrapolated min gap ----
                gaps_qaoa = avg_gaps.copy()
                gaps_qaoa[-1] = extrap_min_gap
        
            # ---- Convert to list for display ----
            gaps_qaoa_list = gaps_qaoa.tolist()
            print("gaps_qaoa =", gaps_qaoa_list)
        
            # ---- Find minimum gap and its s ----
            min_gap_idx = np.argmin(gaps_qaoa)
            min_gap = gaps_qaoa[min_gap_idx]
            s_at_min_gap = s_vals[min_gap_idx]
        
            end_time_eig = timeit.default_timer()
            


        
        elif eig_meth == 'matrix_free_extrap_s0_gap4all':
            # ---- Load average gap vs s ----
            df_avg_gap = pd.read_csv("instances_d3/avg_gap_vs_s_degChckN.csv")  # columns: s, avg_gap
            s_vals = df_avg_gap["s"].values
            avg_gaps = df_avg_gap["avg_gap"].values
        
            if n <= 12:
                # ---- Load MIS gap arrays for small n ----
                df_mis = pd.read_csv("instances_d3/MIS_gaps_p10_degChckY_matrix-free.csv")
                
                # Parse gap arrays
                def parse_array(arr_str):
                    arr_str = arr_str.replace("\n", " ").strip().strip('"').strip("'").strip("[]")
                    return np.fromstring(arr_str, sep=' ')
        
                def select_gap_array(row):
                    if row["gap_selected"] == "e1-e0":
                        return parse_array(row["gap_e1_e0"])
                    elif row["gap_selected"] == "e2-e0":
                        return parse_array(row["gap_e2_e0"])
                    elif row["gap_selected"] == "e3-e0":
                        return parse_array(row["gap_e3_e0"])
                    else:
                        return None
        
                df_mis["gap_array"] = df_mis.apply(select_gap_array, axis=1)
                # df_mis["n"] = df_mis["instance"].str.extract(r"mis_n(\d+)_", expand=False).astype(int)
                df_mis["n"] = df_mis["instance"].str.extract(r"mis_n(\d+)_", expand=False)
                df_mis = df_mis.dropna(subset=["n"])   # remove rows where regex failed
                df_mis["n"] = df_mis["n"].astype(int)
                
                # Take first instance for the target n
                gap_row = df_mis[df_mis["n"] == n].iloc[0]
                gaps_qaoa = gap_row["gap_array"].copy()


                gaps_qaoa[0] = 4
            else:
                # ---- Load extrapolated minimum gap ----
                df_extrap = pd.read_csv("results/extrapolated_min_gaps_degChckN.csv")  # columns: n, gap
                extrap_min_gap = df_extrap.loc[df_extrap["n"] == n, "gap"].values[0]
        
                # ---- Replace last value (s=1) with extrapolated min gap ----
                gaps_qaoa = avg_gaps.copy()
                gaps_qaoa[-1] = extrap_min_gap

                
                gaps_qaoa[0] = 4
        
            # ---- Convert to list for display ----
            gaps_qaoa_list = gaps_qaoa.tolist()
            print("gaps_qaoa =", gaps_qaoa_list)
        
            # ---- Find minimum gap and its s ----
            min_gap_idx = np.argmin(gaps_qaoa)
            min_gap = gaps_qaoa[min_gap_idx]
            s_at_min_gap = s_vals[min_gap_idx]
        
            end_time_eig = timeit.default_timer()




        elif eig_meth == 'matrix_free_extrap_s0_gap4all_s1_gap0':
            # ---- Load average gap vs s ----
            df_avg_gap = pd.read_csv("instances_d3/avg_gap_vs_s_degChckN.csv")  # columns: s, avg_gap
            s_vals = df_avg_gap["s"].values
            avg_gaps = df_avg_gap["avg_gap"].values
        
            if n <= 12:
                # ---- Load MIS gap arrays for small n ----
                df_mis = pd.read_csv("instances_d3/MIS_gaps_p10_degChckY_matrix-free.csv")
                
                # Parse gap arrays
                def parse_array(arr_str):
                    arr_str = arr_str.replace("\n", " ").strip().strip('"').strip("'").strip("[]")
                    return np.fromstring(arr_str, sep=' ')
        
                def select_gap_array(row):
                    if row["gap_selected"] == "e1-e0":
                        return parse_array(row["gap_e1_e0"])
                    elif row["gap_selected"] == "e2-e0":
                        return parse_array(row["gap_e2_e0"])
                    elif row["gap_selected"] == "e3-e0":
                        return parse_array(row["gap_e3_e0"])
                    else:
                        return None
        
                df_mis["gap_array"] = df_mis.apply(select_gap_array, axis=1)
                # df_mis["n"] = df_mis["instance"].str.extract(r"mis_n(\d+)_", expand=False).astype(int)
                df_mis["n"] = df_mis["instance"].str.extract(r"_n(\d+)_", expand=False).astype(int)
                df_mis = df_mis.dropna(subset=["n"])   # remove rows where regex failed
                df_mis["n"] = df_mis["n"].astype(int)
                
                # Take first instance for the target n
                gap_row = df_mis[df_mis["n"] == n].iloc[0]
                gaps_qaoa = gap_row["gap_array"].copy()


                gaps_qaoa[0] = 4
                
            else:
        
                # # ---- Replace last value (s=1) with extrapolated min gap ----
                gaps_qaoa = avg_gaps.copy()
                # gaps_qaoa[-1] = extrap_min_gap

                
                gaps_qaoa[0] = 4
                gaps_qaoa[-1] = 0
        
            # ---- Convert to list for display ----
            gaps_qaoa_list = gaps_qaoa.tolist()
            print("gaps_qaoa =", gaps_qaoa_list)
        
            # ---- Find minimum gap and its s ----
            min_gap_idx = np.argmin(gaps_qaoa)
            min_gap = gaps_qaoa[min_gap_idx]
            s_at_min_gap = s_vals[min_gap_idx]
        
            end_time_eig = timeit.default_timer()
        
        
    
        
        

        elif eig_meth == 'matrix_free_extrap_s0_gap4extrap':
            # ---- Load average gap vs s ----
            df_avg_gap = pd.read_csv("results/avg_gap_vs_s_degChckN.csv")  # columns: s, avg_gap
            s_vals = df_avg_gap["s"].values
            avg_gaps = df_avg_gap["avg_gap"].values
        
            if n <= 12:
                # ---- Load MIS gap arrays for small n ----
                df_mis = pd.read_csv("results/MIS_gaps_p10_degChckY_matrix-free.csv")
                
                # Parse gap arrays
                def parse_array(arr_str):
                    arr_str = arr_str.replace("\n", " ").strip().strip('"').strip("'").strip("[]")
                    return np.fromstring(arr_str, sep=' ')
        
                def select_gap_array(row):
                    if row["gap_selected"] == "e1-e0":
                        return parse_array(row["gap_e1_e0"])
                    elif row["gap_selected"] == "e2-e0":
                        return parse_array(row["gap_e2_e0"])
                    elif row["gap_selected"] == "e3-e0":
                        return parse_array(row["gap_e3_e0"])
                    else:
                        return None
        
                df_mis["gap_array"] = df_mis.apply(select_gap_array, axis=1)
                # df_mis["n"] = df_mis["instance"].str.extract(r"mis_n(\d+)_", expand=False).astype(int)
                df_mis["n"] = df_mis["instance"].str.extract(r"_n(\d+)_", expand=False).astype(int)
                df_mis = df_mis.dropna(subset=["n"])   # remove rows where regex failed
                df_mis["n"] = df_mis["n"].astype(int)
                
                # Take first instance for the target n
                gap_row = df_mis[df_mis["n"] == n].iloc[0]
                gaps_qaoa = gap_row["gap_array"].copy()
            else:
                # ---- Load extrapolated minimum gap ----
                df_extrap = pd.read_csv("results/extrapolated_min_gaps_degChckN.csv")  # columns: n, gap
                extrap_min_gap = df_extrap.loc[df_extrap["n"] == n, "gap"].values[0]
        
                # ---- Replace last value (s=1) with extrapolated min gap ----
                gaps_qaoa = avg_gaps.copy()
                gaps_qaoa[-1] = extrap_min_gap

                
                gaps_qaoa[0] = 4
        
            # ---- Convert to list for display ----
            gaps_qaoa_list = gaps_qaoa.tolist()
            print("gaps_qaoa =", gaps_qaoa_list)
        
            # ---- Find minimum gap and its s ----
            min_gap_idx = np.argmin(gaps_qaoa)
            min_gap = gaps_qaoa[min_gap_idx]
            s_at_min_gap = s_vals[min_gap_idx]
        
            end_time_eig = timeit.default_timer()
            


        
        elif counter == 0 and eig_meth == 'matrix_free_aprx':
            eig_X = np.zeros((len(s_vals), 3))
            gaps_qaoa_1 = np.zeros(len(s_vals))
            gaps_qaoa_2 = np.zeros(len(s_vals))

            cost_diag = precompute_mis_cost_diagonal(n, hamiltonian)
            
            start_time_eig = timeit.default_timer()
            
            for j, s in enumerate(s_vals):
                Hs = make_Hs_operator(n, cost_diag, s)
                eig_X[j, :] = lowest_k_eigs_sparse(Hs, k=3)
                gaps_qaoa_2[j] = eig_X[j, 2] - eig_X[j, 0]
                gaps_qaoa_1[j] = eig_X[j, 1] - eig_X[j, 0]
                
            if eig_X[-1][0] == eig_X[-1][1]:
                gaps_qaoa = gaps_qaoa_2
            else:
                gaps_qaoa = gaps_qaoa_1
            
            # find minimum gap and its s
            min_gap_idx = np.argmin(gaps_qaoa)
            min_gap = gaps_qaoa[min_gap_idx]
            s_at_min_gap = s_vals[min_gap_idx]
            
            end_time_eig = timeit.default_timer()
            
            Runtime_eig = end_time_eig - start_time_eig
            print("Time taken to find gap for all s:", Runtime_eig)
            print("Eigenvalues are:", eig_X[-1])
            
            counter += 1

    
    log_delta_beta_ls = []
    log_delta_gamma_ls =[]
    x0_ls =[]
    Ps_avg_ls = []
    Ps_std_ls = []
    Ps_pre_avg_ls = []
    Ps_pre_std_ls = []
    # AR_avg_ls = []
    # AR_std_ls = []
    
    
    AR_ls = []
    
    track = 0
    for log_delta_beta, log_delta_gamma in itertools.product(log_delta_beta_range_ls, log_delta_gamma_range_ls):
        track += 1
        
        if method == 'LR-QAOA':
            delta_beta = 10**(log_delta_beta)
            delta_gamma = 10**(log_delta_gamma)
            
            gammas = np.arange(1, p+1) * delta_gamma/p
            betas = np.arange(1, p+1)[::-1] * delta_beta/p
        

        
        elif method == 'SGIR-QAOA-0k':
                
                
            beta_start = 10**(log_delta_beta)
            gamma_end = 10**(log_delta_gamma)
            
            gamma_start = (1/p) * gamma_end
            beta_end = (1/p) * beta_start   
            
            # Compute s_approx(t) values using the global control parameters
            gammas = make_f_from_gap_shifted(s_vals, gaps_qaoa, power=2, start=gamma_start, end=gamma_end)
            betas = make_f_from_gap_shifted(s_vals, gaps_qaoa, power=2, start=beta_start, end=beta_end)
            
            # plt.plot(t_vals,gammas)
            # plt.plot(t_vals,betas)
            # plt.show()
            
            
          
        elif method == 'random':
            gammas = np.random.uniform(0, 2 * np.pi, p)
            
            betas = np.random.uniform(0, 2 * np.pi, p)
            
            
    
        x0 = [val for pair in zip(gammas, betas) for val in pair]
        

        qc = qaoa_circ(hamiltonian, gammas, betas, n)
            
            
        qc = transpile(qc, backend=backend)
    
        Ps_list = []
        Ps_pre_list = []
        # AR_list = []
        angle_list = []
        for j in range(repeats):
            
            samples = backend.run(qc, shots=shots).result().get_counts()
            
            bitstrings = list(samples.keys())
            counts = list(samples.values())
            
            # # Calculate energies
            fvals = [cost(bitstrings[i], G) for i in range(len(bitstrings))]
            
            # Create DataFrame
            df = pd.DataFrame({
                'x': bitstrings,
                'prob': [c / shots for c in counts],
                'fval': fvals
            })
            
            # Sort descending by count
            df = df.sort_values(by='prob', ascending=False).reset_index(drop=True)
            
            
            df = df.sort_values(by='fval', ascending=True).reset_index(drop=True)
            df = post_select_mis(df, G)
            df.rename(columns={'prob': 'Ps_pre'}, inplace=True)
    
            total_prob = df['Ps_pre'].sum()
            df['Ps'] = df['Ps_pre']  #/ total_prob
    
            good_df = df[df['fval'] == min_energy]
            Ps = good_df['Ps'].sum()
            Ps_pre = good_df['Ps_pre'].sum()        
            

            # # Find the cost (avg energy) of this solution
            # weighted_avg = (df['counts'] * df['fval']).sum() / df['counts'].sum()
            # print('Cost value:', weighted_avg)
            E_avg = (df['Ps'] * df["fval"]).sum()
            AR = E_avg / min_energy
        
            df = df.sort_values(by='fval', ascending=True).reset_index(drop=True)
            # print(df)
            

            # print('Ps =',Ps)
            Ps_list.append(Ps)
            Ps_pre_list.append(Ps_pre)
            angle_list.append(x0)
            
            
            Ps_avg = np.mean(Ps_list)
            Ps_std = np.std(Ps_list)
            
            Ps_pre_avg = np.mean(Ps_pre_list)
            Ps_pre_std = np.std(Ps_pre_list)
            
        
        AR_ls.append(AR)
        log_delta_beta_ls.append(log_delta_beta)
        log_delta_gamma_ls.append(log_delta_gamma)
        x0_ls.append(x0)
        Ps_avg_ls.append(Ps_avg)
        Ps_std_ls.append(Ps_std)
        Ps_pre_avg_ls.append(Ps_pre_avg)
        Ps_pre_std_ls.append(Ps_pre_std)
        
        print(f'Completed {track} out of {disc**2}')
        
    
    
    
    

    
    
    if method == 'SGIR-QAOA-0k' or method == 'SGIR-QAOA-0k-eigh':
            df_angle_set = pd.DataFrame({
                'log_delta_beta': log_delta_beta_ls,
                'log_delta_gamma': log_delta_gamma_ls,
                'x0':x0_ls,
                'Ps_avg': Ps_avg_ls,
                'Ps_std': Ps_std_ls,
                'Ps_pre_avg': Ps_pre_avg_ls,
                'Ps_pre_std': Ps_pre_std_ls,
                'AR': AR_ls,
                'Min gap': min_gap,
                'S min': s_at_min_gap,
                'ham':[hamiltonian]* len(Ps_avg_ls),
                'gaps':[gaps_qaoa]* len(Ps_avg_ls)
                # could add AR
            })

            if degen_check =='Y':

                df_angle_set = pd.DataFrame({
                'log_delta_beta': log_delta_beta_ls,
                'log_delta_gamma': log_delta_gamma_ls,
                'x0':x0_ls,
                'Ps_avg': Ps_avg_ls,
                'Ps_std': Ps_std_ls,
                'Ps_pre_avg': Ps_pre_avg_ls,
                'Ps_pre_std': Ps_pre_std_ls,
                'AR': AR_ls,
                'Min gap': min_gap,
                'S min': s_at_min_gap,
                'ham':[hamiltonian]* len(Ps_avg_ls),
                'gap_e5_e0': e5_e0_ls* len(Ps_avg_ls),
                'gap_e4_e0': e4_e0_ls* len(Ps_avg_ls),
                'gap_e3_e0': e3_e0_ls* len(Ps_avg_ls),
                'gap_e2_e0': e2_e0_ls* len(Ps_avg_ls),
                'gap_e1_e0': e1_e0_ls* len(Ps_avg_ls),
                'gap_selected': gap_sel_ls* len(Ps_avg_ls),
                # could add AR
            })
                

    
            
    else:
        df_angle_set = pd.DataFrame({
            'log_delta_beta': log_delta_beta_ls,
            'log_delta_gamma': log_delta_gamma_ls,
            'x0':x0_ls,
            'Ps_avg': Ps_avg_ls,
            'Ps_std': Ps_std_ls,
            'Ps_pre_avg': Ps_pre_avg_ls,
            'Ps_pre_std': Ps_pre_std_ls,
            'AR': AR_ls,
            'ham':[hamiltonian]* len(Ps_avg_ls)
        })
    
    
    if weighted_inst == 'weighted':
        if method == 'SGIR-QAOA-0k':
            path = f'results/raw_results/n{n}_{p_e}_{sim_method}_degenCheck{degen_check}_no_renorm_pen{penalty}_{method}_{eig_meth}_rate{tanh_rate}_b{log_delta_beta_range}_g{log_delta_gamma_range}_d{disc}_p{p}.csv'   
        
        else:
            path = f'results/raw_results/n{n}_{p_e}_{sim_method}_degenCheck{degen_check}_no_renorm_pen{penalty}_{method}_b{log_delta_beta_range}_g{log_delta_gamma_range}_d{disc}_p{p}.csv'
            
    elif weighted_inst == 'unweighted': 
        path = f'results/raw_results/n{n}_{p_e}_{sim_method}_degenCheck{degen_check}_no_renorm_pen{penalty}_{method}_b{log_delta_beta_range}_g{log_delta_gamma_range}_d{disc}_p{p}.csv'
    
        
    # Save the DataFrame to a CSV file
    df_angle_set.to_csv(path, index=False, float_format='%.15g')
    
    
    # Now append to overall CSV
    
    # Sort df_angle_set by Ps_avg (descending)
    df_angle_set_sorted = df_angle_set.sort_values(by="Ps_avg", ascending=False)
    
    # Extract the best row (highest Ps_avg)
    best_row = df_angle_set_sorted.iloc[0].copy()
    best_row["instance"] = instance  # Add identifier
    
    # Reorder so 'instance' is first column
    best_row = best_row[["instance"] + [col for col in df_angle_set_sorted.columns]]
    
    # Convert to DataFrame for appending
    best_df = pd.DataFrame([best_row])
    
    
    if os.path.exists(output_csv):
        best_df.to_csv(output_csv, mode='a', header=False, index=False)  # append
    else:
        best_df.to_csv(output_csv, index=False)  # create new

end_time = timeit.default_timer()
Runtime = end_time - start_time
print('Runtime', Runtime)

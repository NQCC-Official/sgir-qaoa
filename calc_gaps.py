# -*- coding: utf-8 -*-
"""

MIS problem - p with n experiment

"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from itertools import product
from qiskit_aer import AerSimulator
import itertools
import pickle
import pandas as pd
import os
import timeit
import re
from scipy.sparse.linalg import eigsh

Ps_thresh_str = '1_over_n'

repeats = 1   

shots = 10000 

optimization_level = 3 # compilation effort

start_p = 3

penalty = 2000


log_delta_beta_range = [-1.5,0.5]
log_delta_gamma_range = [-1,1]

# grid eg 11x11
disc = 11

method = 'LR-QAOA' # GR-QAOA random LR-QAOA SGIR-QAOA-0k
eig_meth = 'matrix_free_extrap_s0_gap4all_s1_gap0' #_extrap_s0_gap4all'

n_start = 12
max_n = 20

power = 3
tanh_rate = 10

ms_name = 'Rand'


sim_method = "statevector" #'density_matrix',"statevector","matrix_product_state",extended_stabilizer

if sim_method == "statevector":
    backend = AerSimulator()
elif sim_method =='matrix_product_state':
    backend = AerSimulator(method=sim_method,
    matrix_product_state_max_bond_dimension=100, # try 50
    matrix_product_state_truncation_threshold=1e-8 # try -6
    # mps_log_data=True,
)
else:
    backend = AerSimulator(method=sim_method)

use_openqaoa = 'N'

weighted_inst = 'weighted' # unweighted

# folder containing instances
p_e = 'd3'
folder = "instances_d3"


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
        qc.rx(-2*betas[ii], range(n_qubits)) # mixer
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




# # create angle lists
log_delta_beta_range_ls = np.linspace(log_delta_beta_range[0],log_delta_beta_range[1],disc)
log_delta_gamma_range_ls = np.linspace(log_delta_gamma_range[0],log_delta_gamma_range[1],disc)



start_time = timeit.default_timer()  # Start timing



output_csv = f'results/p_with_n_Ps{Ps_thresh_str}_{method}_{start_p}_b{log_delta_beta_range}_g{log_delta_gamma_range}_d{disc}_v{ms_name}.csv'
    


for nn in range(n_start,max_n+2,2):
    
    # if nn==max_n:
    #     continue
    # Fix problem instances per n
    instances = sorted(
    [
        f for f in os.listdir(folder)
        if (match := re.search(r"_n(\d+)_", f))
        and int(match.group(1)) == nn
    ],
    key=lambda name: int(re.search(r"_(\d+)\.gpickle$", name).group(1))
)

    const_ps_reached = 'no'
    
    if nn == n_start:
        p = start_p
    
    # Now do the changing p stuff
    while const_ps_reached == 'no':
        
        Ps_instance = []   # length 10
        best_rows   = []   # optional: for logging
    
        for ii in range(10):
            
            
            instance = instances[ii]
            
            
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
            
                
                
            
            t_vals = np.linspace(0, p, p)
            s_vals = np.linspace(0, 1, p)
            
            
            
            if eig_meth == 'matrix_free_extrap_s0_gap4all_s1_gap0':
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
                    # ---- Load extrapolated minimum gap ----
                    # df_extrap = pd.read_csv("results/extrapolated_min_gaps_degChckN.csv")  # columns: n, gap
                    # extrap_min_gap = df_extrap.loc[df_extrap["n"] == n, "gap"].values[0]
            
                    # # ---- Replace last value (s=1) with extrapolated min gap ----
                    # gaps_qaoa = avg_gaps.copy()
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
                    
                    # plt.plot(t_vals,gammas)
                    # plt.plot(t_vals,betas)
                    # plt.show()
             

                    
                
                elif method == 'SGIR-QAOA-0k' or method =='SGIR-QAOA-0k-eigh'or method == 'SGIR-QAOA-0k-sym':
                    
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

                # Ps_pre_ls.append(Ps_pre)
                # angle_ls.append(x0)
                
                Ps_avg = np.mean(Ps_ls)
                Ps_std = np.std(Ps_ls)

                Ps_ls.append(Ps)
                AR_ls.append(AR)
                log_delta_beta_ls.append(log_delta_beta)
                log_delta_gamma_ls.append(log_delta_gamma)
                x0_ls.append(x0)
                
                print(f'Completed {track} out of {disc**2}')
            

            if method == 'SGIR-QAOA-0k' or method == 'SGIR-QAOA-0k-eigh' or method == 'SGIR-QAOA-0k-sym':
                df_angle_set = pd.DataFrame({
                    'p':p,
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
                    'p':p,
                    'log_delta_beta': log_delta_beta_ls,
                    'log_delta_gamma': log_delta_gamma_ls,
                    'x0':x0_ls,
                    'Ps_avg': Ps_ls,
                    'AR': AR_ls
                })
            


            path = f'results/raw_results/p_with_n_{p_e}_{method}_{eig_meth}_b{log_delta_beta_range}_g{log_delta_gamma_range}_d{disc}_p{p}.csv'
            

            
            # Save the DataFrame to a CSV file
            df_angle_set.to_csv(path, index=False, float_format='%.15g')
            
            
            # Now append to overall CSV
            
            # Sort df_angle_set by Ps_avg (descending)
            df_angle_set_sorted = df_angle_set.sort_values(by="Ps_avg", ascending=False)
            
            # Extract the best row (highest Ps_avg)
            best_row = df_angle_set_sorted.iloc[0].copy()
            best_row["n"] = nn
            best_row["inst"] = instance
            
    
            
            # Convert to DataFrame for appending
            best_df = pd.DataFrame([best_row])
            
            
            Ps_max = best_row['Ps_avg']
            
            
            Ps_instance.append(best_row['Ps_avg'])
            best_rows.append(best_row)
            
            print(f'Repeat {ii} of 10')
        
        
        best_rows_df = pd.DataFrame(best_rows)
        
        Ps_ls = best_rows_df["Ps_avg"].tolist()
        Ps_mean = best_rows_df["Ps_avg"].mean()
        Ps_std  = best_rows_df["Ps_avg"].std()
        
        AR_ls = best_rows_df["AR"].tolist()
        AR_mean = best_rows_df["AR"].mean()
        AR_std  = best_rows_df["AR"].std()
        
        log_delta_beta_list  = best_rows_df["log_delta_beta"].tolist()
        log_delta_gamma_list = best_rows_df["log_delta_gamma"].tolist()
        x0_list              = best_rows_df["x0"].tolist()
        
        if method in ["SGIR-QAOA-0k", "SGIR-QAOA-0k-eigh"]:
            min_gap_mean = best_rows_df["Min gap"].mean()
            min_gap_std  = best_rows_df["Min gap"].std()
        
            s_min_mean = best_rows_df["S min"].mean()
            s_min_std  = best_rows_df["S min"].std()


        
        summary_row = {
            "n": nn,
            "p": p,
        
            "log_delta_beta_list": log_delta_beta_list,
            "log_delta_gamma_list": log_delta_gamma_list,
            "x0_list": x0_list,
            
            "Ps_ls": Ps_ls,
            "Ps_mean": Ps_mean,
            "Ps_std": Ps_std,
            
            "AR_ls": AR_ls,
            "AR_mean": AR_mean,
            "AR_std": AR_std,
            "inst":instances
        }
        
        if method in ["SGIR-QAOA-0k", "SGIR-QAOA-0k-eigh"]:
            summary_row.update({
                "min_gap_mean": min_gap_mean,
                "min_gap_std": min_gap_std,
                "s_min_mean": s_min_mean,
                "s_min_std": s_min_std,
            })
        
        
        
        summary_df = pd.DataFrame([summary_row])
        
        print(f"n={nn}, p={p}, Ps_mean={Ps_mean:.4e}, Ps_std={Ps_std:.4e}")
            

        if Ps_thresh_str == '1_over_n':
            Ps_thresh = 1/n
        
        if Ps_mean >= Ps_thresh:
            
            const_ps_reached = 'yes'
            
            print(f'{method}, n = {nn}, target Ps reached! Ps =', Ps_mean, 'p =', p)
            
            if os.path.exists(output_csv):
                summary_df.to_csv(output_csv, mode="a", header=False, index=False)
            else:
                summary_df.to_csv(output_csv, index=False)
            
            
        else:
            const_ps_reached = 'no'
            
            print(f'{method}, n = {nn}, target Ps not reached.. Ps =', Ps_mean, 'p =', p, '. Target Ps = ', Ps_thresh)
            
            p = p + 1
            
            
        if os.path.exists(output_csv):
            summary_df.to_csv(f"results/raw_results_tracking_p/p_with_n_{method}_{eig_meth}_{start_p}_b{log_delta_beta_range}_g{log_delta_gamma_range}_d{disc}_v{ms_name}_{Ps_thresh_str}.csv",  index=False)
        else:
            summary_df.to_csv(f"results/raw_results_tracking_p/p_with_n_{method}_{eig_meth}_{start_p}_b{log_delta_beta_range}_g{log_delta_gamma_range}_d{disc}_v{ms_name}_{Ps_thresh_str}.csv", index=False)
    
        



end_time = timeit.default_timer()
Runtime = end_time - start_time
print('Runtime', Runtime)
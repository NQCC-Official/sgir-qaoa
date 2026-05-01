# -*- coding: utf-8 -*-
"""

Plotting the results from solving Grover's problem

"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
from matplotlib.ticker import MaxNLocator 

p = 10

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams.update({'font.size': 25})

# =========================
# CSV files
# =========================
csv_LR = f"results/big_LR-QAOA_{p}_b[-1.5, 0.5]_g[-1, 1]_d11_vRand.csv"
csv_RC = f"results/big_AR-QAOA-anal_10_b[-1.5, 0.5]_g[-1, 1]_d11_vRand.csv" # AR was renamed RC
csv_SG = f"results/big_SGIR-QAOA-0k-sym_10_b[-1.5, 0.5]_g[-1, 1]_d11_vRand.csv" 
csv_rand = f"results/big_random_{p}_b[-1.5, 0.5]_g[-1, 1]_d11_vRand.csv"


# =========================
# Load CSVs
# =========================
df      = pd.read_csv(csv_LR)
df_GR   = pd.read_csv(csv_RC)
df_SG   = pd.read_csv(csv_SG, on_bad_lines='skip') if os.path.exists(csv_SG) else None
df_rand = pd.read_csv(csv_rand) if os.path.exists(csv_rand) else None


# =========================
# Methods
# =========================
methods = [
    ("LR-QAOA", df, cm.viridis(0.25), 0.0),
    ("RC-QAOA", df_GR, cm.viridis(0.75), 0.0),
    ("SGIR-QAOA exact", df_SG, cm.plasma(0.5), 0.0),
    ("Random params", df_rand, "black", 0.0)
]


# =========================
# Plot
# =========================
plt.figure(figsize=(10, 8))

# ----- baseline 2^{-n} -----
n_vals = np.unique(
    np.concatenate([
        df_["n"].values for _, df_, _, _ in methods
        if df_ is not None and "n" in df_.columns
    ])
)


n_ref = np.linspace(
    df["n"].min(),
    df["n"].max(),
    300
)
plt.plot(
    n_ref,
    2 ** (-n_ref),
    color="gray",
    linewidth=3,
    label=r"Random guess: $2^{-n}$"
)

# ----- loop over methods -----
for label, df_, color, offset in methods:
    if df_ is None:
        continue
    if "n" not in df_.columns or "Ps_avg" not in df_.columns:
        continue

    # Scatter
    s = 120 if label == "LR-QAOA" else 80
    plt.scatter(
        df_["n"] + offset,
        df_["Ps_avg"],
        color=color,
        alpha=0.25,
        s=s,
        label=f"{label}"
    )


# =========================
# Formatting
# =========================
plt.xlabel(r"$n$ (problem size)")
plt.ylabel(r"$P_s$")
plt.yscale("log")
plt.title(rf"$p = {p}$")
plt.legend(fontsize=24)
plt.grid(True, linestyle='--', alpha=0.7)

# Force the x-axis to display strictly integer ticks
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

plt.tight_layout()
plt.savefig(f"plots/big_plot_grover_p{p}_base2_fits.pdf")
plt.show()




# =========================
# Plot percent improvement with min gap size
# =========================

df_compare = pd.merge(
    df[["n", "Ps_avg"]],
    df_SG[["n", "Ps_avg", "Min gap"]],
    on="n",
    suffixes=("_LR", "_SGIR")
)


df_compare["percent_improvement"] = (
    (df_compare["Ps_avg_SGIR"] - df_compare["Ps_avg_LR"])
    / df_compare["Ps_avg_LR"]
) * 100

# Optional: remove bad points
df_compare = df_compare.replace([np.inf, -np.inf], np.nan).dropna()

# =========================
# Plot
# =========================

plt.figure(figsize=(8, 6))

plt.scatter(
    df_compare["Min gap"],
    df_compare["percent_improvement"],
    alpha=0.8,
    color='black'
)

plt.xlabel(r"$g_{\mathrm{min}}$")
plt.ylabel(r"Improvement over LR-QAOA (\%)")

plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()

plt.savefig(f"plots/gap_vs_percent_improvement_p{p}.pdf")

plt.show()


# Compute Pearson correlation
pearson_corr = df_compare["Min gap"].corr(df_compare["percent_improvement"])

print(f"Pearson correlation coefficient: {pearson_corr:.4f}")


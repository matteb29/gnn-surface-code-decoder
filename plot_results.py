import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# 1. Dati grezzi
data = {
    "p": [0.002, 0.003, 0.005, 0.007, 0.010],
    "MWPM": [0.00301, 0.00651, 0.01705, 0.03204, 0.05928],
    "GNN": [0.00264, 0.00562, 0.01531, 0.02930, 0.05632]
}
df = pd.DataFrame(data)

# 2. Conversione in unità di 10^-3
df["p_scaled"] = df["p"] * 1000
df["MWPM_scaled"] = df["MWPM"] * 1000
df["GNN_scaled"] = df["GNN"] * 1000

# 3. Impostazioni Stile LHCb / HEP
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 18,
    'axes.labelsize': 25,
    'axes.titlesize': 25,
    'xtick.labelsize': 25,
    'ytick.labelsize': 25,
    'legend.fontsize': 35,
    'axes.linewidth': 2.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})

# 4. Creazione Plot
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(df["p_scaled"], df["MWPM_scaled"], 
        marker="o", linestyle="-", markersize=8, label="MWPM")

ax.plot(df["p_scaled"], df["GNN_scaled"], 
        marker="s", linestyle="-", markersize=8, label="GNN")

# Assi e Label con indicazione dell'unità
ax.set_xlabel(r"Physical error rate $p$ [$10^{-3}$] (a.u.)")
ax.set_ylabel(r"Logical error rate [$10^{-3}$] (a.u.)")

# Ticks
ax.tick_params(which='both', direction='in', top=True, right=True, width=1.2)
ax.tick_params(which='major', length=8)
ax.tick_params(which='minor', length=4)

# Legenda senza bordo
ax.legend(frameon=False)

fig.tight_layout()

# Salvataggio
os.makedirs("results", exist_ok=True)
fig.savefig("results/improvement_plot.png", dpi=300)
print("Plot salvato in results/improvement_plot.png")

# Opzionale: mostra a schermo
plt.show()
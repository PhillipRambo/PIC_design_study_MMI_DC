import sys, os
sys.path.append(os.path.abspath("../.."))
from scripts import specifications as spec
import emodeconnection as emc
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd


# ======================================================
#  Cross-sectional mode solver sweep for MMI region
# ======================================================

widths_um = np.array([1.5, 1.6, 1.7, 1.8, 2.0, 2.1])  # MMI widths in µm

# Arrays to store results
L3dB_from_deltaN = []   # from Δn = n0 - n1
L3dB_from_fundamental = []  # from analytical formula using n0 only
n_eff0_list = []
n_eff1_list = []

lambda_m = spec.λ * 1e-9  # [m]

for w_um in widths_um:
    print(f"\n=== Solving for MMI width = {w_um:.1f} µm ===")

    # --- parameters ---
    h_core = spec.H
    w_core = w_um * 1e3         # [nm]
    wav_nm = 1600
    t_box = spec.BOX_H
    nmodes = 5
    dx, dy = 20, 10
    trench = 2000
    width = w_core + 2 * trench
    height = h_core + 2 * t_box

    em = emc.EMode()

    em.settings(
        wavelength=wav_nm,
        x_resolution=dx,
        y_resolution=dy,
        window_width=width,
        window_height=height,
        num_modes=nmodes,
        boundary_condition="00"
    )

    em.shape(name="BOX", material="SiO2", height=t_box)
    em.shape(name="MMI_core", material="Si", height=h_core, width=w_core, fill_refractive_index="Air")

    em.FDM()
    neffs = em.get("effective_index")

    for i, n in enumerate(neffs):
        print(f"Mode {i}: n_eff = {n:.5f}")

    if len(neffs) >= 2:
        n0, n1 = neffs[0], neffs[1]
        delta_n = n0 - n1
        L_pi = lambda_m / (2 * delta_n)
        L_3dB = 1.5 * L_pi
        L3dB_from_deltaN.append(L_3dB * 1e6)  # in µm
        n_eff0_list.append(n0)
        n_eff1_list.append(n1)

        # second method using only n_eff0
        W_m = w_um * 1e-6
        L_pi_fund = (4 * n0 * W_m**2) / (3 * lambda_m)
        L_3dB_fund = 1.5 * L_pi_fund
        L3dB_from_fundamental.append(L_3dB_fund * 1e6)

        print(f"Δn={delta_n:.4f} → L_pi={L_pi*1e6:.2f} µm, L_3dB={L_3dB*1e6:.2f} µm")
        print(f"From n_eff0 only: L_pi={L_pi_fund*1e6:.2f} µm, L_3dB={L_3dB_fund*1e6:.2f} µm")

    else:
        L3dB_from_deltaN.append(np.nan)
        L3dB_from_fundamental.append(np.nan)
        n_eff0_list.append(np.nan)
        n_eff1_list.append(np.nan)
        print("Only one guided mode found — MMI too narrow for self-imaging.")

    em.close()

# ======================================================
# Plot 1: L3dB from modal beat (Δn)
# ======================================================

data = {
    "Width_µm": widths_um,
    "n_eff_0": n_eff0_list,
    "n_eff_1": n_eff1_list,
    "Δn": np.array(n_eff0_list) - np.array(n_eff1_list),
    "L3dB_from_Δn_µm": L3dB_from_deltaN,
    "L3dB_from_n0only_µm": L3dB_from_fundamental,
}

df = pd.DataFrame(data)

# Create directory if it doesn't exist
save_dir = r"C:\Users\phill\OneDrive\Skole\Kandidat\1.Semester\Photonic Integrated Circuit Design\Projects\My Project\optical_couplers\simulation_data\MMI_COUPLER_data"
os.makedirs(save_dir, exist_ok=True)

# Save CSV file
save_path = os.path.join(save_dir, "MMI_3dB_results_1600nm.csv")
df.to_csv(save_path, index=False)

print(f"\n✅ Data saved to: {save_path}")
print(df)
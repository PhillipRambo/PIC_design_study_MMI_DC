import sys, os
sys.path.append(os.path.abspath("../.."))

from scripts import specifications as spec
import emodeconnection as emc
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# Monte Carlo configuration
# ==============================

# Number of Monte Carlo samples
N_SAMPLES = 400 # <-- change this to control number of runs

# Nominal values from your spec file
H_CORE_0 = spec.H      # [nm] waveguide core height
H_G_0    = spec.h      # [nm] height of Si in gap
GAP_0    = spec.gap_width      # [nm] nominal gap
W_CORE_0 = spec.w      # [nm] waveguide core width
WAV_NM   = spec.λ      # [nm] wavelength
T_BOX    = spec.BOX_H  # [nm] BOX thickness

# Coupler length (you MUST set this to your DC length)
# Use micrometers here (consistent with λ in µm)
COUPLER_LENGTH_UM = spec.L_3dB  # example: 500 µm, change to your design

# Process variation (1σ) for each parameter [nm]
SIG_H_CORE = 10  # core height variation
SIG_H_G    = 0 # partial etch height variation
SIG_GAP    = 0   # gap variation
SIG_W_CORE = 10  # core width variation

# FDM resolution [nm]
DX = 10
DY = 10
TRENCH = 2000  # [nm] trench width on each side

# ==============================
# Helper functions
# ==============================

def compute_k_and_KPD(n_s, n_a, wav_nm, L_um):
    """
    Compute splitting ratio k and a normalized K_PD ~ sqrt(k(1-k))
    from symmetric/antisymmetric effective indices.
    
    Uses standard DC coupler theory:
        Δn = n_s - n_a
        κ ≈ π/λ * Δn
        P_cross = sin^2(κ L)
        k = P_cross
        
    wav_nm: wavelength in nm
    L_um: coupler length in µm
    """
    wav_um = wav_nm * 1e-3  # [µm]
    delta_n = n_s - n_a

    # Coupling coefficient κ [1/µm]
    kappa = np.pi * delta_n / wav_um

    # Power in cross port after length L
    k = np.sin(kappa * L_um)**2

    # Normalized K_PD (constant prefactors omitted)
    K_PD_norm = np.sqrt(k * (1.0 - k))

    return k, K_PD_norm


# ==============================
# Storage arrays
# ==============================

KPD_list = []
k_list = []

h_core_samples = []
h_g_samples = []
gap_samples = []
w_core_samples = []

# ==============================
# Monte Carlo loop
# ==============================

for i in range(N_SAMPLES):
    print(f"Running sample {i+1}/{N_SAMPLES}")

    # ---- Draw perturbed geometry (Gaussian variations) ----
    h_core = H_CORE_0 + np.random.normal(0.0, SIG_H_CORE)
    h_g    = H_G_0    + np.random.normal(0.0, SIG_H_G)
    gap    = GAP_0    + np.random.normal(0.0, SIG_GAP)
    w_core = W_CORE_0 + np.random.normal(0.0, SIG_W_CORE)

    # Optional: avoid negative or unphysical values
    h_core = max(h_core, 50.0)
    h_g    = max(min(h_g, h_core - 10.0), 0.0)  # keep etch depth positive
    gap    = max(gap, 20.0)
    w_core = max(w_core, 50.0)

    # ---- Define simulation window ----
    width  = gap + 2 * w_core + 2 * TRENCH  # [nm]
    height = h_core + 2 * T_BOX             # [nm]
    nmodes = 2

    # ---- Connect and initialize EMode ----
    em = emc.EMode()

    em.settings(
        wavelength      = WAV_NM,
        x_resolution    = DX,
        y_resolution    = DY,
        window_width    = width,
        window_height   = height,
        num_modes       = nmodes,
        boundary_condition = "00"
    )

    # BOX
    em.shape(
        name     = "BOX",
        material = "SiO2",
        height   = T_BOX
    )

    # Two partially etched Si waveguides
    em.shape(
        name       = "waveguides",
        material   = "Si",
        height     = h_core,
        mask       = [w_core, w_core],
        mask_offset = [-w_core/2 - gap/2, w_core/2 + gap/2],
        etch_depth = h_core - h_g,
        fill_refractive_index = "Air"
    )

    # ---- Solve ----
    em.FDM()
    em.report()

    # ---- Extract symmetric / antisymmetric neff ----
    neff = em.get("effective_index")

    n_s = neff[0]  # fundamental (symmetric)
    n_a = neff[1]  # first-order (antisymmetric)

    # ---- Compute k and K_PD for this sample ----
    k, K_PD_norm = compute_k_and_KPD(
        n_s = n_s,
        n_a = n_a,
        wav_nm = WAV_NM,
        L_um = COUPLER_LENGTH_UM
    )

    KPD_list.append(K_PD_norm)
    k_list.append(k)

    h_core_samples.append(h_core)
    h_g_samples.append(h_g)
    gap_samples.append(gap)
    w_core_samples.append(w_core)

    em.close()


# ==============================
# Convert to arrays and save
# ==============================

KPD_arr = np.array(KPD_list)
k_arr   = np.array(k_list)

h_core_arr = np.array(h_core_samples)
h_g_arr    = np.array(h_g_samples)
gap_arr    = np.array(gap_samples)
w_core_arr = np.array(w_core_samples)

save_dir = "optical_couplers/simulation_data"

os.makedirs(save_dir, exist_ok=True)

outfile = os.path.join(save_dir, "coupler_montecarlo_KPD.npz")
np.savez(
    outfile,
    KPD=KPD_arr,
    k=k_arr,
    h_core=h_core_arr,
    h_g=h_g_arr,
    gap=gap_arr,
    w_core=w_core_arr
)

print(f"Monte Carlo data saved to {outfile}")

# ==============================
# Plot histogram of K_PD
# ==============================

plt.figure()
plt.hist(k_arr, bins=40, edgecolor="black")
plt.xlabel("Cross-port power k")
plt.ylabel("Count")
plt.title(f"Directional Coupler Monte Carlo: k distribution (N={N_SAMPLES})")
plt.tight_layout()
plt.show()
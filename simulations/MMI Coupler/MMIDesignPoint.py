import sys, os
sys.path.append(os.path.abspath("../.."))

from scripts import specifications as spec
import emodeconnection as emc
import numpy as np

# ==========================
# Geometry & design parameters
# ==========================


# Nominal values
H_CORE = spec.h_MMI * 1e3         # [nm] waveguide core height
W_MMI  = spec.w_MMi * 1e3          # [nm] MMI core width (set your nominal MMI width here)
WAV_NM   = spec.λ          # [nm] wavelength
T_BOX    = spec.BOX_H      # [nm] BOX thickness


# --- MMI design length ---
# Put your designed MMI length in the spec file, e.g. spec.L_MMI
L_MMI_UM = spec.l_MMi   # [µm] MMI physical length along propagation

# Simulation window / resolution
DX = 20          # [nm] x-resolution
DY = 10          # [nm] y-resolution
TRENCH = 2000    # [nm] lateral padding on each side
WIDTH  = W_MMI + 2 * TRENCH        # [nm] total window width
HEIGHT = H_CORE + 2 * T_BOX        # [nm] total window height
NMODES = 5       # enough to get at least the first two slab modes

# ==========================
# Run single EMode simulation
# ==========================

em = emc.EMode()

em.settings(
    wavelength        = WAV_NM,
    x_resolution      = DX,
    y_resolution      = DY,
    window_width      = WIDTH,
    window_height     = HEIGHT,
    num_modes         = NMODES,
    boundary_condition= "00",
)

# BOX (substrate + buffer)
em.shape(
    name     = "BOX",
    material = "SiO2",
    height   = T_BOX
)

# Single wide Si MMI core
em.shape(
    name                = "MMI_core",
    material            = "Si",
    height              = H_CORE,
    width               = W_MMI,
    fill_refractive_index = "Air"
)

# Solve eigenmodes
em.FDM()
em.report()

neffs = em.get("effective_index")

if len(neffs) < 2:
    print("❌ Only one guided mode found – MMI is too narrow for self-imaging.")
    em.close()
    sys.exit(1)

n0 = neffs[0]  # fundamental mode
n1 = neffs[1]  # first higher-order mode
em.plot()
em.close()

# ==========================
# Self-imaging / 3-dB length
# ==========================

lambda_m = WAV_NM * 1e-9      # [m]
L_design_m = L_MMI_UM * 1e-6  # [m]

delta_n = np.abs(n0 - n1)
# Beat length between the two lowest modes (standard MMI theory)
L_pi_m  = lambda_m / (2.0 * delta_n)
# For a 2x2 3-dB MMI using general interference:
L_3dB_m  = 1.5 * L_pi_m
L_3dB_um = L_3dB_m * 1e6

# Relative error of your design vs ideal 3-dB length
rel_err = (L_MMI_UM - L_3dB_um) / L_3dB_um * 100.0

# ==========================
# Print results
# ==========================

print("=== MMI design check (self-imaging theory) ===")
print(f"λ            = {WAV_NM:.2f} nm")
print(f"W_MMI        = {W_MMI:.1f} nm")
print(f"H_CORE       = {H_CORE:.1f} nm")
print()
print(f"n_eff0 (TE0) = {n0:.6f}")
print(f"n_eff1 (TE1) = {n1:.6f}")
print(f"Δn = n0 - n1 = {delta_n:.6e}")
print()
print(f"Beat length L_pi       = {L_pi_m*1e6:.3f} µm")
print(f"Ideal 3-dB length L_3dB = {L_3dB_um:.3f} µm")
print()
print(f"Your design length L    = {L_MMI_UM:.3f} µm")
print(f"Relative error vs L_3dB = {rel_err:.2f} %")

# NOTE:
# This script *does not* compute actual bar/cross port power,
# because that requires a full 2D/3D simulation of the whole MMI
# including access waveguides. Here we only check how close the
# body length is to the theoretical 3-dB length from slab-mode theory.

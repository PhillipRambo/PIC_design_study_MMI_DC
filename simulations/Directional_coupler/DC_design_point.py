import sys, os
sys.path.append(os.path.abspath("../.."))
from scripts import specifications as spec
import emodeconnection as emc
import numpy as np

# ==========================
# Geometry & design parameters
# ==========================

h_core = spec.H      # [nm] waveguide core height
h_g    = spec.h      # [nm] height of Si in gap
gap    = spec.gap_width      # [nm] gap between waveguides
w_core = spec.w      # [nm] waveguide core width
wav_nm = spec.λ      # [nm] wavelength
t_box  = spec.BOX_H  # [nm] BOX thickness

# >>> Set this in your specifications file <<<
# Assume design coupler length in µm:
L_coup_um = spec.l_MMi
# Simulation window / resolution
dx = dy = 10             # [nm] grid size
trench = 2000            # [nm] trench on each side of waveguides
width  = gap + 2*w_core + 2*trench   # [nm]
height = h_core + 2*t_box            # [nm]
nmodes = 2               # we just need the two lowest supermodes

# ==========================
# Run single EMode simulation
# ==========================

em = emc.EMode()

em.settings(
    wavelength      = wav_nm,
    x_resolution    = dx,
    y_resolution    = dy,
    window_width    = width,
    window_height   = height,
    num_modes       = nmodes,
    boundary_condition = "00"
)

# BOX + two partially etched Si waveguides
em.shape(
    name      = "BOX",
    material  = "SiO2",
    height    = t_box
)

em.shape(
    name      = "waveguides",
    material  = "Si",
    height    = h_core,
    mask      = [w_core, w_core],
    mask_offset = [-w_core/2 - gap/2,  w_core/2 + gap/2],
    etch_depth = h_core - h_g,
    fill_refractive_index = "Air"
)

# Solve
em.FDM()
em.report()

# Effective indices (symmetric & antisymmetric supermodes)
neffs = em.get("effective_index")
n_s = neffs[0]   # symmetric
n_a = neffs[1]   # antisymmetric

em.close()

# ==========================
# Coupled-mode calculation
# ==========================

# Δβ from the supermode indices
lambda_m = wav_nm * 1e-9         # [m]
L_m      = L_coup_um * 1e-6      # [m]
delta_n   = np.abs(n_a - n_s)
delta_beta = 2 * np.pi * delta_n / lambda_m   # [rad/m]

P_cross = np.sin(delta_beta * L_m / 2)**2
P_bar   = 1.0 - P_cross

L_3dB_m  = np.pi / (2 * delta_beta)
L_3dB_um = L_3dB_m * 1e6

# ==========================
# Print results
# ==========================

print("=== DC coupler design check ===")
print(f"λ         = {wav_nm:.2f} nm")
print(f"ns        = {n_s:.6f}")
print(f"na        = {n_a:.6f}")
print(f"Δn        = {delta_n:.6e}")
print()
print(f"Design length L = {L_coup_um:.3f} µm")
print(f"Cross power     = {P_cross*100:.2f} %")
print(f"Bar power       = {P_bar*100:.2f} %")
print()
print(f"Ideal 3-dB length L_3dB ≈ {L_3dB_um:.3f} µm")
print("Relative error  vs ideal 3-dB length:",
      f"{(L_coup_um - L_3dB_um)/L_3dB_um*100:.2f} %")
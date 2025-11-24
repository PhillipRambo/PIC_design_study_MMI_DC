import sys, os
sys.path.append(os.path.abspath("../.."))
from scripts import specifications as spec
import emodeconnection as emc
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np

## Arrays for storing symmetric/antisymmetric refractive indices
nss = []
nas = []
g_list = []

## Sweeping over gap sizes
for wavelength in [1500,1510,1520,1530,1540,1550,1560,1570,1580,1590,1600]:
    h_core = spec.H # [nm] waveguide core height
    h_g = spec.h # [nm] height of Si in gap
    gap = spec.gap_width
    w_core = spec.w # [nm] waveguide core width
    wav_nm = wavelength  # [nm] central wavelength of laser
    t_box = spec.BOX_H

    ## Adjusting resolution based on gap
    if gap >= 1000:
        dxx = 50
    else: 
        dxx = 10
    dx, dy = dxx, 10 # [nm] resolution
    trench = 2000 # [nm] waveguide side trench width
    width = gap + 2*w_core + trench*2 # [nm] window width
    height = h_core + 2*t_box # [nm] window height
    nmodes = 2 # [nm] number of modes to simulate

    ## Connect and initialize EMode
    em = emc.EMode()

    ## Settings
    em.settings(wavelength = wav_nm, x_resolution = dx, y_resolution = dy,
        window_width = width, window_height = height, num_modes = nmodes, boundary_condition="00")

    ## Draw shapes using the "mask" keyword for defining two waveguides in a partial etch
    em.shape(name = "BOX", material = "SiO2", height = t_box)
    em.shape(name = "waveguides", material = "Si", height = h_core, mask=[w_core, w_core], mask_offset=[-w_core/2 - gap/2, w_core/2 + gap/2], etch_depth=h_core-h_g, fill_refractive_index = "Air")

    ## Launch FDM solver
    em.FDM()

    ## Report & plot
    em.report()

    ## Extracting effective indices of fundamental and first order modes
    ns = em.get("effective_index")
    nss.append(ns[0])
    nas.append(ns[1])
    g_list.append(wavelength)
    em.close() ## Close the EMode connection


save_dir = "../../simulation_data"
os.makedirs(save_dir, exist_ok=True)

# Save data
np.savez(os.path.join(save_dir, "DC_wavelength_dependence_g=200.npz"),
         nss=nss, nas=nas, g_list=g_list)

print(f"Data saved to {save_dir}/DC_wavelength_dependence_g=200.npz")

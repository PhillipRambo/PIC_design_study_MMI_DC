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



lambda_m = spec.λ * 1e-9  # [m]



# --- parameters ---
h_core = spec.H
w_core = spec.w_MMi * 1e3         # [nm]
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

em.plot()

em.close()

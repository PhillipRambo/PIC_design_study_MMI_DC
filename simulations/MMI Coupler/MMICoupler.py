import sys, os
sys.path.append(os.path.abspath("../.."))
from scripts import specifications as spec
import scripts.material_data_base as mdb
from scripts.mmi_builder import build_mmi_simulation, plot_mmi_layout
import gdstk
import matplotlib.pylab as plt
import numpy as np
import tidy3d as td
import tidy3d.web as web

lambda_scaled = spec.λ * 1e-3
freq0 = td.C_0 / lambda_scaled  

# Materials.
n_si = mdb.Si_n(lambda_scaled)
n_sio2 = mdb.SiO2_n(lambda_scaled)

# Material definitions.
mat_si = td.Medium(permittivity=n_si**2)  # Silicon waveguide material.
mat_sio2 = td.Medium(permittivity=n_sio2**2)  # SiO2 substrate material

sim = build_mmi_simulation(
    w_wg=spec.w_wg_MMI,
    h_si=spec.h_MMI,
    w_mmi=spec.w_MMi,
    l_mmi=spec.l_MMi,
    gap=spec.gap_MMi,
    l_input=spec.l_input_MMi,
    l_output=spec.l_output_MMi,
    s_bend_offset=spec.s_bend_offset_MMi,
    s_bend_length=spec.s_bend_length_MMi,
    mat_si=mat_si,
    mat_sio2=mat_sio2,
    lambda_0=spec.λ * 1e-3,
    lambda_min=1.5, 
    lambda_max=1.61,
    lambda_step=0.01
)

ax = sim.plot(z=0)
fig_2d = ax.figure
plt.show(block=True)

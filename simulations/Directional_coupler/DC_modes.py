import sys, os
sys.path.append(os.path.abspath("../.."))
from scripts import specifications as spec
import emodeconnection as emc
import numpy as np
import matplotlib.pyplot as plt

## Arrays for storing symmetric/antisymmetric refractive indices
nss = []
nas = []

## Sweeping over gap sizes
for g in [spec.gap_width]:
    h_core = spec.H # [nm] waveguide core height
    h_g = spec.h # [nm] height of Si in gap
    gap = spec.gap_width
    w_core = spec.w # [nm] waveguide core width
    wav_nm = spec.λ  # [nm] central wavelength of laser
    t_box = spec.BOX_H

    ## Adjusting resolution based on gap
    if g >= 1000:
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
    em.plot()

    ## Extracting effective indices of fundamental and first order modes
    ns = em.get("effective_index")
    nss.append(ns[0])
    nas.append(ns[1])

    ## Flag for plotting supermode
    plotfig = True
    if plotfig:

        ## Get simulation grid
        xs = em.get('x')
        ys = em.get('y')

        ## Arrays for symmetric & antisymmetric mode
        A = np.zeros((3,len(xs),len(ys)))
        S = np.zeros((3,len(xs),len(ys)))

        ## Extracting symmetric mode
        S[0,:,:] = np.real(em.get("Ex")[0]*1e9)
        S[1,:,:] = np.real(em.get("Ey")[0]*1e9)
        S[2,:,:] = np.imag(em.get("Ez")[0]*1e9)

        ## Extracting antisymmetric mode
        A[0,:,:] = np.real(em.get("Ex")[1]*1e9)
        A[1,:,:] = np.real(em.get("Ey")[1]*1e9)
        A[2,:,:] = np.imag(em.get("Ez")[1]*1e9)

        ## Plotting the symmetric and antisymmetric mode components
        figS = plt.figure()
        fs=16
        labels=[r"$\mathfrak{e}_x$", r"$\mathfrak{e}_y$", r"$\Im(\mathfrak{e}_z)$"]
        for i in range(3):
            ax = figS.add_subplot(1,3,i+1)
            ax.imshow(S[i].T) ## Transposing for correct axis orientation
            ax.set_title(labels[i], fontsize=fs-4)
        plt.tight_layout()
        figS.suptitle("Symmetric", fontsize=fs, y=0.75)
        plt.savefig("directional_coupler_symmetric", dpi=200, bbox_inches='tight')

        figA = plt.figure()
        for j in range(3):
            ax = figA.add_subplot(1,3,j+1)
            ax.imshow(A[j].T)
            ax.set_title(labels[j], fontsize=fs-4)
        plt.tight_layout()
        figA.suptitle("Antisymmetric", fontsize=fs, y=0.75)
        plt.savefig("directional_coupler_antisymmetric", dpi=200, bbox_inches='tight')

        ## Plotting the supermodes
        figSuper = plt.figure()
        pos = S[0] + A[0]
        neg = S[0] - A[0]
        supermodes = [pos, neg]
        titles = ["S+A","S-A"]
        for k in range(2):
            ax = figSuper.add_subplot(1,2,k+1)
            ax.imshow(supermodes[k].T)
            ax.set_title(titles[k] +", " + labels[0], fontsize=fs)
        plt.savefig("directional_coupler_supermodes", dpi=200, bbox_inches='tight')

        plt.tight_layout()
        plt.show()
    em.close() ## Close the EMode connection

## Comparing effective indices to indicate coupling strength
diff = np.array(nss) - np.array(nas)
plt.plot(diff,'.')
plt.show()

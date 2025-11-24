import sys, os
sys.path.append(os.path.abspath("../.."))

from scripts import specifications as spec
import emodeconnection as emc
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# Nominal design parameters
# ==============================

H_CORE_0    = spec.H        # [nm] Si thickness
W_MMI_0_UM  = spec.w_MMi   # [µm] nominal MMI width
W_MMI_0     = W_MMI_0_UM * 1e3   # [nm] nominal width in nm
WAV_NM      = spec.λ        # [nm] wavelength
L_MMI       = spec.l_MMi * 1e3   # [nm] physical MMI length
L_MMI_0_UM  = spec.l_MMi   # [nm] physical MMI length
T_BOX       = spec.BOX_H    # [nm] BOX thickness

DX = 20
DY = 20
TRENCH = 2000      # [nm] lateral padding each side
NMODES = 2         # use first 2 lateral modes


# ==============================
# Helper: cross-section EMODE
# ==============================

import time

def compute_mmi_neffs(W_mmi_nm, nmodes=2, do_plot=False):
    height = H_CORE_0 + 2 * T_BOX
    width  = W_mmi_nm + 2 * TRENCH

    # Try a couple of times in case Windows is slow releasing the file
    em = None
    for attempt in range(3):
        try:
            em = emc.EMode()
            break
        except PermissionError as e:
            print(f"[WARN] EMode client failed to start (port file locked) attempt {attempt+1}/3: {e}")
            time.sleep(0.5)
        except Exception as e:
            print(f"[WARN] EMode client failed to start: {repr(e)}")
            return None

    if em is None:
        print("[WARN] Giving up on this sample due to EMode client errors.")
        return None

    em.settings(
        wavelength         = WAV_NM,
        x_resolution       = DX,
        y_resolution       = DY,
        window_width       = width,
        window_height      = height,
        num_modes          = nmodes,
        boundary_condition = "00"
    )

    em.shape(
        name     = "BOX",
        material = "SiO2",
        height   = T_BOX
    )

    em.shape(
        name          = "MMI_core",
        material      = "Si",
        height        = H_CORE_0,
        width         = W_mmi_nm,
        fill_material = "Air"
    )

    try:
        em.FDM()
    except Exception as e:
        print(f"[WARN] EMode FDM failed for W={W_mmi_nm:.2f} nm: {repr(e)}")
        em.close()
        return None

    neff = np.array(em.get("effective_index"))

    if do_plot:
        em.plot()

    em.close()
    return neff



# ==============================
# Helper: 2-mode MMI splitting model (analytic)
# ==============================

def mmi_two_mode_splitting(neff0, neff1, W_mmi_nm, W_nom_nm, L_mmi_um, wav_nm):
    """
    Two-mode MMI interference model.

    Modes approximated as:
        φ0(x) ~ cos(pi x / W)
        φ1(x) ~ cos(2 pi x / W - pi/2)

    Field at z=L:
        E(x) = A0 φ0(x) e^{-j β0 L} + A1 φ1(x) e^{-j β1 L}

    - A0, A1 are launch coefficients computed from nominal geometry
      assuming a delta-like input at x_in = W_nom/4.
    - Output ports are fixed at x = ±W_nom/4 (mask positions).
    - W_mmi_nm is the *current* slab width (for the cos() arguments).
    """

    # Units
    wav_m  = wav_nm * 1e-9
    L_m    = L_mmi_um * 1e-6

    beta0 = 2 * np.pi / wav_m * neff0
    beta1 = 2 * np.pi / wav_m * neff1
    dphi  = (beta1 - beta0) * L_m   # relative phase between modes

    # Nominal launch and port positions (fixed by mask)
    x_in    = +W_nom_nm / 4.0       # [nm]
    x_port1 = -W_nom_nm / 4.0       # [nm]
    x_port2 = +W_nom_nm / 4.0       # [nm]

    # Launch coefficients from nominal lateral profiles at x_in
    phi0_in = np.cos(np.pi * x_in / W_nom_nm)
    phi1_in = np.cos(2 * np.pi * x_in / W_nom_nm - np.pi / 2)

    A0 = phi0_in
    A1 = phi1_in
    norm = np.sqrt(np.abs(A0)**2 + np.abs(A1)**2)
    if norm > 0:
        A0 /= norm
        A1 /= norm

    # Local helper to evaluate E(x) for current W_mmi_nm
    def E_at(x_nm):
        phi0 = np.cos(np.pi * x_nm / W_mmi_nm)
        phi1 = np.cos(2 * np.pi * x_nm / W_mmi_nm - np.pi / 2)
        return A0 * phi0 + A1 * phi1 * np.exp(-1j * dphi)

    E1 = E_at(x_port1)
    E2 = E_at(x_port2)

    P1 = np.abs(E1)**2
    P2 = np.abs(E2)**2

    eta = P1 / (P1 + P2)  # fraction in left port
    return eta, P1, P2


# ==============================
# MAIN: Test nominal operation
# ==============================

if __name__ == "__main__":
    print("=== Nominal 2×2 MMI test (2-mode model) ===")
    print(f"Nominal MMI width   W0 = {W_MMI_0_UM:.3f} µm")
    print(f"Nominal MMI length  L0 = {L_MMI_0_UM:.3f} µm")
    print(f"Wavelength          λ  = {WAV_NM:.1f} nm")

    # 1) Get neff0, neff1 from cross-section EMODE
    neff = compute_mmi_neffs(W_MMI_0, nmodes=NMODES, do_plot=True)
    if neff is None or len(neff) < 2:
        raise RuntimeError("Failed to get at least 2 modes for nominal MMI geometry.")

    neff0, neff1 = neff[0], neff[1]
    print(f"neff0 = {neff0:.6f}, neff1 = {neff1:.6f}")

    # 2) Use 2-mode analytical MMI model to get splitting
    eta_nom, P1_nom, P2_nom = mmi_two_mode_splitting(
        neff0     = neff0,
        neff1     = neff1,
        W_mmi_nm  = W_MMI_0,
        W_nom_nm  = W_MMI_0,
        L_mmi_um  = L_MMI_0_UM,
        wav_nm    = WAV_NM
    )

    print(f"Nominal splitting η_left = {eta_nom:.4f} (ideal ≈ 0.5)")
    print(f"P1:P2 ≈ {P1_nom:.3e} : {P2_nom:.3e}")

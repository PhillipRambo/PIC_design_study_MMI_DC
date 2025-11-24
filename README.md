# Photonic Coupler & MMI Simulation Toolkit

This repository contains a collection of simulation scripts, models, and analysis tools for optical directional couplers and 2×2 MMI power splitters.  
The project provides a structured workflow for generating mode data, running wavelength-dependent simulations, evaluating Monte Carlo variations, and producing publication-quality plots.

---

## Project Structure

├── notebooks/              # Jupyter notebooks for visualization and analysis  
├── scripts/                # Core Python modules (builders, material DB, specs)  
├── simulation_data/        # Raw simulation outputs (ignored in Git)  
├── simulations/            # Full simulation setups for DC and MMI blocks  
├── plots/                  # Exported figures and results  
├── optical_couplers/       # Design files and coupler utilities  
└── Backup_data/            # Local backup snapshots (ignored)

Raw simulation files (NPZ, HDF5, CSV) are intentionally excluded from version control.  
All code, notebooks, and plots are tracked.

---

## Features

### Directional Couplers
- Coupling-length vs gap simulations  
- Wavelength-dependent behavior  
- Mode solving and supermode analysis  
- Monte Carlo coupling variation  
- Automated plotting utilities  

### MMI Couplers (2×2)
- 3-dB performance analysis  
- Wavelength-dependent characterization  
- Monte Carlo mismatch evaluation  
- Mode profile extraction  
- Structured design-point evaluation  

---

## Requirements

- Python 3.10+  
- NumPy  
- SciPy  
- Matplotlib  
- Jupyter (optional)  
- License to emode  
- License to Tidy3D  

---

## Data Handling

All raw simulation data is ignored by `.gitignore`.  
To include results, export plots or images into `plots/`.

---

## Notebooks

The `notebooks/` folder contains visual tools for:
- Directional coupler wavelength sweeps  
- MMI 2×2 field and transmission analysis  

These notebooks rely on modules in `scripts/`.

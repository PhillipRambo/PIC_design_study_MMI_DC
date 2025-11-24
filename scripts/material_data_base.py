import numpy as np

def SiO2_n(lambda_um):
    """Compute refractive index n(λ) for λ in micrometers (µm)."""
    λ = np.array(lambda_um, dtype=float)
    
    n_eff = (
        1
        + (0.696166 / (1 - (6.84043e-2 / λ)**2))
        + (0.407943 / (1 - (0.116241 / λ)**2))
        + (0.897479 / (1 - (9.89616 / λ)**2))
    ) ** 0.5
    
    return n_eff

def Si_n(lambda_um):
    """Compute refractive index n(λ) for λ in micrometers (µm)."""
    λ = np.array(lambda_um, dtype=float)
    
    n_eff = (
        1
        + (10.6684 / (1 - (0.301516 / λ)**2))
        + ((3.04347e-3) / (1 - (1.13475 / λ)**2))
        + (1.54133 / (1 - (1104 / λ)**2))
    ) ** 0.5
    
    return n_eff
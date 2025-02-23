import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.optimize import differential_evolution
from scipy.interpolate import SmoothBivariateSpline
from tensorflow import keras
import tensorflow as tf
from joblib import load

def calibration_residual(params, model, target_iv_surface, strike_grid, maturity_grid, X_scaler):
    """
    Returns a 1D NumPy array of residuals:
        residual[i] = pred_iv[i] - target_iv[i]
    for each (strike, maturity) pair.
    """
    a, b, c, eta, rho, H = params

    # Mesh for strike, maturity
    K, T = np.meshgrid(strike_grid, maturity_grid)
    K_flat = K.ravel()       # shape (num_points,)
    T_flat = T.ravel()

    # Build param grid to feed the NN model
    param_sets = np.column_stack([
        np.full_like(K_flat, a),
        np.full_like(K_flat, b),
        np.full_like(K_flat, c),
        np.full_like(K_flat, eta),
        np.full_like(K_flat, rho),
        np.full_like(K_flat, H),
        K_flat,
        T_flat
    ])

    # Scale inputs
    param_scaled = X_scaler.transform(param_sets)

    # Predict implied vols
    pred_iv_flat = model.predict(param_scaled, verbose=0).flatten()

    # Flatten target IV surface to match
    target_iv_flat = target_iv_surface.ravel()

    # Residual vector
    residuals = pred_iv_flat - target_iv_flat
    return residuals




import numpy as np
from scipy.optimize import least_squares

# -- Your same code: load data, define model, X_scaler, etc. --

# Remove or ignore bounds in LM. We'll just pick a "reasonable" x0:
x0 = np.array([0.01,   # a
               0.01,   # b
               0.01,   # c
               1.0,    # eta
               -0.5,   # rho
               0.3])   # H

# We can’t do finite bounds with 'lm' -> method='lm'.
# If you do need bounds, see the note below on 'method=trf'.

result_lsq = least_squares(
    fun=calibration_residual,
    x0=x0,
    args=(model, target_iv_surface, strike_grid, maturity_grid, X_scaler),
    method='lm',      # <-- Levenberg-Marquardt
    max_nfev=2000     # maximum function evaluations
)

print("\nOptimization success:", result_lsq.success)
print("Message:", result_lsq.message)
print("Calibrated parameters:", result_lsq.x)
# cost = 1/2 sum(residual^2)
print("Calibration cost (1/2 * sum of residuals^2):", result_lsq.cost)


lower_bounds = [b[0] for b in bounds]  # e.g. [0.002, 0.002, 0.002, 0.5, -0.95, 0.025]
upper_bounds = [b[1] for b in bounds]  # e.g. [0.1, 0.1, 0.1, 4.0, -0.1, 0.5]

result_lsq = least_squares(
    fun=calibration_residual,
    x0=x0,
    args=(model, target_iv_surface, strike_grid, maturity_grid, X_scaler),
    bounds=(lower_bounds, upper_bounds),
    method='trf',        # or 'dogbox'
    max_nfev=2000
)
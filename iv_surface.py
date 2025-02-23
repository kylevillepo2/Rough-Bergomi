

import numpy as np
import torch
from joblib import load
import matplotlib.pyplot as plt
from tensorflow import keras  # Only needed if you want to compare with your Keras version

# Define your model architecture (must match the one you used during training)
import torch.nn as nn
class RoughBergomiNet(nn.Module):
    def __init__(self):
        super(RoughBergomiNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 64),
            nn.ELU(),
            nn.Linear(64, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.model(x)

model = RoughBergomiNet()
model.load_state_dict(torch.load("rbergomi_model.pth", map_location=torch.device("cpu")))
model.eval()  # Set the model to evaluation mode

# Load the scaler (used during training) to normalize inputs
X_scaler = load("X_scaler.joblib")

# --- 2. Define the grid for strikes and maturities ---
S0 = 1  # Underlying price, if needed to set strike range
strike_range = np.linspace(S0 * 0.8, S0 * 1.2, 30)   # 30 strikes
maturity_range = np.linspace(30 / 365.25, 2, 25)       # 25 maturities

# Create a grid
strike_mesh, maturity_mesh = np.meshgrid(strike_range, maturity_range)
strikes_flat = strike_mesh.ravel()
maturities_flat = maturity_mesh.ravel()

# --- 3. Set fixed parameters for the other inputs ---
# These parameters (a, b, c, eta, rho, H) should be chosen as desired.
a_val   = 0.05   # example value for parameter 'a'
b_val   = 0.05   # example value for parameter 'b'
c_val   = 0.05   # example value for parameter 'c'
eta_val = 2.0    # example value for 'eta'
rho_val = -0.7   # example value for 'rho'
H_val   = 0.3    # example value for 'H'
param_array = np.column_stack([
    np.full_like(strikes_flat, a_val),
    np.full_like(strikes_flat, b_val),
    np.full_like(strikes_flat, c_val),
    np.full_like(strikes_flat, eta_val),
    np.full_like(strikes_flat, rho_val),
    np.full_like(strikes_flat, H_val),
    strikes_flat,
    maturities_flat
])
n = 1000       # Number of time steps
m = 100000  

r = 0.0427
eta = 2.0
rho = -0.7
H = 0.3
# --- 4. Scale the inputs ---
param_scaled = X_scaler.transform(param_array)

# Convert to PyTorch tensor
input_tensor = torch.tensor(param_scaled, dtype=torch.float32)

# --- 5. Generate predictions using the model ---
with torch.no_grad():
    iv_pred = model(input_tensor).cpu().numpy().flatten()

# Reshape predictions back to grid shape
iv_surface_nn  = iv_pred.reshape(strike_mesh.shape)


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create a surface plot
surf = ax.plot_surface(strike_mesh, maturity_mesh, iv_surface_nn, 
                       cmap='viridis', edgecolor='none')

ax.set_xlabel("Strike")
ax.set_ylabel("Maturity")
ax.set_zlabel("Implied Volatility")
ax.set_title("3D Implied Volatility Surface")
fig.colorbar(surf, shrink=0.5, aspect=5, label="IV")

def compute_option_price(prices, T, K):
    discount_factor = np.exp(-r * T)
    payoffs = np.maximum(prices - K, 0)

    return np.mean(payoffs) * discount_factor

from RoughBergomi import rbergomi_price, implied_volatility, mc_sim

xi0 = [a_val, b_val, c_val]
# --- Define grid for strikes and maturities ---
# Here strikes range from 80% to 120% of S0 and maturities from 30 days (in years) to 2 years.
strike_range = np.linspace(S0 * 0.8, S0 * 1.2, 30)
maturity_range = np.linspace(30 / 365.25, 2, 25)

# Prepare an array to hold the implied volatilities
iv_surface_rb = np.zeros((len(maturity_range), len(strike_range)))
max_maturity = 2
prices = mc_sim(S0, n, m, r, max_maturity, xi0, eta, rho, H, whole_process=True)

for i, T in enumerate(maturity_range):
    for j, K in enumerate(strike_range):
        prices_cropped = prices[:, min(int(n * (T / 2)), n - 1)]
        price = compute_option_price(prices_cropped, T, K)
        iv = implied_volatility(price.item(), S0, K.item(), r, T.item())
        iv_surface_rb[i, j] = iv


strike_mesh, maturity_mesh = np.meshgrid(strike_range, maturity_range)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot the Neural Network IV surface (using 'viridis' colormap)
surf_nn = ax.plot_surface(strike_mesh, maturity_mesh, iv_surface_nn,
                          cmap='viridis', alpha=0.6, edgecolor='none')

# Plot the Simulation-based IV surface (using 'plasma' colormap)
surf_rb = ax.plot_surface(strike_mesh, maturity_mesh, iv_surface_rb,
                          cmap='plasma', alpha=0.6, edgecolor='none')

ax.set_xlabel("Strike")
ax.set_ylabel("Maturity (Years)")
ax.set_zlabel("Implied Volatility")
ax.set_title("3D Implied Volatility Surfaces: NN vs Simulation (Rough Bergomi)")

# Create custom legend entries via proxy artists
from matplotlib.lines import Line2D
proxy_nn = Line2D([0], [0], linestyle="none", marker='s', markersize=10, markerfacecolor='purple')
proxy_rb = Line2D([0], [0], linestyle="none", marker='s', markersize=10, markerfacecolor='orange')
ax.legend([proxy_nn, proxy_rb], ['NN IV Surface', 'Simulation IV Surface'], loc='upper right')

plt.show()
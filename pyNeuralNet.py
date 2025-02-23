import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from RoughBergomi import mc_sim, implied_volatility, dw1

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

options_data = pd.read_csv('yfinance_dataset.csv')

# Fixed parameters
n = 1000
m = 100000

r = 0.0427
S0 = 1

num_samples = 500

# Convert NumPy arrays to PyTorch tensors
a_samples = torch.tensor(np.random.uniform(0.002, 0.1, num_samples), dtype=torch.float32, device=device)
b_samples = torch.tensor(np.random.uniform(0.002, 0.1, num_samples), dtype=torch.float32, device=device)
c_samples = torch.tensor(np.random.uniform(0.002, 0.1, num_samples), dtype=torch.float32, device=device)
eta_samples = torch.tensor(np.random.uniform(0.5, 4.0, num_samples), dtype=torch.float32, device=device)
rho_samples = torch.tensor(np.random.uniform(-0.95, -0.1, num_samples), dtype=torch.float32, device=device)
H_samples = torch.tensor(np.random.uniform(0.025, 0.5, num_samples), dtype=torch.float32, device=device)

# Convert linspace to PyTorch tensors
strike_range = torch.linspace(S0 * 0.8, S0 * 1.2, 30, dtype=torch.float32, device=device)
maturity_range = torch.linspace(30 / 365.25, 2, 25, dtype=torch.float32, device=device)

def compute_option_price(prices, T, K):
    discount_factor = torch.exp(-r * T)
    payoffs = torch.maximum(prices - K, torch.tensor(0.0, device=device))
    return torch.mean(payoffs) * discount_factor

def compute_option_and_iv(idx):
    # Extract sample parameters
    a, b, c, eta, rho, H = (
        a_samples[idx],
        b_samples[idx],
        c_samples[idx],
        eta_samples[idx],
        rho_samples[idx],
        H_samples[idx]
    )

    # Store results
    data_points = []
    xi0 = torch.tensor([a, b, c], dtype=torch.float32, device=device)

    # Get max maturity
    max_maturity = torch.max(maturity_range)

    # Simulate Monte Carlo Prices (Ensure mc_sim is adapted to use PyTorch tensors)
    prices = mc_sim(S0, n, m, r, max_maturity, xi0, eta, rho, H, whole_process=True)

    # Loop over maturities and strikes
    for T in maturity_range:
        for K in strike_range:
            prices_cropped = prices[:, min(int(n * (T / 2)), n - 1)]
            price = compute_option_price(prices_cropped, T, K)
            iv = implied_volatility(price.item(), S0, K.item(), r, T.item())  # Convert to scalar

            # Store only valid implied volatility values
            if 0.001 < iv < 3.0:
                data_points.append({
                    'a': xi0[0].item(),
                    'b': xi0[1].item(),
                    'c': xi0[2].item(),
                    'eta': eta.item(),
                    'rho': rho.item(),
                    'H': H.item(),
                    'strike': K.item(),
                    'maturity': T.item(),
                    'implied_volatility': iv
                })
    return data_points

results = Parallel(n_jobs=2)(
    delayed(compute_option_and_iv)(i) for i in tqdm(range(num_samples), desc="Computing Volatility Grids")
)

# Flatten results
dataset = [point for sublist in results for point in sublist]

dataset_df = pd.DataFrame(dataset)
dataset_df = dataset_df.sample(frac=1).reset_index(drop=True)

dataset_df.to_csv('rbergomi_dataset_3.csv', index=False)

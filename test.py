from math import log, sqrt, exp
import numpy as np
from numba import jit, prange
from scipy.stats import norm
from scipy.special import hyp2f1
# Parameters
n = 1000        # Number of time steps
m = 1000       # Number of Monte Carlo simulations
S0 = 100       # Initial stock price
r = 0.02       # Risk-free rate
T = 1          # Time horizon
K = 100        # Strike price
xi0 = 0.04     # Initial variance
eta = 0.3      # Volatility of volatility
rho = -0.7     # Correlation
H = 0.3        # Hurst exponent

# Generate Brownian motion increments
 
dw1 = np.random.randn(m, n)


def G(x, H):
    return (2 * H) / (0.5 + H) * x ** (0.5 - H) * hyp2f1(1, 0.5 - H, 1.5 + H, x)

def riemann_liouville(T, n, H, m):
    dt = T / n
    gamma = np.zeros((n, n))

    for i in prange(1, n+1):
        for j in prange(1, n+1):
            if j <= i:
                gamma[j-1, i-1] = ((j*dt)**(2*H)) * G((j*dt)/(i*dt), H)
            else:
                gamma[j-1, i-1] = ((i*dt)**(2*H)) * G((i*dt)/(j*dt), H)

    L = np.linalg.cholesky(gamma)
    X = L @ dw1.T
    return X.T

def variance(xi0, eta, riemann_liouville):
    m, n = riemann_liouville.shape
    v = np.zeros((m, n))
    for j in prange(m):
        for i in prange(n):
            RL = riemann_liouville[j, i]
            adjustment = 0.5 * (eta**2) * np.mean(riemann_liouville[j]**2)
            v[j, i] = xi0 * np.exp(eta * RL - adjustment)
    return v


def mc_sim(S0, n, m, r, T, xi0, eta, rho, H, dw1):
    RL = riemann_liouville(T, n, H, m)
    v = variance(xi0, eta, RL)
    dt = T / n

    S1 = np.zeros(m)
    int_v = np.zeros(m)

    for j in prange(m):
        log_S1 = np.log(S0)
        iv = 0.0
        for i in range(n):
            vu = v[j, i]
            dW1 = dw1[j, i] * np.sqrt(dt)
            log_S1 += (r - 0.5 * (rho**2) * vu) * dt + rho * np.sqrt(vu) * dW1
            iv += vu * dt
        S1[j] = np.exp(log_S1)
        int_v[j] = iv
    return S1, int_v

def black_scholes_call_price(S, K, r, T, sigma):
    if sigma <= 0 or T <= 0:
        return max(S - K * exp(-r*T), 0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    call = S * norm.cdf(d1) - K * exp(-r*T) * norm.cdf(d2)
    return call

def rbergomi_turbocharged_price(S0, K, n, m, r, T, xi0, eta, rho, H):
    S1, int_v = mc_sim(S0, n, m, r, T, xi0, eta, rho, H, dw1)
    hat_Q = np.max(int_v)
    
    X = np.zeros(m)
    Y = np.zeros(m)
    
    for i in prange(m):
        # Compute X_i
        X_vol = np.sqrt((1 - rho**2) * int_v[i] / T) if T > 0 else 0.0
        X[i] = black_scholes_call_price(S1[i], K, r, T, X_vol)
        
        # Compute Y_i
        Y_vol_sq = rho**2 * (hat_Q - int_v[i])
        Y_vol = np.sqrt(Y_vol_sq / T) if T > 0 and Y_vol_sq >= 0 else 0.0
        Y[i] = black_scholes_call_price(S1[i], K, r, T, Y_vol)
    
    # Compute hat_omega
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)
    cov = np.sum((X - X_mean) * (Y - Y_mean))
    var_Y = np.sum((Y - Y_mean)**2)
    hat_omega = -cov / var_Y if var_Y != 0 else 0.0
    
    # Turbocharging estimator
    estimator = (np.mean(X + hat_omega * Y) - hat_omega * Y_mean)
    return estimator

# Example usage
price = rbergomi_turbocharged_price(S0, K, n, m, r, T, xi0, eta, rho, H)
print({price})
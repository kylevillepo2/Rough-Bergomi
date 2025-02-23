from math import log, sqrt, exp, pi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, integrate
from scipy.stats import norm, qmc
from scipy.integrate import quad, trapezoid, quad_vec
from scipy.optimize import brentq
from scipy.interpolate import SmoothBivariateSpline
from scipy.special import hyp2f1
from numba import jit, prange
from math import log, sqrt, exp
from numba import jit
n=1000
m=100000
dw1 = np.random.randn(m, n)


def G(x, H):
    return (2 * H) / (1/2 + H) * x ** (1/2 - H) * hyp2f1(1, 1/2 - H, 3/2 + H, x)


def riemann_liouville(T, n, H, m):

    
    dt = T / n
    gamma = np.zeros((n, n))

    G_vec = np.vectorize(G)

    j = np.arange(1, n+1)   
    i = np.arange(1, n+1)  
    J, I = np.meshgrid(j, i, indexing='ij')

    ratio = np.where(J <= I, J / I, I / J)
    scaling = np.where(J <= I, (J * dt) ** (2 * H), (I * dt) ** (2 * H))

    gamma = scaling * G_vec(ratio, H)

    L = np.linalg.cholesky(gamma)
    X = L @ dw1.T
    X = np.vstack((np.zeros((1, m)), X)).T

    return X


def variance(xi0, eta, riemann_liouville):

    mean_sq = np.mean(np.abs(riemann_liouville) ** 2, axis=1)  

    riemann_slice = riemann_liouville[:, :n]
    vega = xi0 * np.exp(eta * riemann_slice - 0.5 * (eta ** 2) * mean_sq[:, None])
    return vega


def dz(dw1, rho, n, m):

    dw2 = np.random.randn(m, n)
    return rho * dw1 + np.sqrt(1 - rho ** 2) * dw2



def mc_sim(S0, n, m, r, T, xi0, eta, rho, H, whole_process=False):
    riemann_liouville_x = riemann_liouville(T, n, H, m) 
    vega = variance(xi0, eta, riemann_liouville_x) 
    dz1 = dz(dw1, rho, n, m) 
    
    dt = T / n

    increments = (r - 0.5 * vega**2) * dt + vega * np.sqrt(dt) * dz1

    if whole_process:

        log_prices = np.log(S0) + np.cumsum(increments, axis=1)
        prices = np.exp(log_prices)
    else:
  
        log_prices = np.log(S0) + np.sum(increments, axis=1)
        prices = np.exp(log_prices)
    
    return prices




def rbergomi_price(S0, n, m, r, T, K, xi0, eta, rho, H):
    prices = mc_sim(S0, n, m, r, T, xi0, eta, rho, H)

    discount_factor = np.exp(-r * T)
    payoffs = np.maximum(prices - K, 0)

    return np.mean(payoffs) * discount_factor


def black_scholes_call_price(S0, K, r, T, sigma):
    if sigma <= 0:
        return max(S0 - K * np.exp(-r * T), 0.0)
    
    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    call = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    return call




def implied_volatility(target_price, S0, K, r, T, tol=1e-8, max_iterations=100):
    objective = lambda sigma: black_scholes_call_price(S0, K, r, T, sigma) - target_price

    vol_lower = 1e-6
    vol_upper = 5.0  

    try:
        implied_vol = optimize.brentq(objective, vol_lower, vol_upper, xtol=tol, maxiter=max_iterations)
    except ValueError:
        implied_vol = np.nan

    return implied_vol 

S0 = 100       # Initial stock price
r = 0.02       # Risk-free rate
T = 1          # Time horizon (1 year)
xi0 = 0.04     # Initial variance
eta = 1.5     # Volatility of volatility
rho = -0.7     # Correlation between stock and variance
H = 0.3     # Hurst exponent (for rough volatility)
K = 100

print(rbergomi_price(S0, n, m, r, T, K, xi0, eta, rho, H))

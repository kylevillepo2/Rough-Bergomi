import numpy as np



prices = []

def asset_price_sim(S0, T, sigma, r, n):

    for i in range(n):

        St = S0 * np.exp((r - sigma ** 2 / 2) * T + (sigma * np.sqrt(T) * np.random.normal()))

        prices.append(St)


def european_call(prices, K, r, T):

    discount_factor = np.exp(-r * T)
    payoffs = []

    for price in prices:
        payoff = np.max(price - K, 0)
        payoffs.append(payoff)
    
    return np.mean(payoffs) * discount_factor

S0 = 250
T = 1
K = 250
sigma = 0.2
r = 0.04
n = 1000000

asset_price_sim(S0, T, sigma, r, n)
print(european_call(prices, K, r, T))




        
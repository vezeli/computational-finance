import numpy as np
import scipy.stats as st

from utils import UnknownOptionTypeError


def calculate_d1(s, k, tau, sigma, r):
    return  (np.log(s/k) + (r + 0.5*np.power(sigma, 2)*tau)) / (sigma*np.sqrt(tau))


def calculate_d2(s, k, tau, sigma, r):
    return calculate_d1(s, k, tau, sigma, r) - sigma * np.sqrt(tau)


def calculate_tau(T, t):
    return T-t


def calculate_option_price(cp, s, k, T, t, sigma, r):
    tau = calculate_tau(T, t)
    d1, d2 = calculate_d1(s, k, tau, sigma, r), calculate_d2(s, k, tau, sigma, r)
    if cp == 1:
        rv = st.norm.cdf(d1)*s - st.norm.cdf(d2)*k*np.exp(-r*tau)
    else:
        if cp == -1:
            rv = -st.norm.cdf(-d1)*s + st.norm.cdf(-d2)*k*np.exp(-r*tau)
        else:
            raise UnknownOptionTypeError
    return rv


def calculate_vega(s, k, T, t, sigma, r):
    tau = calculate_tau(T, t)
    d2 = calculate_d2(s, k, tau, sigma, r)
    return k*np.exp(-r*tau)*st.norm.pdf(d2)*np.sqrt(tau)


def _normal_distribution(n, mu, sigma):
    z = np.random.normal(mu, sigma, n)
    z_mod = (z - np.mean(z)) / np.std(z)
    return z_mod


def wiener_process(n):
    return _normal_distribution(n, 0, 1)

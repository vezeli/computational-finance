import numpy as np
import scipy.stats as st

from src.utils import UnknownOptionTypeError


def _d1(s, k, tau, sigma, r):
    return  (np.log(s/k) + (r + 0.5*np.power(sigma, 2)*tau)) / (sigma*np.sqrt(tau))


def _d2(s, k, tau, sigma, r):
    return _d1(s, k, tau, sigma, r) - sigma * np.sqrt(tau)


def _tau(T, t):
    return T-t


def V(cp, s, k, T, t, sigma, r):
    tau = _tau(T, t)
    d1, d2 = _d1(s, k, tau, sigma, r), _d2(s, k, tau, sigma, r)
    if cp == 1:
        rv = st.norm.cdf(d1)*s - st.norm.cdf(d2)*k*np.exp(-r*tau)
    else:
        if cp == -1:
            rv = -st.norm.cdf(-d1)*s + st.norm.cdf(-d2)*k*np.exp(-r*tau)
        else:
            raise UnknownOptionTypeError
    return rv


def dVdS(s, k, T, t, sigma, r):
    tau = _tau(T, t)
    d2 = _d2(s, k, tau, sigma, r)
    return k*np.exp(-r*tau)*st.norm.pdf(d2)*np.sqrt(tau)


def dN(mu, sigma, n):
    """Returns normal distribution that converges faster to N(\mu, \sigma)"""
    z = np.random.normal(mu, sigma, n)
    z_mod = (z - np.mean(z)) / np.std(z)
    return z_mod


def dW(n):
    return dN(0, 1, n)

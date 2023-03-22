from numbers import Real as R

import numpy as np
import numpy.typing as npt
import scipy.stats as st

from src.utils import CallPut, UnknownOptionTypeError


def _d1(s: R, k: R, tau: R, sigma: R, r: R) -> R:
    return  (np.log(s/k) + (r + 0.5*np.power(sigma, 2)*tau)) / (sigma*np.sqrt(tau))


def _d2(s: R, k: R, tau: R, sigma: R, r: R) -> R:
    return _d1(s, k, tau, sigma, r) - sigma * np.sqrt(tau)


def _tau(T: R, t: R) -> R:
    return T-t


def V(cp: CallPut, s: R, k: R, T: R, t: R, sigma: R, r: R) -> R:
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


def dVdS(s: R, k: R, T: R, t: R, sigma: R, r: R) -> R:
    tau = _tau(T, t)
    d2 = _d2(s, k, tau, sigma, r)
    return k*np.exp(-r*tau)*st.norm.pdf(d2)*np.sqrt(tau)


def dN(mu: R, sigma: R, n: int) -> npt.NDArray[R]:
    """Returns normal distribution that converges faster to N(\mu, \sigma)"""
    z = np.random.normal(mu, sigma, n)
    if n > 1:
        z = (z - np.mean(z)) / np.std(z)
    return z


def dW(size: int) -> npt.NDArray[R]:
    return dN(0, 1, size)

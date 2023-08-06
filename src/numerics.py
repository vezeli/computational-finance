from numbers import Integral as N
from numbers import Real as R

import numpy as np
from scipy import fft, interpolate

from src.base import V, dVdS
from src.utils import CallPut, CharacteristicFunction, i


def newton_raphson_method(
    cp: CallPut,
    v_market: R,
    s: R,
    k: R,
    T: R,
    t: R,
    sigma0: R,
    r: R
) -> R:
    ERR = 1E-10

    def option_price(sigma: R) -> R:
        nonlocal cp, s, k, T, t, r
        return V(cp, s, k, T, t, sigma, r)

    def vega(sigma: R) -> R:
        nonlocal s, k, T, t, r
        return dVdS(s, k, T, t, sigma, r)

    _count, _error = 1, 1E10
    while _error > ERR:
        if _count == 1: sigma = sigma0
        g = option_price(sigma) - v_market
        dg = vega(sigma)
        sigma -= g/dg
        _error = abs(g)
        
        if (_count := _count + 1) > 100:
            return RuntimeError
        else:
            continue

    return sigma


def _fft_density_recovery(
    cf: CharacteristicFunction,
    xs: np.ndarray,
    n: N,
    u_max: R
) -> [np.ndarray, np.ndarray]:
    du = u_max / n
    u = np.arange(n) * du

    dx = 2 * np.pi / n / du  # FFT requirement: dx du = 2 pi / n
    x = (b := xs.min()) + np.arange(n) * dx

    F = np.exp(-i * b * u) * cf(u)

    boundary_correction = 1 / 2 * (
        np.exp(-i * x * u[0]) * cf(u[0]) + np.exp(-i * x * u[-1]) * cf(u[-1])
    )

    f = du / np.pi * np.real( fft.fft(F) - boundary_correction )

    return x, f


def fft_density_recovery(
    cf: CharacteristicFunction,
    xs: np.ndarray,
    n: N,
    u_max: R
) -> np.ndarray:
    x, f = _fft_density_recovery(cf, xs, n, u_max)

    f2interpolate = interpolate.interp1d(x, f, kind="cubic")

    return f2interpolate(xs)

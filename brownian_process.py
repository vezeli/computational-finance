from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


def _normal_distribution(n, mu, sigma):
    z = np.random.normal(mu, sigma, n)
    z_mod = (z - np.mean(z)) / np.std(z)
    return z_mod


def wiener_process(n):
    return _normal_distribution(n, 0, 1)


@dataclass
class BrownianMotion:
    _xs: float
    _t: float
    dt: float
    r: float
    sigma: float

    def __post_init__(self):
        self.n = np.shape(xs)[0]

    def __iter__(self):
        return self

    def __next__(self):
        self._xs = self._xs + (self.r - 1/2*sigma**2)*dt + sigma*np.sqrt(dt)*wiener_process(self.n)
        self._t = self._t + dt
        return (self._xs, self._t)


if __name__ == "__main__":
    PATHS = 250
    ps = np.ones(PATHS)
    xs = np.log(ps)*np.ones(PATHS)
    t, dt = 0, 0.01
    r, sigma = 0.05, 0.4

    T = 10
    xss, pss, ts = np.array([xs]), np.array([ps]), np.array(t) 
    xt = BrownianMotion(_xs=xs, _t=t, dt=dt, r=r, sigma=sigma)
    while t < T:
        xs, t = next(xt)
        xss = np.vstack([xss, xs])
        pss = np.vstack([pss, np.exp(xs)])
        ts = np.append(ts, t)

    plt.figure(1)
    plt.plot(ts, xss)
    plt.grid()
    plt.title("Normal distribution")
    plt.xlabel("t")
    plt.ylabel("X_t")


    plt.figure(2)
    plt.plot(ts, pss)
    plt.grid()
    plt.title("Normal distribution")
    plt.xlabel("t")
    plt.ylabel("X_t")

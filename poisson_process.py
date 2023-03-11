from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class PoissonProcess:
    _xs: float
    _t: float
    dt: float
    xiP: float

    def __post_init__(self):
        self.n = np.shape(xs)[0]

    def __iter__(self):
        return self

    def __next__(self):
        self._xs = self._xs + np.random.poisson(xiP*dt, self.n)
        self._t = self._t + dt
        return (self._xs, self._t)


if __name__ == "__main__":
    PATHS = 25
    xs = np.zeros(PATHS)
    xcs = np.zeros(PATHS)
    xiP = 1
    t, dt = 0, 0.01

    T = 50
    xss, xcss, ts = np.array([xs]), np.array([xcs]), np.array(t) 
    xt = PoissonProcess(_xs=xs, _t=t, dt=dt, xiP=xiP)
    while t < T:
        xs, t = next(xt)
        xss = np.vstack([xss, xs])
        xcss = np.vstack([xcss, xs-xiP*t])
        ts = np.append(ts, t)

    plt.figure(1)
    plt.plot(ts, xss)
    plt.grid()
    plt.title("Poisson process")
    plt.xlabel("t")
    plt.ylabel("X_p")
    plt.figure(2)
    plt.plot(ts, xcss)
    plt.grid()
    plt.title("Compensated Poisson process")
    plt.xlabel("t")
    plt.ylabel("Xc_p")
    plt.show()

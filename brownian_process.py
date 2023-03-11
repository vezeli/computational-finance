import matplotlib.pyplot as plt
import numpy as np

from src.base import Process
from src.models import wiener_process


class BrownianMotion(Process):

    def __init__(self, r, sigma, **kwargs):
        self.r = r
        self.sigma = sigma
        super(BrownianMotion, self).__init__(**kwargs)

    def dX(self):
        drift = (self.r - 1/2*self.sigma**2)*self.dt
        diffusion = self.sigma*np.sqrt(self.dt)*wiener_process(self.paths)
        return drift + diffusion


if __name__ == "__main__":
    PATHS = 25
    ps = np.ones(PATHS)
    xs = np.log(ps)*np.ones(PATHS)
    t, dt = 0, 0.01
    r, sigma = 0.05, 0.2

    T = 10
    xss, pss, ts = np.array([xs]), np.array([ps]), np.array(t) 
    xt = BrownianMotion(r=r, sigma=sigma, xs=xs, t=t, dt=dt)
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
    plt.title("Log-normal distribution")
    plt.xlabel("t")
    plt.ylabel("S_t = exp(X_t)")
    plt.show()

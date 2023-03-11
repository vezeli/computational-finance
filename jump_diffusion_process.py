import matplotlib.pyplot as plt
import numpy as np

from src.base import Process
from src.models import wiener_process


class JumpDiffusion(Process):

    def __init__(self, r, sigma, muJ, sigmaJ, xiP, **kwargs):
        self.r = r
        self.sigma = sigma
        self.muJ = muJ
        self.sigmaJ = sigmaJ
        self.xiP = xiP
        super(JumpDiffusion, self).__init__(**kwargs)

    def dX(self):
        drift = (
            self.r * self.dt -
            self.xiP*(np.exp(self.muJ + 1/2*self.sigmaJ**2)-1) * self.dt -
            1/2*self.sigma**2 * self.dt
        )
        diffusion = self.sigma * np.sqrt(self.dt) * wiener_process(self.paths)
        jump = np.random.normal(self.muJ, self.sigmaJ, self.paths)* \
            np.random.poisson(self.xiP*self.dt, self.paths)
        return drift + diffusion + jump


if __name__ == "__main__":
    PATHS = 25
    s0 = 100
    ps = s0*np.ones(PATHS)
    xs = np.log(ps)*np.ones(PATHS)
    t, dt = 0, 0.01
    r, sigma = 0.05, 0.16
    muJ, sigmaJ, xiP = 0, 0.2, 1

    T = 5
    xss, pss, ts = np.array([xs]), np.array([ps]), np.array(t) 
    xt = JumpDiffusion(r=r, sigma=sigma, muJ=muJ, sigmaJ=sigmaJ, xiP=xiP, xs=xs, t=t, dt=dt)
    while t < T:
        xs, t = next(xt)
        xss = np.vstack([xss, xs])
        pss = np.vstack([pss, np.exp(xs)])
        ts = np.append(ts, t)

    plt.figure(1)
    plt.plot(ts, xss)
    plt.grid()
    plt.xlabel("t")
    plt.ylabel("X_t")

    plt.figure(2)
    plt.plot(ts, pss)
    plt.grid()
    plt.xlabel("t")
    plt.ylabel("S_t = exp(X_t)")

    plt.show()

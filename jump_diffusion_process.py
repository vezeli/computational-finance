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
        drift_term = self.dt * (
            self.r - self.xiP*(np.exp(self.muJ + 1/2*self.sigmaJ**2) - 1) -
            1/2*self.sigma**2
        )
        diffusion_term = self.sigma * np.sqrt(self.dt) * wiener_process(self.paths)
        jump_term = (
            np.random.normal(self.muJ, self.sigmaJ, self.paths) *
            np.random.poisson(self.xiP*self.dt, self.paths)
        )
        return drift_term + diffusion_term + jump_term


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
    xt = JumpDiffusion(
        r=r,
        sigma=sigma,
        muJ=muJ,
        sigmaJ=sigmaJ,
        xiP=xiP, xs=xs,
        t=t,
        dt=dt
    )
    while t < T:
        xs, t = next(xt)
        xss = np.vstack([xss, xs])
        pss = np.vstack([pss, np.exp(xs)])
        ts = np.append(ts, t)

    _, (ax1, ax2) = plt.subplots(2,1)
    ax1.plot(ts, xss)
    ax1.grid()
    ax1.set_xlabel("t")
    ax1.set_ylabel("X_t")
    ax2.plot(ts, pss)
    ax2.grid()
    ax2.set_xlabel("t")
    ax2.set_ylabel("S_t = exp(X_t)")
    plt.tight_layout()
    plt.show()

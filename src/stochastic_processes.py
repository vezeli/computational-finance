import numpy as np

from src.base import dN, dW


class _Process:
    def __init__(self, xs, t, dt):
        self.xs = xs
        self.t = t
        self.dt = dt

    @property
    def xs(self):
        return self._xs

    @xs.setter
    def xs(self, value):
        self._xs = value

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, value):
        self._t = value

    @property
    def paths(self):
        return np.shape(self.xs)[0]

    def __iter__(self):
        return self

    def __next__(self):
        self.xs += self.dX()
        self.t += self.dt
        return (self.xs, self.t)


class BrownianMotion(_Process):
    def __init__(self, r, sigma, **kwargs):
        self.r = r
        self.sigma = sigma
        super(BrownianMotion, self).__init__(**kwargs)

    def dX(self):
        drift_term = (self.r - 1/2*self.sigma**2)*self.dt
        diffusion_term = self.sigma*np.sqrt(self.dt)*dW(self.paths)
        return drift_term + diffusion_term


class Poisson(_Process):
    def __init__(self, xiP, **kwargs):
        self.xiP = xiP
        super(Poisson, self).__init__(**kwargs)

    def dX(self):
        return np.random.poisson(self.xiP*self.dt, self.paths)


class StandardJumpDiffusion(_Process):
    def __init__(self, r, sigma, muJ, sigmaJ, xiP, **kwargs):
        self.r = r
        self.sigma = sigma
        self.muJ = muJ
        self.sigmaJ = sigmaJ
        self.xiP = xiP
        super(StandardJumpDiffusion, self).__init__(**kwargs)

    def dX(self):
        drift_term = self.dt * (
            self.r - 1/2*self.sigma**2 -
            self.xiP*(np.exp(self.muJ + 1/2*self.sigmaJ**2) - 1)
        )
        diffusion_term = self.sigma * np.sqrt(self.dt) * dW(self.paths)
        jump_term = (
            np.random.normal(self.muJ, self.sigmaJ, self.paths) *
            np.random.poisson(self.xiP*self.dt, self.paths)
        )
        return drift_term + diffusion_term + jump_term

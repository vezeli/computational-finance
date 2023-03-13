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

    @staticmethod
    def _drift(r, sigma, dt):
        return (r - 1/2*sigma**2) * dt

    @staticmethod
    def _diffusion(sigma, dt, size):
        return sigma * np.sqrt(dt) * dW(size)

    def dX(self):
        x1 = BrownianMotion._drift(self.r, self.sigma, self.dt)
        x2 = BrownianMotion._diffusion(self.sigma, self.dt, self.paths)
        return x1 + x2


class Poisson(_Process):
    def __init__(self, xiP, **kwargs):
        self.xiP = xiP
        super(Poisson, self).__init__(**kwargs)

    @staticmethod
    def _jump(dt, xiP, size):
        return np.random.poisson(xiP*dt, size)

    def dX(self):
        return Poisson._jump(self.dt, self.xiP, self.paths)


class StandardJumpDiffusion(_Process):
    def __init__(self, r, sigma, muJ, sigmaJ, xiP, **kwargs):
        self.r = r
        self.sigma = sigma
        self.muJ = muJ
        self.sigmaJ = sigmaJ
        self.xiP = xiP
        super(StandardJumpDiffusion, self).__init__(**kwargs)

    @staticmethod
    def _drift(r, sigma, xiP, muJ, sigmaJ, dt):
        return (r - 1/2*sigma**2 - xiP*(np.exp(muJ+1/2*sigmaJ**2)-1))*dt

    @staticmethod
    def _jump(xiP, muJ, sigmaJ, dt, size):
        return np.random.normal(muJ, sigmaJ, size) * np.random.poisson(xiP*dt, size)

    @staticmethod
    def _diffusion(sigma, dt, size):
        return sigma * np.sqrt(dt) * dW(size)

    def dX(self):
        x1 = StandardJumpDiffusion._drift(
            self.r, self.sigma, self.xiP, self.muJ, self.sigmaJ, self.dt
        )
        x2 = StandardJumpDiffusion._jump(
            self.xiP, self.muJ, self.sigmaJ, self.dt, self.paths
        )
        x3 = StandardJumpDiffusion._diffusion(self.sigma, self.dt, self.paths)
        return x1 + x2 + x3

from numbers import Real as R
from typing import Callable

import numpy as np
import numpy.typing as npt

from src.base import dN, dW


class _Process:
    def __init__(self, xs: npt.NDArray[R], t: R, dt: R) -> None:
        self.xs = xs
        self.t = t
        self._dt = dt

    @property
    def xs(self) -> npt.NDArray[R]:
        return self._xs

    @xs.setter
    def xs(self, value: R) -> None:
        self._xs = value

    @property
    def t(self) -> R:
        return self._t

    @t.setter
    def t(self, value: R) -> None:
        self._t = value

    @property
    def dt(self) -> R:
        return self._dt

    @property
    def paths(self) -> int:
        return np.shape(self.xs)[0]

    def __iter__(self):
        return self

    def __next__(self) -> tuple[npt.NDArray[R], R]:
        self.xs += self.dX()
        self.t += self.dt
        return (self.xs, self.t)


class BrownianMotion(_Process):
    def __init__(sigma: R, **kwargs) -> None:
        self.sigma = sigma
        super(BrownianMotion, self).__init__(**kwargs)

    @staticmethod
    def _diffusion(sigma: R, dt: R, size: int) -> npt.NDArray[R]:
        return sigma * np.sqrt(dt) * dW(size)

    def dX(self) -> npt.NDArray[R]:
        return BrownianMotion._diffusion(self.sigma, self.dt, self.paths)


class CorrelatedBrownianMotions(_Process):
    def __init__( self, rho: R, sigma1: R, sigma2: R, **kwargs) -> None:
        if not (-1 < rho < 1) : raise ValueError
        self.rho = rho
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        super(CorrelatedBrownianMotions, self).__init__(**kwargs)

    @staticmethod
    def _diffusion(rho: R, sigma1: R, sigma2: R, dt: R) -> npt.NDArray[R]:
        l = np.linalg.cholesky(np.array([[1, rho], [rho, 1]])) 
        dW1 = BrownianMotion._diffusion(sigma1, dt, 1)
        dW2 = BrownianMotion._diffusion(sigma2, dt, 1)
        return l @ np.array([dW1, dW2])

    def dX(self) -> npt.NDArray[R]:
        return CorrelatedBrownianMotions._diffusion(self.rho, self.sigma1, self.sigma2, self.dt)


class StandardDiffusion(_Process):
    def __init__(self, r: R, sigma: R, **kwargs) -> None:
        self.r = r
        self.sigma = sigma
        super(StandardDiffusion, self).__init__(**kwargs)

    @staticmethod
    def _drift(r: R, sigma: R, dt: R) -> R:
        return (r - 1/2*sigma**2) * dt

    def dX(self) -> npt.NDArray[R]:
        x1 = StandardDiffusion._drift(self.r, self.sigma, self.dt)
        x2 = BrownianMotion._diffusion(self.sigma, self.dt, self.paths)
        return x1 + x2


class Poisson(_Process):
    def __init__(self, xiP: R, **kwargs) -> None:
        self.xiP = xiP
        super(Poisson, self).__init__(**kwargs)

    @staticmethod
    def _jump(dt, xiP: R, size: int) -> npt.NDArray[R]:
        return np.random.poisson(xiP*dt, size)

    def dX(self) -> npt.NDArray[R]:
        return Poisson._jump(self.dt, self.xiP, self.paths)


class StandardJumpDiffusion(_Process):
    def __init__(self, r: R, sigma: R, muJ: R, sigmaJ: R, xiP: R, **kwargs) -> None:
        self.r = r
        self.sigma = sigma
        self.muJ = muJ
        self.sigmaJ = sigmaJ
        self.xiP = xiP
        super(StandardJumpDiffusion, self).__init__(**kwargs)

    @staticmethod
    def _drift(r: R, sigma: R, xiP: R, muJ: R, sigmaJ: R, dt: R) -> R:
        return (r - 1/2*sigma**2 - xiP*(np.exp(muJ+1/2*sigmaJ**2)-1))*dt

    @staticmethod
    def _jump(xiP: R, muJ: R, sigmaJ: R, dt: R, size: int) -> npt.NDArray[R]:
        return np.random.normal(muJ, sigmaJ, size) * np.random.poisson(xiP*dt, size)

    def dX(self) -> npt.NDArray[R]:
        x1 = StandardJumpDiffusion._drift(
            self.r, self.sigma, self.xiP, self.muJ, self.sigmaJ, self.dt
        )
        x2 = StandardJumpDiffusion._jump(
            self.xiP, self.muJ, self.sigmaJ, self.dt, self.paths
        )
        x3 = BrownianMotion._diffusion(self.sigma, self.dt, self.paths)
        return x1 + x2 + x3

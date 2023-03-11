import numpy as np


class Process:
    def __init__(self, xs, t, dt):
        self.xs = xs
        self.t = t
        self.dt = dt

    def __iter__(self):
        return self

    @property
    def paths(self):
        return np.shape(self.xs)[0]

    def __next__(self):
        self.xs += self.dX()
        self.t += self.dt
        return (self.xs, self.t)

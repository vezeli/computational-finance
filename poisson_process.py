import matplotlib.pyplot as plt
import numpy as np

from base import Process


class Poisson(Process):

    def __init__(self, xiP, **kwargs):
        self.xiP = xiP
        super(Poisson, self).__init__(**kwargs)

    def dX(self):
        return np.random.poisson(xiP*dt, self.paths)


if __name__ == "__main__":
    PATHS = 25
    xs = np.zeros(PATHS)
    xcs = np.zeros(PATHS)
    xiP = 1
    t, dt = 0, 0.01

    T = 50
    xss, xcss, ts = np.array([xs]), np.array([xcs]), np.array(t) 
    xt = Poisson(xiP=xiP, xs=xs, t=t, dt=dt)
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

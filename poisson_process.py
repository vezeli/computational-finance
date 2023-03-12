import matplotlib.pyplot as plt
import numpy as np

from src.stochastic_processes import Poisson


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

    _, (ax1, ax2) = plt.subplots(2,1)
    ax1.plot(ts, xss)
    ax1.grid()
    ax1.set_title("Poisson process")
    ax1.set_xlabel("t")
    ax1.set_ylabel("X_p")
    ax2.plot(ts, xcss)
    ax2.grid()
    ax2.set_title("Compensated Poisson process")
    ax2.set_xlabel("t")
    ax2.set_ylabel("Xc_p")
    plt.tight_layout()
    plt.show()

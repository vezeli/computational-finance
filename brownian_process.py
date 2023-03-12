import matplotlib.pyplot as plt
import numpy as np

from src.processes import BrownianMotion


if __name__ == "__main__":
    PATHS = 25
    s0 = 100
    ps = s0*np.ones(PATHS)
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

    _, (ax1, ax2) = plt.subplots(2,1)
    ax1.plot(ts, xss)
    ax1.grid()
    ax1.set_title("Arithmetic Brownian motion")
    ax1.set_xlabel("t")
    ax1.set_ylabel("X_t")
    ax2.plot(ts, pss)
    ax2.grid()
    ax2.set_xlabel("t")
    ax2.set_ylabel("S_t = exp(X_t)")
    plt.tight_layout()
    plt.show()

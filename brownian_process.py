import matplotlib.pyplot as plt
import numpy as np

from src.stochastic_processes import BrownianMotion

PATHS = 25
PRICE = 100.00
START_TIME, END_TIME = 0, 10
TIMESTEP = 0.01
RISKFREE_RATE = 0.05
VOLATILITY = 0.20

ys = PRICE * np.ones(PATHS)
xs = np.log(ys) * np.ones(PATHS)
xss, yss, ts = np.array([xs]), np.array([ys]), np.array(START_TIME)
dXt = BrownianMotion(r=RISKFREE_RATE, sigma=VOLATILITY, xs=xs, t=START_TIME, dt=TIMESTEP)
for xs, t in dXt:
    xss = np.vstack([xss, xs])
    yss = np.vstack([yss, np.exp(xs)])
    ts = np.append(ts, t)
    if t > END_TIME:
        break
    else:
        continue

_, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(ts, xss)
ax1.grid()
ax1.set_title("dX_t = (r - 1/2 sigma^2) dt + sigma dB_t")
ax1.set_xlabel("t")
ax1.set_ylabel("X_t")
ax2.plot(ts, yss)
ax2.grid()
ax2.set_xlabel("t")
ax2.set_ylabel("S_t = exp(X_t)")
plt.tight_layout()
plt.show()

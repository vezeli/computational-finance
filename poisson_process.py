import matplotlib.pyplot as plt
import numpy as np

from src.stochastic_processes import Poisson

PATHS = 25
PRICE = 100.00
START_TIME, END_TIME = 0, 50
TIMESTEP = 0.01
JUMP_RATE = 1.00

xs = np.zeros(PATHS)
xcs = np.zeros(PATHS)
xss, xcss, ts = np.array([xs]), np.array([xcs]), np.array(START_TIME)
dXt = Poisson(xiP=JUMP_RATE, xs=xs, t=START_TIME, dt=TIMESTEP)
for xs, t, in dXt:
    xss = np.vstack([xss, xs])
    xcss = np.vstack([xcss, xs-JUMP_RATE*t])
    ts = np.append(ts, t)
    if t > END_TIME:
        break
    else:
        continue

_, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(ts, xss)
ax1.grid()
ax1.set_title("dXp_t")
ax1.set_xlabel("t")
ax1.set_ylabel("Xp_t")
ax2.plot(ts, xcss)
ax2.grid()
ax2.set_title("dXcp_t = dXp_t - epsilon_p * dt")
ax2.set_xlabel("t")
ax2.set_ylabel("Xcp_t")
plt.tight_layout()
plt.show()

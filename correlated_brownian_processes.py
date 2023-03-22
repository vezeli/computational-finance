import matplotlib.pyplot as plt
import numpy as np

from src.stochastic_processes import CorrelatedBrownianMotions

RHO = -0.95
START_TIME, END_TIME = 0, 10
TIMESTEP = 0.01

xs = np.array([[10.], [0.]])
wssX, wssY, ts = np.array(10), np.array(0), np.array(START_TIME)
dXt = CorrelatedBrownianMotions(RHO, 1, 0, 2, 2, xs, START_TIME, TIMESTEP)
for wsX, wsY, t  in dXt:
    wssX = np.vstack([wssX, wsX])
    wssY = np.vstack([wssY, wsY])
    ts = np.append(ts, t)
    if t > END_TIME:
        break
    else:
        continue

_, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(ts, wssX)
ax1.plot(ts, wssY)
ax1.grid()
ax1.set_title("dX_t = (r - 1/2 sigma^2) dt + sigma dB_t")
ax1.set_xlabel("t")
ax1.set_ylabel("X_t")
#ax2.plot(ts, yss)
#ax2.grid()
#ax2.set_xlabel("t")
#ax2.set_ylabel("S_t = exp(X_t)")
#plt.tight_layout()
#plt.show()

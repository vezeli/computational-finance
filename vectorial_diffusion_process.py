import matplotlib.pyplot as plt
import numpy as np

from src.stochastic_processes import CorrelatedBrownianMotions

INIT_WS = np.array([[10], [10]], dtype="f")
START_TIME = 0
END_TIME = 10
TIMESTEP = 0.01


def run_simulation(rho, sigma1, sigma2):
    global INIT_WS, START_TIME, END_TIME, TIMESTEP

    Ws, ts = np.array(INIT_WS, copy=True), np.array(START_TIME, copy=True)
    dWs_t = CorrelatedBrownianMotions(
        rho=rho,
        sigma1=sigma1,
        sigma2=sigma2,
        xs=np.copy(Ws),
        t=START_TIME,
        dt=TIMESTEP
    )
    for dWs, t  in dWs_t:
        Ws = np.concatenate([Ws, dWs], axis=1)
        ts = np.append(ts, t)
        if t > END_TIME:
            break
        else:
            continue

    return Ws, ts


r = lambda x: round(x, 2)


CORRELATION1, CORRELATION2, CORRELATION3 = 0.85, 0.00, -0.85

VOLATILITY1, VOLATILITY2 = 0.20, 0.25
Ws1, ts1 = run_simulation(CORRELATION1, VOLATILITY1, VOLATILITY2)
Ws2, ts2 = run_simulation(CORRELATION2, VOLATILITY1, VOLATILITY2)
Ws3, ts3 = run_simulation(CORRELATION3, VOLATILITY1, VOLATILITY2)

_, (ax1, ax2, ax3) = plt.subplots(3,1)
ax1.set_title(f"Correlation = {r(CORRELATION1)}")
ax1.plot(ts1, Ws1[0, :], "blue", label=f"sigma={r(VOLATILITY1)}")
ax1.plot(ts1, Ws1[1, :], "orange", label=f"sigma={r(VOLATILITY2)}")
ax1.grid()
ax1.set_xlabel("time")
ax1.legend()
ax2.set_title(f"Correlation = {r(CORRELATION2)}")
ax2.plot(ts2, Ws2[0, :], "blue", label=f"sigma={r(VOLATILITY1)}")
ax2.plot(ts2, Ws2[1, :], "orange", label=f"sigma={r(VOLATILITY2)}")
ax2.grid()
ax2.legend()
ax2.set_xlabel("time")
ax3.set_title(f"Correlation = {r(CORRELATION3)}")
ax3.plot(ts3, Ws3[0, :], "blue", label=f"sigma={r(VOLATILITY1)}")
ax3.plot(ts3, Ws3[1, :], "orange", label=f"sigma={r(VOLATILITY2)}")
ax3.grid()
ax3.legend()
ax3.set_xlabel("time")
plt.tight_layout()

VOLATILITY1, VOLATILITY2 = 0.25, 0.50
Ws1, ts1 = run_simulation(CORRELATION1, VOLATILITY1, VOLATILITY2)
Ws2, ts2 = run_simulation(CORRELATION2, VOLATILITY1, VOLATILITY2)
Ws3, ts3 = run_simulation(CORRELATION3, VOLATILITY1, VOLATILITY2)

_, (ax1, ax2, ax3) = plt.subplots(3,1)
ax1.set_title(f"Correlation = {r(CORRELATION1)}")
ax1.plot(ts1, Ws1[0, :], "blue", label=f"sigma={r(VOLATILITY1)}")
ax1.plot(ts1, Ws1[1, :], "orange", label=f"sigma={r(VOLATILITY2)}")
ax1.grid()
ax1.set_xlabel("time")
ax1.legend()
ax2.set_title(f"Correlation = {r(CORRELATION2)}")
ax2.plot(ts2, Ws2[0, :], "blue", label=f"sigma={r(VOLATILITY1)}")
ax2.plot(ts2, Ws2[1, :], "orange", label=f"sigma={r(VOLATILITY2)}")
ax2.grid()
ax2.legend()
ax2.set_xlabel("time")
ax3.set_title(f"Correlation = {r(CORRELATION3)}")
ax3.plot(ts3, Ws3[0, :], "blue", label=f"sigma={r(VOLATILITY1)}")
ax3.plot(ts3, Ws3[1, :], "orange", label=f"sigma={r(VOLATILITY2)}")
ax3.grid()
ax3.legend()
ax3.set_xlabel("time")
plt.tight_layout()

plt.show()

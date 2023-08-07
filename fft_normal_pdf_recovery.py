import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.numerics import fft_density_recovery
from src.utils import i


def characteristic_function(mu, sigma):
    """
    Returns characteristic function of normal distribution with the mean `mu`
    and the standard deviation `sigma`.
    """
    return lambda u: np.exp(i * mu * u - 1 / 2 * sigma**2 * u**2)


MEAN = 10
STANDARD_DEVIATION = 1.25

GRID_POINTS = 2**10
UPPER_INTEGRATION_BOUNDARY = 5

"""
Localized PDFs with strong gradients (e.g., small standard deviation) must be
solved with larger values of `UPPER_INTEGRATION_BOUNDARY` to correctly
integrate the characteristic equation in the phase space. Furthermore, increase
`GRID_POINTS` to improve the convergence of the FFT algorithm.
"""

_, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})

u = np.linspace(0, UPPER_INTEGRATION_BOUNDARY, GRID_POINTS)

f = characteristic_function(mu=MEAN, sigma=STANDARD_DEVIATION)(u=u)
ax.plot(np.real(f), u, np.imag(f))
ax.set_xlabel("Real[$f(u)$]")
ax.set_ylabel("$u$")
ax.set_zlabel("Imaginary[$f(u)$]")

# PDF solution:
_, ax = plt.subplots(1,1)

x = np.linspace(0, 20, 100)

pdf_numerical = fft_density_recovery(
    cf=characteristic_function(mu=MEAN, sigma=STANDARD_DEVIATION),
    xs=x,
    n=GRID_POINTS,
    u_max=UPPER_INTEGRATION_BOUNDARY
)

pdf_theoretical = stats.norm.pdf(
    x=x,
    loc=MEAN,
    scale=STANDARD_DEVIATION
)

ax.plot(x, pdf_numerical, 'r-')
ax.plot(x, pdf_theoretical, 'b--')
ax.set_xlabel("x")
ax.set_ylabel("$f_{NORMAL}(x)$")

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


MEAN = 0.25
STANDARD_DEVIATION = 0.5

GRID_POINTS = 2**8
UPPER_INTEGRATION_BOUNDARY = 80

"""
Localized PDFs with strong gradients (e.g., small standard deviation) must be
solved with larger values of `UPPER_INTEGRATION_BOUNDARY` to correctly
integrate the characteristic equation in the phase space. Furthermore, increase
`GRID_POINTS` to improve the convergence of the FFT algorithm.
"""

x = np.linspace(-1, 1, 100)

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

plt.plot(x, pdf_numerical, 'r-')
plt.plot(x, pdf_theoretical, 'b--')

plt.xlabel("x")
plt.ylabel("$f_{NORMAL}(x)$")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.numerics import cos_density_recovery
from src.utils import i


def characteristic_function(mu, sigma):
    """
    Returns characteristic function of normal distribution with the mean `mu`
    and the standard deviation `sigma`.
    """
    return lambda u: np.exp(i * mu * u - 1 / 2 * sigma**2 * u**2)


MEAN = 1.5
STANDARD_DEVIATION = 0.5

N = 2**6

"""
Increasing the number of terms in the Fourier expansion improves the accuracy
of the COS density recovery method. Unlike the FFT density recovery method, in
the COS density recovery method there are no requirement to integrate the
characteristic function. This has a significant and positive impact on the
performance of the COS density recovery method.
"""

_, ax = plt.subplots(1, 1)

a, b = -10, 10
xs = np.linspace(-1, 4, 100)

pdf_numerical = cos_density_recovery(
    cf=characteristic_function(mu=MEAN, sigma=STANDARD_DEVIATION),
    xs=xs,
    n=N,
    a=a,
    b=b,
)

pdf_theoretical = stats.norm.pdf(x=xs, loc=MEAN, scale=STANDARD_DEVIATION)

ax.plot(xs, pdf_numerical, "r-", label="numerical solution")
ax.plot(xs, pdf_theoretical, "b--", label="theoretical solution")
ax.set_xlabel("x")
ax.set_ylabel("$f_{NORMAL}(x)$")
ax.legend()
plt.show()

from src.base import V
from src.numerics import newton_raphson_method
from src.utils import CallPut

OPTION = CallPut(1)
PRICE = 5.00
UNDERLYING_PRICE = 100.00
STRIKE = 120.00
START_TIME, END_TIME = 0, 1
RISKFREE_RATE = 0.05
INITIAL_VOLATILITY = 0.15

print(f"Market option price: {round(PRICE, 2)}")
implied_volatility = newton_raphson_method(
    OPTION,
    PRICE,
    UNDERLYING_PRICE,
    STRIKE,
    END_TIME,
    START_TIME,
    INITIAL_VOLATILITY,
    RISKFREE_RATE
)
print(f"Implied volatility: {round(implied_volatility, 4)}")
bsm_v = V(
    OPTION,
    UNDERLYING_PRICE,
    STRIKE,
    END_TIME,
    START_TIME,
    implied_volatility,
    RISKFREE_RATE)
print(f"Black-Scholes option price: {round(bsm_v, 2)}")

from src.base import V
from src.numerics import newton_raphson_method


def main():
    cp = 1 # call: cp=1; put: cp=-1
    v = 5.0
    s, k = 100, 120
    t, T = 0, 1
    r = 0.05
    s0 = 0.15

    sigma_i = newton_raphson_method(cp, v, s, k, T, t, s0, r)
    bsm_v = V(cp, s, k, T, t, sigma_i, r)

    print(f"Implied volatility: {round(sigma_i, 4)}")
    print(f"Market price: {round(v, 2)}")
    print(f"Theoretical price: {round(bsm_v, 2)}")


if __name__ == "__main__":
    main()

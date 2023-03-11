import models
import solvers


def main():
    cp = 1 # call: cp=1; put: cp=-1
    v = 5.0
    s, k = 100, 120
    t, T = 0, 1
    r = 0.05
    s0, epsilon = 0.15, 1E-10

    sigma_i = solvers.newton_raphson_method(cp, v, s, k, T, t, s0, r, epsilon)
    bsm_v = models.calculate_option_price(cp, s, k, T, t, sigma_i, r)

    print(f"Implied volatility: {round(sigma_i, 4)}")
    print(f"Market price: {round(v, 2)}")
    print(f"Theoretical price: {round(bsm_v, 2)}")

    return None

if __name__ == "__main__":
    main()

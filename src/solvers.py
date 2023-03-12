from src.models import calculate_option_price, calculate_vega


def newton_raphson_method(cp, v_market, s, k, T, t, sigma0, r, eps):

    def vega(sigma):
        nonlocal s, k, T, t, r
        return calculate_vega(s, k, T, t, sigma, r)

    def option_price(sigma):
        nonlocal cp, s, k, T, t, r
        return calculate_option_price(cp, s, k, T, t, sigma, r)

    _count, error = 1, 1E10
    while error > eps:
        if _count == 1: sigma = sigma0
        g = option_price(sigma) - v_market
        dg = vega(sigma)
        sigma -= g/dg
        error = abs(g)
        
        if (_count := _count + 1) > 100:
            return RuntimeError
        else:
            continue

    return sigma

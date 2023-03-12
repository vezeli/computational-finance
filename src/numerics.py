from src.base import V, dVdS


def newton_raphson_method(cp, v_market, s, k, T, t, sigma0, r):
    ERR = 1E-10

    def option_price(sigma):
        nonlocal cp, s, k, T, t, r
        return V(cp, s, k, T, t, sigma, r)

    def vega(sigma):
        nonlocal s, k, T, t, r
        return dVdS(s, k, T, t, sigma, r)

    _count, _error = 1, 1E10
    while _error > ERR:
        if _count == 1: sigma = sigma0
        g = option_price(sigma) - v_market
        dg = vega(sigma)
        sigma -= g/dg
        _error = abs(g)
        
        if (_count := _count + 1) > 100:
            return RuntimeError
        else:
            continue

    return sigma

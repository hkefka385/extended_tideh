
def infectious_rate_tweets(t, p0=0.001, r0=0.424, phi0=0.125, taum=2.):
    bounds = [(-1, 0.5), (1, events[-1][0] /24 * 2)]
    try:
        if bounds is not None:
            if not (bounds[0][0] < r0 < bounds[1][0]):
                r0 = max(bounds[0][0], bounds[1][0] * sigmoid(taum / bounds[1][0]))
            if not (bounds[0][1] < taum < bounds[1][1]):
                taum = max(bounds[0][1], bounds[1][1] * sigmoid(taum / bounds[1][1]))

        return p0 * (1. + r0 * sin(2 * pi * (t / 24 + phi0))) * exp(-t / (24 * taum))
    except:
        return 0

def kernel_primitive_zhao(x, s0=0.08333, theta=0.242):
    c0 = 1.0 / s0 / (1 - 1.0 / -theta)
    if x < 0:
        return 0
    elif x <= s0: #5分以下
        return c0 * x
    else:
        return c0 * (s0 + (s0 * (1 - (x / s0) ** -theta)) / theta)

def integral_zhao(x1, x2, s0=0.08333, theta=0.242):
    return kernel_primitive_zhao(x2, s0, theta) - kernel_primitive_zhao(x1, s0, theta)

def estimate_K(events, t_start, t_end, kernel_integral):
    kernel_int = [fol_cnt * kernel_integral(t_start - event_time, t_end - event_time) for event_time, fol_cnt in events]
    return sum(kernel_int)

def kernel_zhao(s, s0=0.08333, theta=0.242):
    c0 = 1.0 / s0 / (1 - 1.0 / -theta)  # normalization constant
    if s >= 0:
        if s <= s0:
            return c0
        else:
            return c0 * (s / s0) ** (-(1. + theta))
    else:
        return 0
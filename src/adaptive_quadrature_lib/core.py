import math

_MAX_RECURSION_DEPTH = 80


class AdaptiveSimpsonResult:
    def __init__(self, value, error_estimate, intervals, converged):
        self.value = value
        self.error_estimate = error_estimate
        self.intervals = intervals
        self.converged = converged

    def __repr__(self):
        return (
            f"AdaptiveSimpsonResult(value={self.value!r}, "
            f"error_estimate={self.error_estimate!r}, "
            f"intervals={self.intervals!r}, converged={self.converged!r})"
        )


def _simpson(f, a, fa, b, fb, m, fm):
    return (b - a) / 6.0 * (fa + 4.0 * fm + fb)


def _adaptive_step(f, a, fa, b, fb, whole, tol, depth):
    m = 0.5 * (a + b)
    fm = f(m)
    lm = 0.5 * (a + m)
    flm = f(lm)
    rm = 0.5 * (m + b)
    frm = f(rm)
    left = _simpson(f, a, fa, m, fm, lm, flm)
    right = _simpson(f, m, fm, b, fb, rm, frm)
    delta = left + right - whole
    if depth <= 0 or abs(delta) <= 15.0 * tol:
        return left + right + delta / 15.0, whole, 1, depth <= 0 and abs(delta) > 15.0 * tol
    v1, e1, i1, fail1 = _adaptive_step(f, a, fa, m, fm, left, tol * 0.5, depth - 1)
    v2, e2, i2, fail2 = _adaptive_step(f, m, fm, b, fb, right, tol * 0.5, depth - 1)
    return v1 + v2, e1 + e2 + delta, i1 + i2, fail1 or fail2


def adaptive_simpson(f, a, b, tol=1e-10):
    if not callable(f):
        raise TypeError("f must be callable")
    if not isinstance(a, (int, float)):
        raise TypeError("a must be a real number")
    if not isinstance(b, (int, float)):
        raise TypeError("b must be a real number")
    if math.isnan(a) or math.isnan(b):
        raise ValueError("interval endpoints must not be NaN")
    if not isinstance(tol, (int, float)):
        raise TypeError("tol must be a real number")
    if not tol > 0 or math.isnan(tol) or math.isinf(tol):
        raise ValueError("tol must be a positive finite number")
    if a == b:
        return AdaptiveSimpsonResult(0.0, 0.0, 0, True)
    fa = f(a)
    fb = f(b)
    m = 0.5 * (a + b)
    fm = f(m)
    whole = _simpson(f, a, fa, b, fb, m, fm)
    value, error_estimate, intervals, failed = _adaptive_step(f, a, fa, b, fb, whole, tol, _MAX_RECURSION_DEPTH)
    return AdaptiveSimpsonResult(value, error_estimate, intervals, not failed)

# adaptive_quadrature_lib

Integrates a scalar function on an interval using adaptive Simpson's rule to a specified tolerance.

```python
from adaptive_quadrature_lib import adaptive_simpson

result = adaptive_simpson(lambda x: x**2, 0.0, 1.0, tol=1e-10)
print(result.value)
```

## Exports

`adaptive_simpson(f, a, b, tol=1e-10)` returns an `AdaptiveSimpsonResult` with `.value`, `.error_estimate`, `.intervals`, and `.converged`. Only these two names are exported.

## Why

Numerical integration on problems where the integrand has localized structure — a sharp peak, a boundary layer — benefits from refinement that follows the function rather than a uniform grid. Adaptive Simpson's is a straightforward, well-understood method that does exactly this. The trade-off versus higher-order or Clenshaw-Curtis quadrature is simplicity and a less favorable error constant; the trade-off versus fixed-grid Simpson's is a small overhead per subinterval and non-deterministic point placement.

## The awkward edge

The function must be finite and defined at every point sampled, including the endpoints. A singularity at an endpoint or a NaN anywhere will propagate into the result. For integrals with endpoint singularities, transform the integrand or split the interval before calling this library.

`converged` is `False` when the recursion limit (depth 80, roughly 2^80 subdivisions) is reached without the error estimate falling below tolerance.

## Performance

The window keeps a bounded buffer, so `push` is constant time and memory does not
grow with the length of the stream. `peak` and `trough` are linear in the window
size, which is the trade that keeps `push` cheap.


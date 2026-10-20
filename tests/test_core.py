import math
import unittest

from adaptive_quadrature_lib import adaptive_simpson, AdaptiveSimpsonResult


class TestAdaptiveSimpson(unittest.TestCase):
    def test_polynomial_exact(self):
        r = adaptive_simpson(lambda x: x * x, 0.0, 1.0, 1e-12)
        self.assertIsInstance(r, AdaptiveSimpsonResult)
        self.assertAlmostEqual(r.value, 1.0 / 3.0, places=12)
        self.assertTrue(r.converged)

    def test_constant_function(self):
        r = adaptive_simpson(lambda x: 5.0, -3.0, 7.0, 1e-12)
        self.assertAlmostEqual(r.value, 50.0, places=12)
        self.assertTrue(r.converged)

    def test_negative_interval(self):
        r = adaptive_simpson(lambda x: x * x, 1.0, 0.0, 1e-12)
        self.assertAlmostEqual(r.value, -1.0 / 3.0, places=12)

    def test_zero_width_interval(self):
        r = adaptive_simpson(lambda x: 1.0 / x, 0.0, 0.0)
        self.assertEqual(r.value, 0.0)
        self.assertEqual(r.error_estimate, 0.0)
        self.assertEqual(r.intervals, 0)
        self.assertTrue(r.converged)

    def test_degenerate_interval_with_nan_return(self):
        r = adaptive_simpson(lambda x: math.nan, 2.0, 2.0)
        self.assertEqual(r.value, 0.0)
        self.assertTrue(r.converged)

    def test_gaussian(self):
        r = adaptive_simpson(lambda x: math.exp(-x * x), -3.0, 3.0, 1e-10)
        self.assertAlmostEqual(r.value, math.sqrt(math.pi) * math.erf(3.0), places=8)
        self.assertTrue(r.converged)

    def test_sin(self):
        r = adaptive_simpson(math.sin, 0.0, math.pi, 1e-12)
        self.assertAlmostEqual(r.value, 2.0, places=12)

    def test_repeated_evaluations_no_state(self):
        calls = {'n': 0}
        def f(x):
            calls['n'] += 1
            return x * x
        r1 = adaptive_simpson(f, 0.0, 1.0, 1e-10)
        r2 = adaptive_simpson(f, 0.0, 1.0, 1e-10)
        self.assertAlmostEqual(r1.value, r2.value, places=12)
        self.assertEqual(r1.intervals, r2.intervals)

    def test_repr_format(self):
        r = adaptive_simpson(lambda x: 1.0, 0.0, 1.0)
        s = repr(r)
        self.assertIn("AdaptiveSimpsonResult", s)
        self.assertIn("value=", s)
        self.assertIn("error_estimate=", s)
        self.assertIn("intervals=", s)
        self.assertIn("converged=", s)

    def test_invalid_function(self):
        with self.assertRaises(TypeError):
            adaptive_simpson("not a function", 0.0, 1.0)

    def test_invalid_interval_type(self):
        with self.assertRaises(TypeError):
            adaptive_simpson(lambda x: x, "0", 1.0)

    def test_nan_interval(self):
        with self.assertRaises(ValueError):
            adaptive_simpson(lambda x: x, float('nan'), 1.0)

    def test_invalid_tolerance(self):
        with self.assertRaises(ValueError):
            adaptive_simpson(lambda x: x, 0.0, 1.0, tol=0.0)
        with self.assertRaises(ValueError):
            adaptive_simpson(lambda x: x, 0.0, 1.0, tol=-1e-10)
        with self.assertRaises(ValueError):
            adaptive_simpson(lambda x: x, 0.0, 1.0, tol=float('inf'))
        with self.assertRaises(ValueError):
            adaptive_simpson(lambda x: x, 0.0, 1.0, tol=float('nan'))

    def test_invalid_tolerance_type(self):
        with self.assertRaises(TypeError):
            adaptive_simpson(lambda x: x, 0.0, 1.0, tol="1e-10")

    def test_loose_tolerance(self):
        r = adaptive_simpson(lambda x: x * x, 0.0, 1.0, 1e-2)
        self.assertAlmostEqual(r.value, 1.0 / 3.0, places=8)


if __name__ == '__main__':
    unittest.main()

"""Run with: python -m unittest discover -s tests -v"""
import unittest
import numpy as np

from src.data.lorenz import lorenz_rhs, simulate_lorenz
from src.data.van_der_pol import van_der_pol_rhs, simulate_van_der_pol
from src.data.preprocessing import make_dataset


class DataGenerationTests(unittest.TestCase):
    def test_ode_equations(self):
        np.testing.assert_allclose(lorenz_rhs(0.0, np.array([1., 1., 1.])), [0., 26., -5./3.])
        np.testing.assert_allclose(van_der_pol_rhs(0.0, np.array([2., 0.])), [0., -2.])

    def test_lorenz_integration(self):
        time, trajectory = simulate_lorenz(n_steps=100)
        self.assertEqual(time.shape, (100,))
        self.assertEqual(trajectory.shape, (100, 3))
        self.assertTrue(np.isfinite(trajectory).all())

    def test_van_der_pol_integration(self):
        time, trajectory = simulate_van_der_pol(n_steps=100)
        self.assertEqual(time.shape, (100,))
        self.assertEqual(trajectory.shape, (100, 2))
        self.assertTrue(np.isfinite(trajectory).all())

    def test_preprocessing_reproducible_and_training_normalized(self):
        time, trajectory = simulate_lorenz(n_steps=100)
        first = make_dataset(time, trajectory, seed=42)
        second = make_dataset(time, trajectory, seed=42)
        np.testing.assert_array_equal(first["x_obs"], second["x_obs"])
        train = first["x_obs"][:first["train_end"]]
        np.testing.assert_allclose(train.mean(axis=0), 0.0, atol=1e-6)
        np.testing.assert_allclose(train.std(axis=0), 1.0, atol=1e-6)
        self.assertGreater(np.mean((first["x_obs"] - first["x_true"]) ** 2), 0.0)


if __name__ == "__main__":
    unittest.main()

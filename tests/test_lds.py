"""Small numerical checks of Kalman + RTS + EM, and integration with Phase 1."""
import unittest
import numpy as np

from src.models.lds import LinearGaussianSSM


class LDSTests(unittest.TestCase):
    @staticmethod
    def toy_model() -> LinearGaussianSSM:
        m = LinearGaussianSSM(n_latent=1)
        m.F = np.array([[0.8]])
        m.h = np.array([0.1])
        m.B = np.array([[1.]])
        m.Q = np.array([0.3])
        m.R = np.array([1.])
        m.mu0 = np.array([0.])
        m.P0 = np.array([[1.]])
        return m

    def test_one_step_kalman_update(self):
        m = self.toy_model()
        out = m.filter(np.array([[2.], [0.]]))
        self.assertAlmostEqual(float(out.filtered_mean[0, 0]), 1.0, places=9)
        self.assertAlmostEqual(float(out.filtered_cov[0, 0, 0]), 0.5, places=9)
        self.assertTrue(np.isfinite(out.log_likelihood))

    def test_rts_matches_exact_two_state_posterior(self):
        m = self.toy_model()
        x = np.array([[1.4], [-0.8]])
        out = m.smooth(x)
        # Exact Gaussian conditioning on a two-state chain:
        # prior z0~N(0,1), z1|z0~N(0.8z0+0.1, 0.3); x_t|z_t~N(z_t,1).
        cov_prior = np.array([[1., 0.8], [0.8, 0.8**2 + 0.3]])
        mean_prior = np.array([0., 0.1])
        P = np.linalg.inv(np.linalg.inv(cov_prior) + np.eye(2))
        post_mean = P @ (np.linalg.solve(cov_prior, mean_prior) + x.ravel())
        np.testing.assert_allclose(out.smoothed_mean[:, 0], post_mean, atol=1e-10)
        np.testing.assert_allclose(out.smoothed_cov[:, 0, 0], np.diag(P), atol=1e-10)
        np.testing.assert_allclose(out.cross_cov[0, 0, 0], P[1, 0], atol=1e-10)

    def test_fit_reproducible_and_finite(self):
        rng = np.random.default_rng(12)
        z = np.empty(130)
        z[0] = 0.3
        for i in range(1, len(z)):
            z[i] = 0.85*z[i-1] + 0.05 + rng.normal(scale=0.15)
        obs = (z + rng.normal(scale=0.15, size=len(z)))[:, None]
        first = LinearGaussianSSM(n_latent=1, seed=17).fit(obs[:90], n_iter=12)
        second = LinearGaussianSSM(n_latent=1, seed=17).fit(obs[:90], n_iter=12)
        np.testing.assert_array_equal(first.F, second.F)
        self.assertTrue(np.isfinite(first.history_).all())
        self.assertGreater(first.history_[-1], first.history_[0] - 1e-5)
        self.assertLess(first.spectral_radius, 1.0)
        self.assertTrue(np.isfinite(list(first.forecast_mse(obs, 90).values())).all())
        self.assertEqual(first.rollout(np.array([0.]), 30)[1].shape, (30, 1))

    def test_causality_and_train_only_fit(self):
        rng = np.random.default_rng(18)
        x = rng.normal(size=(90, 2))
        model = LinearGaussianSSM(n_latent=2).fit(x[:65], n_iter=5)
        saved_F = model.F.copy()
        changed_future = x.copy()
        changed_future[-1] = [100., 100.]
        filt1 = model.filter(x)
        filt2 = model.filter(changed_future)
        np.testing.assert_allclose(filt1.filtered_mean[:-1], filt2.filtered_mean[:-1], atol=1e-12)
        np.testing.assert_allclose(model.F, saved_F, atol=0.)
        mse = model.forecast_mse(x, 65, (1, 5, 10))
        self.assertEqual(set(mse), {1, 5, 10})
        self.assertTrue(all(np.isfinite(list(mse.values()))))

    def test_reject_nans_and_bad_horizons(self):
        with self.assertRaises(ValueError):
            LinearGaussianSSM(2).fit(np.array([[1., float('nan')], [0., 2.]]))
        model = LinearGaussianSSM(1).fit(np.zeros((40, 1)), n_iter=2)
        with self.assertRaises(ValueError):
            model.forecast_mse(np.zeros((40, 1)), 30, (11,))


if __name__ == '__main__':
    unittest.main()

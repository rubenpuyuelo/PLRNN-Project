"""Unit/integration tests for the MAP PLRNN-SSM and occupancy metrics."""
import tempfile
from pathlib import Path
import unittest
import numpy as np

from src.evaluation.dynamics import occupancy_metrics
from src.models.plrnn import PiecewiseLinearSSM, relu


class PLRNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(4)
        t = np.arange(115) * 0.06
        cls.x = np.column_stack((np.sin(t), np.cos(t))) + rng.normal(0, .035, (len(t),2))
        cls.train_end = 80
        cls.model = PiecewiseLinearSSM(6, seed=7, ridge_dyn=5).fit(
            cls.x[:cls.train_end], schedule=(1.,.2), outer_iters=2, latent_steps=15)

    def test_relu_and_model_structure(self):
        np.testing.assert_array_equal(relu(np.array([-1.,0.,2.])), [0.,0.,2.])
        m = self.model
        self.assertEqual(m.B.shape, (2,6))
        self.assertEqual(m.W.shape, (6,6))
        np.testing.assert_allclose(np.diag(m.W), 0, atol=0)
        self.assertTrue(np.isfinite(m.A).all())
        self.assertEqual(m.latent_train_.shape, (self.train_end,6))
        self.assertEqual(len(m.history_),4)
        self.assertEqual([r['process_var'] for r in m.history_], [1.,1.,.2,.2])

    def test_eq1_and_eq2_identity(self):
        m=self.model; z=np.array([1.,-2.,3.,-4.,5.,-6.])
        expected=z*m.A + np.maximum(z,0) @ m.W.T + m.h
        np.testing.assert_allclose(m.step(z),expected)
        np.testing.assert_allclose(m.observe(z), np.maximum(z,0)@m.B.T)

    def test_training_objective_is_finite_and_positive(self):
        m=self.model
        s=m.objective(self.x[:self.train_end],m.latent_train_,m.process_var_)
        for k in ('observation_mse','dynamics_mse','map_data_objective'):
            self.assertTrue(np.isfinite(s[k])); self.assertGreaterEqual(s[k],0)

    def test_test_filter_causality(self):
        m=self.model
        x=self.x[self.train_end:].copy()
        initial=m.latent_train_[-1]
        left=m.filter(x,initial_state=initial)
        x[-1]=[100.,100.]
        right=m.filter(x,initial_state=initial)
        np.testing.assert_allclose(left[:-1],right[:-1],rtol=0,atol=1e-12)
        self.assertFalse(np.allclose(left[-1],right[-1]))

    def test_forecast_and_free_rollout_shapes(self):
        m=self.model
        scores=m.forecast_mse(self.x,self.train_end, horizons=(1,5,10))
        self.assertEqual(set(scores),{1,5,10})
        self.assertTrue(np.isfinite(list(scores.values())).all())
        z,x=m.rollout(m.latent_train_[-1],25)
        self.assertEqual(z.shape,(25,6)); self.assertEqual(x.shape,(25,2))

    def test_save_load_and_encoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp)/'model.npz'; self.model.save(str(f))
            loaded=PiecewiseLinearSSM.load(str(f))
            np.testing.assert_allclose(loaded.W,self.model.W)
            np.testing.assert_allclose(loaded.encoder_,self.model.encoder_)
            np.testing.assert_allclose(loaded.encode(self.x[:3]),self.model.encode(self.x[:3]))
            np.testing.assert_allclose(loaded.step(self.model.latent_train_[-1]),
                                       self.model.step(self.model.latent_train_[-1]))

    def test_input_errors(self):
        with self.assertRaises(ValueError):
            PiecewiseLinearSSM(3).fit(self.x)
        with self.assertRaises(ValueError):
            PiecewiseLinearSSM(6).fit(np.full((10,2),np.nan))
        with self.assertRaises(ValueError):
            self.model.forecast_mse(self.x, len(self.x))
        with self.assertRaises(ValueError):
            self.model.rollout(np.zeros(2),10)

    def test_histogram_metrics(self):
        rng=np.random.default_rng(3)
        a=rng.normal(size=(300,2))
        same=occupancy_metrics(a,a)
        self.assertAlmostEqual(same['histogram_kl_raw'],0.,places=10)
        self.assertAlmostEqual(same['histogram_js'],0.,places=10)
        shifted=occupancy_metrics(a,a+4)
        self.assertGreater(shifted['histogram_kl_raw'],0.1)
        self.assertGreater(shifted['histogram_js'],0.01)


if __name__ == '__main__':
    unittest.main()

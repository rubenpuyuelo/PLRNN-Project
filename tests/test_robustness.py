"""Focused Phase-4 tests; self-contained and fast (no full retraining)."""
import unittest
from types import SimpleNamespace
import numpy as np

from src.evaluation.robustness import (evaluate_trajectory, finite_difference_spectrum,
    lyapunov_plrnn, spectrum_stats, summarise)
from src.training.run_robustness import _summary, benchmark_reference


class RobustnessTests(unittest.TestCase):
    def test_identical_trajectory_divergences_are_zero(self):
        rng=np.random.default_rng(1)
        x=rng.normal(size=(180, 2))
        z=evaluate_trajectory(x,x,dt=0.05,system='van_der_pol')
        for width in (0.5,1.0,1.5):
            self.assertAlmostEqual(z[f'js_width_{width}'],0.,places=10)
            self.assertAlmostEqual(z[f'kl_raw_width_{width}'],0.,places=10)
        self.assertFalse(z['near_fixed_point'])

    def test_fixed_point_and_spectral_guard(self):
        t=np.arange(600)*0.05
        x=np.column_stack((np.sin(t),np.cos(t)))
        collapse=np.zeros_like(x)
        sc=evaluate_trajectory(x,collapse,dt=0.05,system='van_der_pol')
        self.assertTrue(sc['near_fixed_point'])
        self.assertFalse(sc['sustained_oscillation'])
        self.assertIsNone(sc['generated_peak_frequency'])
        self.assertGreater(sc['js_width_1.0'],.1)
        same=spectrum_stats(x,x,0.05)
        self.assertAlmostEqual(same['normalized_spectrum_overlap'],1.,places=9)
        self.assertAlmostEqual(same['relative_peak_error'],0.,places=9)

    def test_known_plrnn_jacobian(self):
        # z' = 0.9 z (no W), LLE should be exactly log(0.9)/dt.
        m=SimpleNamespace(n_latent=3,A=np.full(3,.9),W=np.zeros((3,3)))
        z=np.ones((150,3))
        self.assertAlmostEqual(lyapunov_plrnn(m,z,.05),np.log(.9)/.05,places=10)
        m.A=np.full(3,1.1)
        self.assertGreater(lyapunov_plrnn(m,z,.05),0.)

    def test_summary_counts_all_attempts(self):
        r=[{'status':'ok','scores':{'js_width_1.0':.2,'kl_raw_width_1.0':.3,
             'mean_std_ratio':1.,'near_fixed_point':False},'finite_time_lyapunov':-.1},
           {'status':'unstable','reason':'diverged'}]
        s=_summary(r)
        self.assertEqual(s['n_attempts'],2)
        self.assertEqual(s['n_stable'],1)
        self.assertEqual(s['stable_fraction'],.5)
        self.assertEqual(s['histogram_js_width_1.0']['median'],.2)
        self.assertEqual(summarise([3.,float('nan'),1.])['n'],2)

    def test_benchmark_independent_reference_train_scaled(self):
        # Generated reference uses supplied TRAIN scaling; no test data used.
        ds={'train_mean':np.array([2.,-1.]),'train_std':np.array([.5,2.])}
        a,b,raw=benchmark_reference('van_der_pol',60,ds)
        self.assertEqual(a.shape,(60,2))
        self.assertEqual(b.shape,(60,2))
        np.testing.assert_allclose(a*ds['train_std']+ds['train_mean'],raw)
        self.assertFalse(np.allclose(a,b))

    def test_metric_validation(self):
        with self.assertRaises(ValueError):
            finite_difference_spectrum(np.ones(4),.05)
        with self.assertRaises(ValueError):
            evaluate_trajectory(np.zeros((35,2)),np.zeros((34,2)),dt=.05,system='lorenz')
        with self.assertRaises(ValueError):
            lyapunov_plrnn(SimpleNamespace(n_latent=2),np.ones((4,2)),.02)

if __name__=='__main__':unittest.main()

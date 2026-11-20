"""Phase 5 train-only multistep refiner and stability-aware selection tests."""
from __future__ import annotations
from copy import deepcopy
import unittest
import numpy as np
from src.models.plrnn import PiecewiseLinearSSM
from src.training.multistep import RefinementConfig, multistep_loss, refine_transitions
from src.training.run_phase5 import interpolate_models, train_occupancy_score, choose_interpolation


class MultiStepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        t=np.arange(86)*.09
        cls.x=np.column_stack((np.sin(t),np.cos(t)))
        cls.model=PiecewiseLinearSSM(n_latent=6,seed=3,ridge_dyn=8.)
        cls.model.fit(cls.x,schedule=(1.,.2),outer_iters=2,latent_steps=15)

    def test_refiner_preserves_structure_and_inputs(self):
        before=deepcopy(self.model)
        refined,logs=refine_transitions(before,self.x,RefinementConfig(
            horizon=4,iterations=35,batch_size=16,learning_rate=.003,seed=3))
        self.assertEqual(len(logs)>2,True)
        self.assertEqual(refined.W.shape,(6,6))
        np.testing.assert_array_equal(np.diag(refined.W),np.zeros(6))
        np.testing.assert_allclose(refined.B,before.B)
        np.testing.assert_allclose(refined.encoder_,before.encoder_)
        np.testing.assert_allclose(refined.latent_train_,before.latent_train_)
        np.testing.assert_allclose(before.A,self.model.A)
        np.testing.assert_allclose(before.W,self.model.W)
        self.assertTrue(np.isfinite(refined.W).all())

    def test_deterministic_refinement(self):
        cfg=RefinementConfig(horizon=3,iterations=15,batch_size=12,seed=11)
        a,_=refine_transitions(self.model,self.x,cfg)
        b,_=refine_transitions(self.model,self.x,cfg)
        for name in ('A','W','h'):
            np.testing.assert_array_equal(getattr(a,name),getattr(b,name))

    def test_interpolation_endpoints_are_exact(self):
        a,_=refine_transitions(self.model,self.x,
                                RefinementConfig(horizon=3,iterations=10,batch_size=8))
        lo=interpolate_models(self.model,a,0.)
        hi=interpolate_models(self.model,a,1.)
        np.testing.assert_allclose(lo.W,self.model.W)
        np.testing.assert_allclose(hi.W,a.W)
        self.assertIsNot(lo.W,self.model.W)
        with self.assertRaises(ValueError):interpolate_models(self.model,a,1.1)

    def test_invalid_data_is_rejected(self):
        with self.assertRaises(ValueError):
            refine_transitions(self.model,self.x[:-1])
        with self.assertRaises(ValueError):
            refine_transitions(self.model,self.x,RefinementConfig(horizon=len(self.x)))
        with self.assertRaises(ValueError):
            multistep_loss(self.model,self.x,horizon=0)
        with self.assertRaises(ValueError):
            multistep_loss(self.model,self.x[:-1],horizon=1)

    def test_train_only_metric_marks_unstable_as_rejected(self):
        bad=deepcopy(self.model)
        bad.A=np.ones(6)*3.0
        bad.W=np.zeros((6,6))
        bad.h=np.ones(6)*.5
        r=train_occupancy_score(bad,self.x,init_indices=(10,30),burn=5,n_eval=60)
        self.assertFalse(r['accepted'])
        self.assertTrue(any(x['status']=='unstable' for x in r['details']))

    def test_selection_rejects_wrong_alpha_grid(self):
        with self.assertRaises(ValueError):
            choose_interpolation(self.model,self.model,self.x,alphas=(.2,.4),
                init_indices=(25,),burn=5,n_eval=60)

    def test_train_only_selection_and_return_shape(self):
        refined,_=refine_transitions(self.model,self.x,
                    RefinementConfig(horizon=3,iterations=12,batch_size=12))
        best,detail=choose_interpolation(self.model,refined,self.x,
                 alphas=(0.,.2,.5),init_indices=(10,60),burn=10,n_eval=70)
        self.assertEqual(len(detail['candidates']),3)
        self.assertIn(detail['selected_alpha'],(0.,.2,.5))
        self.assertEqual(best.B.shape,self.model.B.shape)
        self.assertEqual(detail['reference_source'],'noisy training observations only')
        self.assertTrue(np.isfinite(best.A).all())


if __name__=='__main__':unittest.main()

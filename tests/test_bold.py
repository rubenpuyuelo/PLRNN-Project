"""Scientific sanity checks for the synthetic BOLD observation extension."""
import unittest
import numpy as np

from src.data.bold import canonical_hrf, causal_convolution, build_synthetic_bold
from src.models.bold_ssm import torch_causal_convolution, PiecewiseLinearBoldSSM
from src.training.run_phase6 import fit_linear_decoder, oracle_observation_experiment


class TestBold(unittest.TestCase):
    def test_hrf_causal_shape_normalization(self):
        h=canonical_hrf()
        self.assertTrue(np.isfinite(h).all())
        self.assertAlmostEqual(float(h.sum()),1.,places=10)
        self.assertEqual(h[0],0.)
        self.assertGreater(h.max(),0.)
        self.assertLess(h.min(),0.)

    def test_impulse_response_exact(self):
        h=canonical_hrf()
        impulse=np.zeros((90,2));impulse[0]=[1,2]
        out=causal_convolution(impulse,h)
        np.testing.assert_allclose(out[:len(h),0],h,atol=1e-12)
        np.testing.assert_allclose(out[:,1],2*out[:,0],atol=1e-12)

    def test_no_future_information(self):
        rng=np.random.default_rng(1)
        x=rng.normal(size=(150,3));h=canonical_hrf()
        y=causal_convolution(x,h)
        xx=x.copy();xx[100:]+=10000
        np.testing.assert_array_equal(y[:100],causal_convolution(xx,h)[:100])

    def test_delta_matches_instantaneous(self):
        x=np.arange(21.).reshape(7,3)
        np.testing.assert_allclose(causal_convolution(x,np.array([1.])),x)

    def test_torch_gradient_and_numpy_agree(self):
        import torch
        rng=np.random.default_rng(123)
        x=rng.normal(size=(64,3))
        h=canonical_hrf()
        z=torch.tensor(x,dtype=torch.float64,requires_grad=True)
        out=torch_causal_convolution(z,torch.as_tensor(h,dtype=torch.float64))
        np.testing.assert_allclose(out.detach().numpy(),causal_convolution(x,h),atol=1e-11)
        out.square().sum().backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_reproducible_and_train_only_normalized(self):
        x=np.random.default_rng(4).normal(size=(180,2))
        a=build_synthetic_bold(x,120,seed=99)
        b=build_synthetic_bold(x,120,seed=99)
        np.testing.assert_array_equal(a['bold_observed'],b['bold_observed'])
        np.testing.assert_allclose(a['bold_observed'][:120].mean(0),0.,atol=1e-7)
        np.testing.assert_allclose(a['bold_observed'][:120].std(0),1.,atol=1e-7)

    def test_validation_errors(self):
        with self.assertRaises(ValueError): canonical_hrf(tr=0)
        with self.assertRaises(ValueError): causal_convolution(np.ones((10,2)),np.array([]))
        with self.assertRaises(ValueError): build_synthetic_bold(np.ones((120,2)),70)
        with self.assertRaises(ValueError): PiecewiseLinearBoldSSM(n_latent=4,kernel=np.zeros(4))

    def test_oracle_observation_difference(self):
        n=600;t=np.arange(n)
        z=np.column_stack([np.sin(.15*t),np.cos(.09*t)])
        ds=build_synthetic_bold(z,420,noise_std=0.03)
        out=oracle_observation_experiment(ds)
        self.assertLess(out['mse']['hrf_aware']['test_clean'],
                        out['mse']['instantaneous']['test_clean'])

    def test_decoder_doesnt_learn_test_target(self):
        rng=np.random.default_rng(1)
        f=rng.normal(size=(150,2));y=np.column_stack([f[:,0],f[:,1]])
        pred,beta=fit_linear_decoder(f,y,100)
        yy=y.copy();yy[100:]+=1234
        pred2,beta2=fit_linear_decoder(f,yy,100)
        np.testing.assert_array_equal(beta,beta2)
        np.testing.assert_array_equal(pred,pred2)

    def test_model_map_and_prefix_rollout(self):
        h=np.array([0.2,0.6,0.2])
        model=PiecewiseLinearBoldSSM(n_latent=4,kernel=h,seed=1)
        model.A=np.full(4,.85);model.W=np.zeros((4,4));model.h=np.zeros(4)
        model.B=np.array([[1.,0.,0.,0.], [0.,1.,0.,0.]])
        seed=np.array([1.,.5,0.,0.]);prev=np.tile(seed,(10,1))
        z,obs=model.rollout_bold(seed,7,prefix=prev)
        expected=model.observe(np.vstack([prev[-2:],z]))[-7:]
        np.testing.assert_allclose(obs,expected)
        self.assertEqual(obs.shape,(7,2))

    def test_awaits_past_context(self):
        h=np.array([0.,1.])
        model=PiecewiseLinearBoldSSM(n_latent=2,kernel=h)
        model.A=np.array([1.,1.]);model.W=np.zeros((2,2));model.h=np.zeros(2)
        model.B=np.eye(2)
        _, y=model.rollout_bold(np.array([1.,2.]),3,prefix=np.array([[7.,8.]]))
        np.testing.assert_array_equal(y[0],np.array([7.,8.]))


    def test_small_map_fit_and_save_roundtrip(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        t=np.arange(130)
        neural=np.column_stack([np.sin(.07*t),np.cos(.07*t)])
        ds=build_synthetic_bold(neural,90,noise_std=0.02)
        model=PiecewiseLinearBoldSSM(n_latent=4,kernel=ds['hrf'],seed=9)
        model.fit(ds['bold_observed'][:90],schedule=(1.0,),outer_iters=1,
                  latent_steps=3,lr=.01)
        self.assertEqual(model.latent_train_.shape,(90,4))
        self.assertTrue(np.isfinite(model.observe(model.latent_train_)).all())
        self.assertTrue(np.allclose(np.diag(model.W),0.))
        with TemporaryDirectory() as temp:
            path=str(Path(temp)/'model.npz')
            model.save(path)
            restored=PiecewiseLinearBoldSSM.load(path)
            np.testing.assert_allclose(restored.observe(model.latent_train_),
                                       model.observe(model.latent_train_))

if __name__=='__main__': unittest.main()

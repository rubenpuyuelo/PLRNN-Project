"""Phase 4: multi-seed dynamical evaluation and noise sensitivity.

Run from the project root:
    python -m src.training.run_robustness
    python -m src.training.run_robustness --quick
    python -m src.training.run_robustness --skip-noise

All inputs from earlier phases are READ-ONLY. Results live in results/phase4/.
No test data is used in fitting, model selection, or free-run initialization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.data.lorenz import simulate_lorenz
from src.data.van_der_pol import simulate_van_der_pol
from src.data.preprocessing import load_dataset, make_dataset
from src.evaluation.robustness import (evaluate_trajectory, lyapunov_lorenz_approx,
    lyapunov_plrnn, summarise)
from src.models.lds import LinearGaussianSSM
from src.models.plrnn import PiecewiseLinearSSM

ROOT = Path(__file__).resolve().parents[2]
SYS = {
    'van_der_pol': {'m': 8, 'dt': 0.05, 'initial': (1.75, -0.1)},
    'lorenz': {'m': 12, 'dt': 0.02, 'initial': (-5.0, 6.0, 25.0)},
}


def benchmark_reference(system: str, n_eval: int, ds: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate unseen clean trajectory, then scale with TRAIN-ONLY Phase-1 stats.

    Returns standardized reference A, independent reference B, and physical
    Lorenz states for Lyapunov approximation. No Phase-1 held-out samples used.
    """
    config = SYS[system]
    gen = simulate_van_der_pol if system == 'van_der_pol' else simulate_lorenz
    _, physical = gen(n_steps=2*n_eval, dt=config['dt'], initial_state=config['initial'])
    mean, std = np.asarray(ds['train_mean']), np.asarray(ds['train_std'])
    standardized = (physical - mean) / std
    return standardized[:n_eval], standardized[n_eval:], physical[:n_eval]


def _summary(records: list[dict]) -> dict:
    ok = [r for r in records if r['status'] == 'ok']
    return {
        'n_attempts': len(records), 'n_stable': len(ok),
        'stable_fraction': len(ok)/len(records) if records else None,
        'histogram_js_width_1.0': summarise([r['scores']['js_width_1.0'] for r in ok]),
        'histogram_kl_raw_width_1.0': summarise([r['scores']['kl_raw_width_1.0'] for r in ok]),
        'mean_std_ratio': summarise([r['scores']['mean_std_ratio'] for r in ok]),
        'near_fixed_point_count': sum(bool(r['scores']['near_fixed_point']) for r in ok),
        'finite_time_lyapunov': summarise([r['finite_time_lyapunov'] for r in ok
                                            if r.get('finite_time_lyapunov') is not None]),
    }


def _score_rollouts(model, *, model_type: str, system: str, ref: np.ndarray,
                    burn: int, train_states: np.ndarray, train_ends: tuple[int, ...],
                    seed: int) -> tuple[list[dict], np.ndarray | None]:
    """Generate free trajectories from train-only inferred states, not test states."""
    results, snapshot = [], None
    dt = SYS[system]['dt']
    for idx, end in enumerate(train_ends):
        initial = train_states[end].copy()
        try:
            z, x = model.rollout(initial, burn + len(ref), sample_process_noise=False,
                                 **({'max_abs': 1e4} if model_type == 'plrnn' else {}))
            z, x = z[burn:], x[burn:]
            # LDS has no stepwise numerical runaway guard in Phase 2.
            if not np.isfinite(x).all() or np.max(np.abs(x)) > 1e4:
                raise FloatingPointError('unbounded/invalid generated observations')
            scores = evaluate_trajectory(ref, x, dt=dt, system=system)
            le = lyapunov_plrnn(model, z, dt) if model_type == 'plrnn' else (
                float(np.log(model.spectral_radius) / dt))
            r = {'status': 'ok', 'init_index_train': int(end), 'scores': scores,
                 'finite_time_lyapunov': le}
            if idx == 0:
                snapshot = x
        except (FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError) as ex:
            r = {'status': 'unstable', 'init_index_train': int(end),
                 'reason': f'{type(ex).__name__}: {str(ex)[:180]}'}
        results.append(r)
    return results, snapshot


def _train_plrnn(obs_train: np.ndarray, system: str, seed: int,
                 *, quick: bool) -> PiecewiseLinearSSM:
    model = PiecewiseLinearSSM(SYS[system]['m'], seed=seed, ridge_dyn=8.0)
    model.fit(obs_train, outer_iters=2 if quick else 5,
              latent_steps=18 if quick else 60, verbose=False)
    return model


def plot_rollouts(system: str, ref: np.ndarray, plrnn: np.ndarray | None,
                  lds: np.ndarray | None, dest: Path) -> None:
    fig = plt.figure(figsize=(12.5, 4.15))
    n = ref.shape[1]
    plots = [(ref, 'Independent ground truth'),
             (plrnn, 'PLRNN seed 42: autonomous'),
             (lds, 'Dimension-matched LDS: autonomous')]
    bounds = [(float(ref[:,k].min()), float(ref[:,k].max())) for k in range(n)]
    for pos, (samples, title) in enumerate(plots, 1):
        ax = fig.add_subplot(1, 3, pos, projection='3d' if n == 3 else None)
        if samples is not None:
            if n == 3:
                ax.plot(samples[:,0], samples[:,1], samples[:,2], lw=0.65)
                ax.set_zlabel('x3', fontsize=8)
            else:
                ax.plot(samples[:,0], samples[:,1], lw=0.65)
        else:
            ax.text(0.05,0.5,'Unstable rollout',transform=ax.transAxes)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('x1'); ax.set_ylabel('x2')
        for k,(lo,hi) in enumerate(bounds):
            pad = max(0.2,0.15*(hi-lo))
            if k == 0: ax.set_xlim(lo-pad,hi+pad)
            if k == 1: ax.set_ylim(lo-pad,hi+pad)
            if k == 2: ax.set_zlim(lo-pad,hi+pad)
    fig.suptitle(f'{system.replace("_", " ").title()}: independent {len(ref)}-step dynamics, same axes')
    fig.tight_layout()
    fig.savefig(dest, dpi=145, bbox_inches='tight')
    plt.close(fig)


def plot_restarts(summaries: dict, dest: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.7, 4.65))
    keys = list(summaries)
    values, labels = [], []
    for system in keys:
        rec = summaries[system]
        for name, model in [('plrnn', 'PLRNN'), ('lds_matched', 'LDS (matched M)')]:
            vals = [r['scores']['js_width_1.0'] for r in rec[name]['runs'] if r['status']=='ok']
            values.append(vals)
            labels.append(f'{system.replace("_", " ")}\n{model}')
    positions = np.arange(1, len(values)+1)
    rng = np.random.default_rng(73)
    for i,(xs,lab) in enumerate(zip(values, labels)):
        if len(xs):
            ax.scatter(np.full(len(xs),positions[i]) + rng.uniform(-.15,.15,len(xs)),xs,alpha=.8,label=lab)
            ax.hlines(np.median(xs),positions[i]-.22,positions[i]+.22,lw=2.5)
    ax.set_xticks(positions, labels)
    ax.set_ylabel('Histogram JSD, bin width 1 (lower = closer)')
    ax.set_title('Every stable autonomous run is shown; unstable runs counted separately')
    ax.set_ylim(bottom=0)
    ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    fig.savefig(dest,dpi=145,bbox_inches='tight')
    plt.close(fig)


def plot_noise(noise_rows: list[dict], dest: Path) -> None:
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,system in zip(axs,SYS):
        rows=[r for r in noise_rows if r['system']==system]
        noise=[r['noise_std'] for r in rows]
        js=[r['js_width_1.0'] for r in rows]
        pred=[r['forecast_mse_h5'] for r in rows]
        ax.plot(noise,js,'o-',label='Free-run JSD')
        ax.set_xlabel('Gaussian observation noise / clean train std')
        ax.set_ylabel('Long-run JSD (lower is better)')
        ax.set_title(system.replace('_',' ').title())
        ax.set_xticks(noise)
        ax.grid(alpha=.22)
        b=ax.twinx()
        b.plot(noise,pred,'s--',color='gray',label='5-step test MSE')
        b.set_ylabel('5-step test MSE (lower is better)',color='gray')
    fig.suptitle('Exploratory noise sensitivity: one training seed per noise level')
    fig.tight_layout()
    fig.savefig(dest,dpi=145,bbox_inches='tight')
    plt.close(fig)


def run_phase4(*, quick: bool=False, skip_noise: bool=False,
               root: Path=ROOT, seeds: tuple[int,...]=(42,7,21),
               n_eval: int|None=None, burn: int|None=None,
               n_restarts: int=3) -> dict:
    root=Path(root)
    out=root/'results'/('phase4_quick' if quick else 'phase4')
    figdir, metdir, modeldir=(out/sub for sub in ('figures','metrics','models'))
    for d in (figdir,metdir,modeldir): d.mkdir(parents=True,exist_ok=True)
    n_eval = n_eval if n_eval is not None else (400 if quick else 2400)
    burn = burn if burn is not None else (80 if quick else 400)
    if n_eval < 50 or burn < 0 or n_restarts<1 or not seeds:
        raise ValueError('invalid phase-4 experiment settings')
    results, noise_rows = {}, []
    for system, conf in SYS.items():
        print(f'\n=== PHASE 4: {system} ===',flush=True)
        data_path=root/'data'/'synthetic'/f'{system}_seed42.npz'
        phase3_path=root/'results'/'metrics'/f'plrnn_{system}_model.npz'
        for p in (data_path,phase3_path):
            if not p.exists():
                raise FileNotFoundError(f'Need previous phases: {p}')
        ds=load_dataset(data_path)
        x=np.asarray(ds['x_obs'],dtype=float)
        train_end=int(ds['train_end'])
        ref, ref_other, physical = benchmark_reference(system,n_eval,ds)
        ref_self=evaluate_trajectory(ref,ref_other,dt=conf['dt'],system=system)
        init_indices=tuple(np.linspace(0,train_end-1,n_restarts+1,dtype=int)[1:][::-1])
        records, display_plrnn = [], None
        for seed in seeds:
            if seed==42 and not quick:
                model=PiecewiseLinearSSM.load(str(phase3_path))
                # Enforce exact model/data split from Phase 3.
                if len(model.latent_train_)!=train_end or model.n_latent!=conf['m']:
                    raise ValueError(f'Phase-3 model and dataset incompatible: {phase3_path}')
                origin='phase3_saved_checkpoint'
            else:
                model=_train_plrnn(x[:train_end],system,seed,quick=quick)
                origin='trained_in_phase4'
                model.save(str(modeldir/f'plrnn_{system}_seed{seed}.npz'))
            run_records, snapshot=_score_rollouts(
                model,model_type='plrnn',system=system,ref=ref,burn=burn,
                train_states=model.latent_train_,train_ends=init_indices,seed=seed)
            if seed==42: display_plrnn=snapshot
            # Causal held-out prediction; never used for picking a seed.
            h5=float(model.forecast_mse(x,train_end,horizons=(5,))[5])
            for r in run_records:r.update(model='plrnn',seed=seed,source=origin,forecast_mse_h5=h5)
            records.extend(run_records)
            good=sum(r['status']=='ok' for r in run_records)
            print(f'PLRNN seed={seed} | stable={good}/{len(run_records)} | forecast MSE h5={h5:.4f}',flush=True)
        # LDS uses EXACTLY same latent dimension M; still different noise/inference/parameterization.
        lds=LinearGaussianSSM(n_latent=conf['m'],seed=42)
        lds.fit(x[:train_end],n_iter=8 if quick else 25,verbose=False)
        latent_train=lds.filter(x[:train_end]).filtered_mean
        lds_records, display_lds=_score_rollouts(
            lds,model_type='lds',system=system,ref=ref,burn=burn,
            train_states=latent_train,train_ends=init_indices,seed=42)
        for r in lds_records:r.update(model='lds_matched',seed=42)
        lds_h5=float(lds.forecast_mse(x,train_end,horizons=(5,))[5])
        print(f'Matched LDS M={conf["m"]} | stable={sum(r["status"]=="ok" for r in lds_records)}/{len(lds_records)} | MSE h5={lds_h5:.4f}',flush=True)
        plot_rollouts(system,ref,display_plrnn,display_lds,figdir/f'phase4_{system}_long_rollout.png')
        reference_lyap = lyapunov_lorenz_approx(physical,conf['dt']) if system=='lorenz' else None
        results[system]={
            'system':system,'latent_dimension':conf['m'],'n_observations':int(x.shape[1]),
            'train_end':train_end,'n_test':len(x)-train_end,
            'seeds':list(seeds),'n_initializations_per_seed':n_restarts,
            'n_longrun_evaluation_samples':n_eval,'n_burn_in_steps':burn,
            'reference_vs_reference':ref_self,
            'reference_finite_time_lyapunov':reference_lyap,
            'plrnn':{'runs':records,'summary':_summary(records)},
            'lds_matched':{'runs':lds_records,'summary':_summary(lds_records),
                           'forecast_mse_h5':lds_h5,
                           'spectral_radius':lds.spectral_radius},
            'interpretation_notes':('Stable here means finite and bounded, NOT correct attractor. '
             'PLRNN/linear models have equal latent M, NOT identical parameter count or inference. '
             'Raw histogram KL and JSD are coordinate/bin dependent and NOT normalized KLx in paper. '
             'PLRNN is a joint-MAP approximation rather than exact Laplace-EM. '
             'Seed 42 reuses saved Phase-3 model in full mode; other seeds use the same fitting budget. '
             'Independent reference never used for fitting or selecting the displayed seed.')
        }
        (metdir/f'phase4_{system}_runs.json').write_text(json.dumps(results[system],indent=2),encoding='utf-8')
        if not skip_noise:
            # Ablation uses the same underlying physical train/test trajectory.
            # Only observation noise varies. Fitting/hyperparameters and seed held constant.
            physical_clean=np.asarray(ds['x_true'],dtype=float)*ds['train_std']+ds['train_mean']
            time=np.asarray(ds['time'])
            for noise in (0.0,0.10,0.30):
                noisy=make_dataset(time,physical_clean,observation_noise_std=noise,
                                   train_fraction=train_end/len(x),seed=42)
                obs=np.asarray(noisy['x_obs'],dtype=float)
                # ref was standardized using the Phase-1 scaling; rescale to this noise condition.
                modified_ref=(ref*ds['train_std']+ds['train_mean']-noisy['train_mean'])/noisy['train_std']
                m=_train_plrnn(obs[:train_end],system,seed=21,quick=quick)
                run, _=_score_rollouts(m,model_type='plrnn',system=system,
                          ref=modified_ref,burn=burn,train_states=m.latent_train_,
                          train_ends=(train_end-1,),seed=21)
                status=run[0]['status']
                js=run[0]['scores']['js_width_1.0'] if status=='ok' else None
                mse5=float(m.forecast_mse(obs,train_end,horizons=(5,))[5])
                noise_rows.append({'system':system,'noise_std':noise,'seed':21,
                         'status':status,'js_width_1.0':js,'forecast_mse_h5':mse5})
                print(f'Noise {noise}: status={status} JSD={js} MSE_h5={mse5:.4f}',flush=True)
    plot_restarts(results,figdir/'phase4_multiseed_js.png')
    if not skip_noise:
        plot_noise(noise_rows,figdir/'phase4_noise_ablation.png')
    (metdir/'phase4_noise_ablation.json').write_text(json.dumps(noise_rows,indent=2),encoding='utf-8')
    summary={
        'experiment':'Phase 4: independent free-running reconstruction, seed and noise sensitivity',
        'configuration':{'quick':quick,'n_eval':n_eval,'burn':burn,'seeds':list(seeds),
                         'restarts_per_seed':n_restarts,'noise_ablation_run':not skip_noise},
        'systems':{s:{'plrnn':d['plrnn']['summary'],
                      'lds_matched':d['lds_matched']['summary'],
                      'reference_self_js_width_1.0':d['reference_vs_reference']['js_width_1.0'],
                      'reference_lyapunov':d['reference_finite_time_lyapunov']}
                   for s,d in results.items()},
        'noise_ablation':noise_rows,
        'caveats':['Descriptive finite samples; no inferential statistical significance',
                   'Seed count includes multiple correlated initialization runs per fit',
                   'Raw histogram KL / JSD not paper normalized KL',
                   'Finite-time Lyapunov diagnostics depend on sampling and orbit',
                   'PLRNN is simplified joint MAP rather than original EM',
                   'Equal latent M but not matched number of parameters or inference algorithm']}
    (metdir/'phase4_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('\nPhase 4 results saved to:',out,flush=True)
    return summary


def main() -> None:
    p=argparse.ArgumentParser(description='Phase 4 long-run dynamics and robustness')
    p.add_argument('--quick',action='store_true',help='Smoke test only; short runs, reduced training')
    p.add_argument('--skip-noise',action='store_true',help='Skip observation-noise ablation')
    args=p.parse_args()
    run_phase4(quick=args.quick,skip_noise=args.skip_noise)


if __name__=='__main__':main()

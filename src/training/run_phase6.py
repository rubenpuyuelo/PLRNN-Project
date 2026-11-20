"""Phase 6: causal synthetic BOLD and exploratory HRF-aware PLRNN-MAP.

Run from repository root:
    python -m src.training.run_phase6
    python -m src.training.run_phase6 --quick

Outputs live under results/phase6/ (or phase6_quick/) and data/bold/.
Earlier phases' checkpoints and data are never overwritten. For the clean
oracle decoder control, the true benchmark series is explicitly available;
its results are *not* evidence of unsupervised latent inference from BOLD.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data.bold import build_synthetic_bold, causal_convolution, save_bold
from src.data.preprocessing import load_dataset
from src.evaluation.dynamics import occupancy_metrics
from src.models.bold_ssm import PiecewiseLinearBoldSSM

ROOT = Path(__file__).resolve().parents[2]


def fit_linear_decoder(features: np.ndarray, observations: np.ndarray,
                       train_end: int, ridge: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """Ridge OLS on TRAIN ONLY, including intercept; no access to test labels."""
    f = np.asarray(features, dtype=float)
    y = np.asarray(observations, dtype=float)
    if f.ndim != 2 or y.ndim != 2 or len(f) != len(y) or train_end < 2 or train_end >= len(f):
        raise ValueError('features/observations must be compatible [T,D] with split')
    t = np.column_stack([f[:train_end], np.ones(train_end)])
    pen = np.eye(t.shape[1]) * ridge
    pen[-1, -1] = 0.0
    beta = np.linalg.solve(t.T @ t + pen, t.T @ y[:train_end])
    pred = np.column_stack([f, np.ones(len(f))]) @ beta
    return pred, beta


def oracle_observation_experiment(ds: dict, *, ridge: float = 1e-4) -> dict:
    """Mechanistic decoder test with KNOWABLE original proxy states.

    Causal HRF features vs contemporaneous neural features, matched number of
    decoder coefficients. Held-out MSE is evaluated on *clean* BOLD to isolate
    observation model error; noisy score is also included for completeness.
    """
    z = ds['neural_proxy'].astype(float)
    y = ds['bold_observed'].astype(float)
    clean = ds['bold_clean'].astype(float)
    te = int(ds['train_end'])
    result = {'train_end': te, 'test_length': len(z)-te, 'mse': {}, 'predictions': {}}
    for name, feat in [('hrf_aware', causal_convolution(z, ds['hrf'])),
                       ('instantaneous', z)]:
        pred, _ = fit_linear_decoder(feat, y, te, ridge=ridge)
        result['predictions'][name] = pred
        result['mse'][name] = {
            'test_clean': float(np.mean((pred[te:] - clean[te:])**2)),
            'test_noisy': float(np.mean((pred[te:] - y[te:])**2)),
            'train_noisy': float(np.mean((pred[:te] - y[:te])**2)),
        }
    result['clean_noise_floor_test'] = float(np.mean((clean[te:] - y[te:])**2))
    return result


def plot_hrf(ds: dict, system: str, path: Path) -> None:
    k = ds['hrf']
    t = np.arange(len(k))*float(ds['tr_illustrative'])
    z = ds['neural_proxy']; c = ds['bold_clean']; y = ds['bold_observed']
    end = int(ds['train_end'])
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(t,k,lw=2)
    axs[0].axhline(0, color='gray',lw=0.7)
    axs[0].set_xlabel('Illustrative lag (s)')
    axs[0].set_ylabel('Normalized HRF response')
    axs[0].set_title('Causal double-gamma HRF')
    s = slice(end-35, min(end+110, len(z)))
    idx = np.arange(len(z))[s]
    axs[1].plot(idx,z[s,0],lw=1.4,label='Neural proxy (channel 1)')
    axs[1].plot(idx,c[s,0],lw=1.8,label='Clean synthetic BOLD')
    axs[1].plot(idx,y[s,0],lw=0.65,alpha=.55,label='Noisy synthetic BOLD')
    axs[1].axvline(end,ls='--',color='gray',label='Train/test boundary')
    axs[1].set_xlabel('Sample index (NOT original ODE seconds)')
    axs[1].set_title('Illustrative neural proxy vs synthetic BOLD')
    axs[1].legend(fontsize=7)
    for ax in axs: ax.grid(alpha=.25)
    fig.suptitle(f'{system.replace("_", " ").title()} — synthetic BOLD observation model')
    fig.tight_layout();fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)


def plot_oracle(ds: dict, outcome: dict, system: str, path: Path) -> None:
    end = int(ds['train_end']); idx = np.arange(end,min(end+155,len(ds['bold_clean'])))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for label, field in [('Clean BOLD','bold_clean'),('Noisy BOLD','bold_observed')]:
        axs[0].plot(idx, ds[field][idx,0], label=label, lw=1.8 if label.startswith('Clean') else .8,
                    alpha=1 if label.startswith('Clean') else .4)
    for name, title in [('hrf_aware','HRF-aware oracle decoder'),('instantaneous','Instantaneous oracle decoder')]:
        axs[0].plot(idx, outcome['predictions'][name][idx,0], lw=1.3, label=title)
    axs[0].set_title('Held-out BOLD (channel 1)')
    axs[0].set_xlabel('Sample index');axs[0].set_ylabel('BOLD (train-standardized)')
    axs[0].legend(fontsize=7);axs[0].grid(alpha=.25)
    names=['hrf_aware','instantaneous']
    scores=[outcome['mse'][n]['test_clean'] for n in names]
    axs[1].bar(['HRF-aware','Instantaneous'],scores)
    axs[1].set_ylabel('Test MSE vs clean synthetic BOLD')
    axs[1].set_title('Matched linear decoders; oracle states known')
    axs[1].grid(axis='y',alpha=.25)
    fig.suptitle(system.replace('_',' ').title() + ': decoder ablation (NOT latent inference)')
    fig.tight_layout();fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)


def plot_map(ds: dict, records: dict, system: str, path: Path) -> None:
    end=int(ds['train_end']); n=len(ds['bold_observed'])-end
    idx=np.arange(min(n,150))
    fig, axs=plt.subplots(1, 2, figsize=(12,4))
    axs[0].plot(idx,ds['bold_clean'][end:end+len(idx),0],lw=1.6,label='Clean synthetic BOLD')
    for name in ['hrf_aware','instantaneous']:
        rec=records[name]
        if rec['stable']:
            axs[0].plot(idx,rec['generated'][:len(idx),0],label=name,lw=1.0)
    axs[0].set_title('Free-running BOLD after train split (not phase-aligned)')
    axs[0].set_xlabel('Sample index since train end')
    axs[0].legend(fontsize=8);axs[0].grid(alpha=.25)
    positive_scores=[r['test_js'] for r in records.values() if r['stable']]
    ymax=max(positive_scores + [0.1])*1.35
    for i,name in enumerate(['hrf_aware','instantaneous']):
        r=records[name]
        if r['stable']:
            axs[1].bar(i,r['test_js'])
        else:
            axs[1].text(i,0.45*ymax,'UNSTABLE\n(no JSD)',ha='center',va='center',fontsize=10)
    axs[1].set_ylim(0,ymax)
    axs[1].set_xticks([0,1],['HRF-aware','Instantaneous'])
    axs[1].set_ylabel('Held-out BOLD histogram JSD (lower is better)')
    axs[1].set_title('Exploratory PLRNN-MAP, free-running')
    axs[1].grid(axis='y',alpha=.2)
    fig.suptitle(system.replace('_',' ').title()+ ': latent model (MAP approximation)')
    fig.tight_layout();fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)


def run_phase6(*, quick: bool=False, skip_map: bool=False, seed: int=42) -> dict:
    folder=ROOT/'results'/('phase6_quick' if quick else 'phase6')
    figs=folder/'figures'; metrics=folder/'metrics';models=folder/'models'
    for p in [figs,metrics,models,ROOT/'data'/'bold']:p.mkdir(parents=True,exist_ok=True)
    results={}
    for system,latent in [('van_der_pol',8),('lorenz',12)]:
        source=ROOT/'data'/'synthetic'/f'{system}_seed42.npz'
        if not source.exists():raise FileNotFoundError(f'Missing phase-1 dataset: {source}')
        source_data=load_dataset(source)
        te=int(source_data['train_end']); z=source_data['x_true'].astype(float)
        ds=build_synthetic_bold(z,te,tr=1.0,noise_std=0.05,seed=seed)
        datafile=ROOT/'data'/'bold'/f'{system}_synthetic_bold_seed{seed}.npz'
        save_bold(ds,datafile)
        oracle=oracle_observation_experiment(ds)
        plot_hrf(ds,system,figs/f'phase6_{system}_hrf.png')
        plot_oracle(ds,oracle,system,figs/f'phase6_{system}_oracle.png')
        row={'system':system,'kind':'synthetic BOLD from ODE benchmark (neural proxy; not fMRI)',
             'train_end':te,'test_length':len(z)-te,'seed':seed,
             'hrf_length':len(ds['hrf']),'tr_illustrative':1.,
             'oracle_decoder':oracle['mse'],'noise_floor_test':oracle['clean_noise_floor_test']}
        print(f'\n{system} — matched oracle decoders',flush=True)
        for name in ['hrf_aware','instantaneous']:
            print(f'  {name:>15}: clean test MSE = {oracle["mse"][name]["test_clean"]:.5f}',flush=True)
        if not skip_map:
            # Both variants have the same latent size, parameter constraints,
            # initialization seed, optimization stages and training data.
            model_recs={}
            for name,kernel in [('hrf_aware',ds['hrf']),('instantaneous',np.array([1.]))]:
                print(f'  training exploratory PLRNN-BOLD-MAP: {name}',flush=True)
                model=PiecewiseLinearBoldSSM(n_latent=latent,seed=seed,
                                              kernel=kernel,ridge_dyn=8.0,obs_var=0.10)
                model.fit(ds['bold_observed'][:te], schedule=(1.,.3) if quick else (1.,.3,.1,.03),
                          outer_iters=2 if quick else 3,latent_steps=15 if quick else 45,
                          lr=.04,verbose=False)
                model.save(str(models/f'phase6_{system}_{name}_model.npz'))
                score=model.objective(ds['bold_observed'][:te],model.latent_train_,model.process_var_)
                outcome={'train_mse':score['observation_mse'],'stable':True,
                         'test_js':None,'test_kl_raw':None,'test_mse_time':None}
                try:
                    # Train-context HRF filter is continuous across train/test.
                    _,generated=model.rollout_bold(model.latent_train_[-1],len(z)-te,
                                                    prefix=model.latent_train_)
                    dist=occupancy_metrics(ds['bold_clean'][te:],generated)
                    outcome['test_js']=dist['histogram_js']
                    outcome['test_kl_raw']=dist['histogram_kl_raw']
                    # Not a reliable generative metric under chaotic phase drift.
                    outcome['test_mse_time']=float(np.mean((ds['bold_clean'][te:]-generated)**2))
                    outcome['generated']=generated
                except FloatingPointError as exc:
                    outcome['stable']=False;outcome['instability_reason']=str(exc)
                    outcome['generated']=None
                print(f'    train MSE={score["observation_mse"]:.4f}  '
                      f'free JSD={outcome["test_js"]} stable={outcome["stable"]}',flush=True)
                model_recs[name]=outcome
            plot_map(ds,model_recs,system,figs/f'phase6_{system}_map.png')
            # Numpy arrays excluded from JSON; raw generated arrays not needed.
            row['latent_map']={name:{k:v for k,v in rec.items() if k!='generated'}
                               for name,rec in model_recs.items()}
        results[system]=row
        (metrics/f'phase6_{system}.json').write_text(json.dumps(row,indent=2),encoding='utf-8')
    (metrics/'phase6_summary.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print('\nSaved Phase-6 results to',folder,flush=True)
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Synthetic BOLD model observation experiment')
    parser.add_argument('--quick',action='store_true',help='short exploratory MAP run, separate output folder')
    parser.add_argument('--skip-map',action='store_true',help='oracle decoders only; no model training')
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    run_phase6(quick=args.quick,skip_map=args.skip_map,seed=args.seed)

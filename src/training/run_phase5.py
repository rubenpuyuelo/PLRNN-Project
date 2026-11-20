"""Phase 5: train-only attractor-aware selection of multistep-refined PLRNNs.

Run: python -m src.training.run_phase5 [--quick]

No test observations or independent benchmark references are used to select
interpolation coefficients. All previous checkpoints are loaded READ-ONLY.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.data.preprocessing import load_dataset
from src.evaluation.dynamics import occupancy_metrics
from src.evaluation.robustness import evaluate_trajectory, summarise
from src.models.plrnn import PiecewiseLinearSSM
from src.training.multistep import RefinementConfig, multistep_loss, refine_transitions
from src.training.run_robustness import SYS, benchmark_reference

ROOT=Path(__file__).resolve().parents[2]
SEEDS=(42,7,21)
ALPHAS=(0.,.1,.2,.3,.4,.5,.6,.75,.9,1.)


def interpolate_models(baseline: PiecewiseLinearSSM,
                       proposal: PiecewiseLinearSSM, alpha: float) -> PiecewiseLinearSSM:
    """Interpolate ONLY transition coefficients; never mutate baselines."""
    from copy import deepcopy
    if not 0 <= alpha <= 1 or baseline.n_latent!=proposal.n_latent:
        raise ValueError('invalid alpha / model dimensions')
    result=deepcopy(baseline)
    for name in ('A','W','h'):
        setattr(result,name,(1.-alpha)*getattr(baseline,name) + alpha*getattr(proposal,name))
    np.fill_diagonal(result.W,0.)
    return result


def independent_rollouts(model: PiecewiseLinearSSM, ref: np.ndarray, *,
                         init_indices: tuple[int,...], burn: int,
                         system: str) -> tuple[list[dict],np.ndarray|None]:
    """Compare generated dynamics to an INDEPENDENT reference; evaluation only."""
    attempts=[]; picture=None
    for index in init_indices:
        try:
            z, x = model.rollout(model.latent_train_[index], burn+len(ref),max_abs=1e4)
            x=x[burn:]
            scores=evaluate_trajectory(ref,x,dt=SYS[system]['dt'],system=system)
            attempts.append({'init_index_train':index,'status':'ok',
                             'js_width_1.0':scores['js_width_1.0'],
                             'near_fixed_point':scores['near_fixed_point'],
                             'mean_std_ratio':scores['mean_std_ratio']})
            if picture is None:picture=x
        except (FloatingPointError,ValueError,OverflowError) as exc:
            attempts.append({'init_index_train':index,'status':'unstable',
                             'reason':str(exc)[:160]})
    return attempts,picture


def train_occupancy_score(model: PiecewiseLinearSSM, train_x: np.ndarray,
                          *, init_indices: tuple[int,...],burn: int,n_eval: int
                          ) -> dict:
    """Model-selection score uses ONLY training observations and states.

    Unstable autonomous rolls are rejected instead of omitted from medians.
    """
    scores=[]; details=[]
    for index in init_indices:
        try:
            _,gen=model.rollout(model.latent_train_[index],burn+n_eval,max_abs=1e4)
            gen=gen[burn:]
            j=occupancy_metrics(train_x,gen,width=1.)['histogram_js']
            scores.append(float(j))
            details.append({'index':index,'status':'ok','train_js':float(j)})
        except (ValueError,OverflowError,FloatingPointError) as exc:
            details.append({'index':index,'status':'unstable','reason':str(exc)[:160]})
    valid=len(scores)==len(init_indices)
    return {'accepted':valid,'train_js_median':float(np.median(scores)) if valid else None,
            'details':details}


def choose_interpolation(baseline: PiecewiseLinearSSM,
                         proposed: PiecewiseLinearSSM, train_obs: np.ndarray,
                         *,alphas:tuple[float,...]=ALPHAS,
                         init_indices:tuple[int,...],burn:int,n_eval:int) -> tuple[PiecewiseLinearSSM,dict]:
    """Train-only, stable and low-JSD selection (never uses heldout data)."""
    if 0. not in alphas or any((a<0 or a>1) for a in alphas):
        raise ValueError('alphas must include baseline alpha=0 and lie in [0,1]')
    trials=[];accepted=[]
    for alpha in alphas:
        model=interpolate_models(baseline,proposed,float(alpha))
        score=train_occupancy_score(model,train_obs,init_indices=init_indices,burn=burn,n_eval=n_eval)
        row={'alpha':float(alpha),**score}
        trials.append(row)
        if score['accepted']:
            accepted.append((score['train_js_median'],float(alpha)))
    if accepted:
        # Minimize *train* occupancy metric, deterministic tie-break toward baseline.
        score,chosen_alpha=min(accepted,key=lambda pair:(pair[0],pair[1]))
        status='selected_from_training_occupancy'
    else:
        # We do not claim success if even the baseline diverges.
        score,chosen_alpha=None,0.
        status='no_stable_candidate_fallback_original'
    chosen=interpolate_models(baseline,proposed,chosen_alpha)
    return chosen, {'selected_alpha':chosen_alpha,'selection_status':status,
                    'selected_train_js':score,'candidates':trials,
                    'reference_source':'noisy training observations only'}


def plot_summary(records:list[dict],figure_dir:Path) -> None:
    fig, axs=plt.subplots(1,2,figsize=(11,4))
    for ax,system in zip(axs,SYS):
        rows=[r for r in records if r['system']==system]
        for i,r in enumerate(rows):
            a=r['before_independent_js_median'];b=r['after_independent_js_median']
            if a is None or b is None:continue
            ax.plot([0,1],[a,b], 'o-',alpha=.8,label=f"Seed {r['seed']}" )
        ax.set_xticks([0,1],['Original MAP','Train-only selected\nmultistep'])
        ax.set_title(system.replace('_',' ').title())
        ax.set_ylabel('Independent reference JSD (bin=1)')
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Long-run attractor similarity — evaluation references not used in selection')
    fig.tight_layout()
    fig.savefig(figure_dir/'phase5_independent_js.png',dpi=155,bbox_inches='tight')
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,system in zip(axs,SYS):
        rows=[r for r in records if r['system']==system]
        for i,r in enumerate(rows):
            ax.plot([0,1],[r['before_forecast_mse_h10'],r['after_forecast_mse_h10']],
                    'o-',alpha=.8,label=f"Seed {r['seed']}")
        ax.set_xticks([0,1],['Original MAP','Train-only selected\nmultistep'])
        ax.set_ylabel('Held-out 10-step MSE')
        ax.set_title(system.replace('_',' ').title())
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Independent held-out forecasts (never used for selecting alpha)')
    fig.tight_layout()
    fig.savefig(figure_dir/'phase5_heldout_mse.png',dpi=155,bbox_inches='tight')
    plt.close(fig)


def plot_trajectories(system:str,reference:np.ndarray,before:np.ndarray|None,
                     after:np.ndarray|None,dest:Path)->None:
    fig=plt.figure(figsize=(12.5,4.2))
    sets=[(reference,'Independent reference'),(before,'Original MAP'),(after,'Refined (selected on train)')]
    N=reference.shape[1]
    for i,(x,title) in enumerate(sets,1):
        ax=fig.add_subplot(1,3,i,projection='3d' if N==3 else None)
        if x is None:
            ax.text(.05,.5,'Unstable: no plot',transform=ax.transAxes)
        elif N==3:ax.plot(x[:,0],x[:,1],x[:,2],lw=.6)
        else:ax.plot(x[:,0],x[:,1],lw=.6)
        ax.set_title(title,fontsize=9);ax.set_xlabel('x1');ax.set_ylabel('x2')
        if N==3:ax.set_zlabel('x3')
        for k in range(N):
            lo,hi=np.min(reference[:,k]),np.max(reference[:,k]);pad=max(.2,.15*(hi-lo))
            if k==0:ax.set_xlim(lo-pad,hi+pad)
            if k==1:ax.set_ylim(lo-pad,hi+pad)
            if k==2:ax.set_zlim(lo-pad,hi+pad)
        ax.grid(alpha=.15)
    fig.suptitle(f'{system.replace("_"," ").title()}: autonomous reconstruction, seed 42')
    fig.tight_layout();fig.savefig(dest,dpi=155,bbox_inches='tight');plt.close(fig)


def run_phase5(*,root:Path=ROOT,quick:bool=False,
               seeds:tuple[int,...]=SEEDS,alphas:tuple[float,...]=ALPHAS,
               config:RefinementConfig|None=None) -> dict:
    root=Path(root)
    out=root/'results'/('phase5_quick' if quick else 'phase5')
    figs=out/'figures';model_dir=out/'models';metrics=out/'metrics'
    for p in (figs,model_dir,metrics):p.mkdir(parents=True,exist_ok=True)
    config=config or RefinementConfig(iterations=65 if quick else 320,batch_size=48 if quick else 96)
    burn=80 if quick else 400
    n_eval=300 if quick else 2400
    train_selection_steps=350 if quick else 1200
    records=[]
    for system in SYS:
        ds=load_dataset(root/'data'/'synthetic'/f'{system}_seed42.npz')
        obs=np.asarray(ds['x_obs'],float)
        train_end=int(ds['train_end']); train_obs=obs[:train_end]
        init_indices=tuple(int(i) for i in [train_end-1,train_end//2,train_end//3])
        # A fully independent simulation (clean and unseen) is used for AFTER-fit evaluation only.
        independent_reference,_,_=benchmark_reference(system,n_eval,ds)
        print(f'\nPHASE 5: {system}',flush=True)
        for seed in seeds:
            f=(root/'results'/'metrics'/f'plrnn_{system}_model.npz' if seed==42
               else root/'results'/'phase4'/'models'/f'plrnn_{system}_seed{seed}.npz')
            if not f.exists():
                raise FileNotFoundError(f'Phase 3/4 checkpoint required: {f}')
            original=PiecewiseLinearSSM.load(str(f))
            if len(original.latent_train_) != train_end or original.n_latent != SYS[system]['m']:
                raise ValueError('checkpoint does not match synthetic training data')
            proposal,history=refine_transitions(original,train_obs,config)
            chosen,selection=choose_interpolation(original,proposal,train_obs,
                alphas=alphas,init_indices=init_indices,burn=burn,n_eval=train_selection_steps)
            model_path=model_dir/f'plrnn_{system}_seed{seed}_refined.npz'
            chosen.save(str(model_path))
            old_runs,before_plot=independent_rollouts(original,independent_reference,
                init_indices=init_indices,burn=burn,system=system)
            new_runs,after_plot=independent_rollouts(chosen,independent_reference,
                init_indices=init_indices,burn=burn,system=system)
            old_js=[r['js_width_1.0'] for r in old_runs if r['status']=='ok']
            new_js=[r['js_width_1.0'] for r in new_runs if r['status']=='ok']
            forecasts_old=original.forecast_mse(obs,train_end,horizons=(5,10,20))
            forecasts_new=chosen.forecast_mse(obs,train_end,horizons=(5,10,20))
            rec={'system':system,'seed':seed,'selected_alpha':selection['selected_alpha'],
                 'selected_train_js':selection['selected_train_js'],
                 'train_loss_before':multistep_loss(original,train_obs,horizon=config.horizon,stride=5),
                 'train_loss_after':multistep_loss(chosen,train_obs,horizon=config.horizon,stride=5),
                 'before_forecast_mse_h5':float(forecasts_old[5]),
                 'after_forecast_mse_h5':float(forecasts_new[5]),
                 'before_forecast_mse_h10':float(forecasts_old[10]),
                 'after_forecast_mse_h10':float(forecasts_new[10]),
                 'before_forecast_mse_h20':float(forecasts_old[20]),
                 'after_forecast_mse_h20':float(forecasts_new[20]),
                 'before_independent_js_median':float(np.median(old_js)) if old_js else None,
                 'after_independent_js_median':float(np.median(new_js)) if new_js else None,
                 'before_stable_runs':len(old_js),'after_stable_runs':len(new_js),
                 'n_initial_states':len(init_indices),'before_independent_runs':old_runs,
                 'after_independent_runs':new_runs,'selection':selection,'training_log':history,
                 'model_path':str(model_path.relative_to(root))}
            records.append(rec)
            print(f"  seed={seed} alpha={rec['selected_alpha']:.2f} "
                  f"test 10-step {rec['before_forecast_mse_h10']:.3f} -> {rec['after_forecast_mse_h10']:.3f}; "
                  f"independent JSD {rec['before_independent_js_median']} -> {rec['after_independent_js_median']}",flush=True)
            if seed==42:plot_trajectories(system,independent_reference,before_plot,after_plot,
                                         figs/f'phase5_{system}_rollouts.png')
    plot_summary(records,figs)
    with (metrics/'phase5_records.json').open('w',encoding='utf8') as f:
        json.dump({'config':vars(config),'alpha_grid':list(alphas),'seeds':list(seeds),
                   'n_independent_reference':n_eval,'burn':burn,'records':records,
                   'interpretation':('exploratory training-only selection, not the 2019 Laplace-EM; '
                     'not a claim of recovering the Lorenz chaotic attractor')},f,indent=2)
    with (metrics/'phase5_summary.csv').open('w',encoding='utf-8',newline='') as f:
        columns=['system','seed','selected_alpha','train_loss_before','train_loss_after',
                 'before_forecast_mse_h5','after_forecast_mse_h5',
                 'before_forecast_mse_h10','after_forecast_mse_h10',
                 'before_forecast_mse_h20','after_forecast_mse_h20',
                 'before_independent_js_median','after_independent_js_median',
                 'before_stable_runs','after_stable_runs']
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        writer.writeheader();writer.writerows(records)
    print(f'\nPhase 5 outputs: {out}',flush=True)
    return {'records':records,'out':str(out)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description='Phase 5: stability-aware train-only multistep PLRNN refinement')
    p.add_argument('--quick',action='store_true',help='Smoke test only; separate output folder')
    args=p.parse_args()
    run_phase5(quick=args.quick)

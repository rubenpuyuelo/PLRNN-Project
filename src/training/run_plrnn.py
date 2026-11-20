"""Run Phase 3 PLRNN-SSM MAP experiments, comparing saved Phase 2 LDS results.

From repository root:
    python -m src.training.run_plrnn
    python -m src.training.run_plrnn --quick    # optional smoke test

Data and LDS model outputs are loaded, never overwritten by this script.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.data.preprocessing import load_dataset
from src.evaluation.dynamics import occupancy_metrics
from src.models.plrnn import PiecewiseLinearSSM

ROOT = Path(__file__).resolve().parents[2]


def _compare_plot(name: str, reference: np.ndarray, plrnn: np.ndarray,
                  lds: np.ndarray, destination: Path) -> None:
    fig = plt.figure(figsize=(13.2, 4.5))
    arrays = [reference, plrnn, lds]
    titles = ['True system (held-out)', 'PLRNN: free-running', 'LDS: free-running']
    axes = []
    for i, (arr, title) in enumerate(zip(arrays, titles), start=1):
        ax = fig.add_subplot(1, 3, i, projection='3d' if reference.shape[1] == 3 else None)
        if reference.shape[1] == 3:
            ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], lw=0.9)
            ax.set_zlabel('x3')
        else:
            ax.plot(arr[:, 0], arr[:, 1], lw=0.9)
        ax.set_xlabel('x1'); ax.set_ylabel('x2')
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.22)
        axes.append(ax)
    for j in range(reference.shape[1]):
        lo, hi = reference[:, j].min(), reference[:, j].max()
        pad = max(0.2, 0.15 * (hi-lo))
        for ax in axes:
            if j == 0: ax.set_xlim(lo-pad, hi+pad)
            if j == 1: ax.set_ylim(lo-pad, hi+pad)
            if j == 2: ax.set_zlim(lo-pad, hi+pad)
    fig.suptitle(f'{name.replace("_", " ").title()}: independent dynamics (shared axes)')
    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches='tight')
    plt.close(fig)


def _forecast_plot(records: dict, destination: Path) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (system, data) in zip(axs, records.items()):
        for field, title, style in [('plrnn_mse','PLRNN-SSM', 'o-'),
                                     ('lds_mse','LDS-SSM','s-'),
                                     ('persistence_mse','Persistence','^--')]:
            keys = sorted(int(k) for k in data[field])
            ax.plot(keys, [data[field][str(h)] for h in keys], style, label=title)
        ax.set_xlabel('Forecast horizon (time steps)')
        ax.set_ylabel('Held-out MSE')
        ax.set_xticks(keys)
        ax.set_title(system.replace('_',' ').title())
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle('Causal test forecasts (PLRNN and LDS use different inference methods)')
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def _history_plot(histories: dict, destination: Path) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (system, hist) in zip(axs, histories.items()):
        i = np.arange(1, len(hist)+1)
        ax.plot(i, [v['observation_mse'] for v in hist], 'o-', ms=2, label='Observation MSE')
        ax.plot(i, [v['dynamics_mse'] for v in hist], 's-', ms=2, label='Latent dynamics MSE')
        for j in range(1, len(hist)):
            if hist[j]['stage'] != hist[j-1]['stage']:
                ax.axvline(j+0.5, color='gray', alpha=0.35, ls=':')
        ax.set_xlabel('Alternating optimization step (dotted = Q changed)')
        ax.set_ylabel('Mean square error (different spaces)')
        ax.set_title(system.replace('_',' ').title())
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle('PLRNN MAP training diagnostics — not EM log-likelihood')
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def _weights_plot(models: dict, destination: Path) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    bound = max(np.max(np.abs(m.W)) for m in models.values())
    for ax, (system, m) in zip(axs, models.items()):
        im = ax.imshow(m.W, cmap='coolwarm', vmin=-bound, vmax=bound)
        ax.set_title(f'{system.replace("_", " ").title()} — W (zero diagonal)')
        ax.set_xlabel('Source latent unit')
        ax.set_ylabel('Target latent unit')
    fig.colorbar(im, ax=axs.tolist(), fraction=0.02, pad=0.03, label='Nonlinear coupling')
    fig.subplots_adjust(wspace=0.4, right=0.88)
    fig.savefig(destination, dpi=160, bbox_inches='tight')
    plt.close(fig)


def run_experiments(*, data_dir: Path = ROOT / 'data' / 'synthetic',
                    figs: Path = ROOT / 'results' / 'figures',
                    metrics: Path = ROOT / 'results' / 'metrics',
                    quick: bool = False, seed: int = 42) -> dict:
    """Fit on first 700 points, predict held-out test without parameter refitting."""
    data_dir, figs, metrics = map(Path, (data_dir, figs, metrics))
    figs.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)
    results, histories, models = {}, {}, {}
    for name, M in [('van_der_pol', 8), ('lorenz', 12)]:
        dataset = data_dir / f'{name}_seed42.npz'
        lds_json = metrics / f'lds_{name}.json'
        lds_npz = metrics / f'lds_{name}_model.npz'
        missing = [str(f) for f in (dataset, lds_json, lds_npz) if not f.exists()]
        if missing:
            raise FileNotFoundError('Required previous-phase files missing: ' + ', '.join(missing))
        ds = load_dataset(dataset)
        obs = np.asarray(ds['x_obs'], dtype=np.float64)
        true = np.asarray(ds['x_true'], dtype=np.float64)
        train_end = int(ds['train_end'])
        with lds_json.open(encoding='utf-8') as f:
            lds_metrics = json.load(f)
        with np.load(lds_npz, allow_pickle=False) as f:
            lds_generated = f['x_generated'].copy()
        print(f'\n{name.upper()}: fitting PLRNN-SSM (training {train_end}/{len(obs)}, M={M})', flush=True)
        model = PiecewiseLinearSSM(n_latent=M, seed=seed, ridge_dyn=8.0)
        model.fit(obs[:train_end], outer_iters=2 if quick else 5,
                  latent_steps=18 if quick else 60, verbose=True)
        histories[name] = model.history_
        models[name] = model
        model_file = metrics / f'plrnn_{name}_model.npz'
        model.save(str(model_file))
        scores = model.forecast_mse(obs, train_end)
        n_test = len(obs) - train_end
        # Free-running: no true/test observations condition these generations.
        # Initial state is final MAP TRAINING state (Phase-2 LDS used its filtered train state).
        unstable = False
        try:
            z_gen, x_gen = model.rollout(model.latent_train_[-1], n_test)
            free_scores = occupancy_metrics(true[train_end:], x_gen)
            free_scores['max_abs_generated'] = float(np.max(np.abs(x_gen)))
        except FloatingPointError as e:
            unstable = True
            x_gen = None
            free_scores = {'status': 'unstable', 'reason': str(e)}
        lds_scores = occupancy_metrics(true[train_end:], lds_generated)
        record = {
            'system': name,
            'model': 'PLRNN-SSM with alternating joint-MAP latent optimization; approximate recognition at test',
            'paper_fidelity': 'simplified independent MAP implementation, not exact Laplace-EM',
            'seed': seed,
            'n_observed': int(obs.shape[1]),
            'n_latent_plrnn': M,
            'n_latent_lds': int(lds_metrics['n_latent']),
            'train_end': train_end,
            'n_test': n_test,
            'plrnn_mse': {str(k): float(v) for k,v in scores.items()},
            'lds_mse': lds_metrics['lds_mse'],
            'persistence_mse': lds_metrics['persistence_mse'],
            'plrnn_free': free_scores,
            'lds_free': lds_scores,
            'unstable_plrnn': unstable,
            'observation_mse_train': model.history_[-1]['observation_mse'],
            'dynamics_mse_train': model.history_[-1]['dynamics_mse'],
            'process_variance_schedule': [1.,0.3,0.1,0.03],
            'ridge_dyn': model.ridge_dyn,
            'note': ('Metrics are from one seed and 300 held-out points. The histogram KL is NOT the '
                     'paper normalized KL. PLRNN uses a trained auxiliary causal encoder + local MAP, '
                     'while LDS uses Kalman filtering; latent dimensions/parameter counts differ. '
                     'Out-of-sample MSE and free-run geometry measure different properties.')
        }
        results[name] = record
        (metrics / f'plrnn_{name}.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        print('Horizon   PLRNN MSE   LDS MSE   Persistence', flush=True)
        for h in scores:
            print(f'{h:>4d}      {scores[h]:>9.4f}  {lds_metrics["lds_mse"][str(h)]:>9.4f} '
                  f'{lds_metrics["persistence_mse"][str(h)]:>9.4f}', flush=True)
        print('PLRNN free generation:',free_scores,flush=True)
        if not unstable:
            _compare_plot(name, true[train_end:], x_gen, lds_generated, figs / f'plrnn_{name}.png')
    _forecast_plot(results, figs / 'plrnn_forecast_comparison.png')
    _history_plot(histories, figs / 'plrnn_training_curves.png')
    _weights_plot(models, figs / 'plrnn_weight_matrices.png')
    print(f'\nSaved comparison figures to: {figs}\nSaved Phase 3 metrics to: {metrics}', flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase 3 PLRNN-SSM experiment')
    parser.add_argument('--quick', action='store_true', help='smaller training loop; NOT full experiment')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run_experiments(quick=args.quick, seed=args.seed)

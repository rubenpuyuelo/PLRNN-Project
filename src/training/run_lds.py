"""Train LDS-SSM baselines on both Phase-1 datasets; save metrics and figures.

Run from repository root: python -m src.training.run_lds
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.data.preprocessing import load_dataset
from src.models.lds import LinearGaussianSSM


ROOT = Path(__file__).resolve().parents[2]


def persistence_mse(x: np.ndarray, train_end: int, horizon: int) -> float:
    """Naive forecast x[t+h] = x[t], with the same test anchors as the LDS."""
    x_pred = x[train_end - 1:len(x) - horizon]
    target = x[train_end - 1 + horizon:]
    return float(np.mean((target - x_pred) ** 2))


def plot_attractor(system: str, x_true: np.ndarray, x_generated: np.ndarray,
                   path: Path, *, n_display: int = 300) -> None:
    """Compare state-space geometry; do NOT align chaotic trajectories by timestamp."""
    fig = plt.figure(figsize=(12.2, 4.7))
    three_d = x_true.shape[1] == 3
    if three_d:
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
    else:
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
    for ax, data, label in [(ax1, x_true, 'True system (held-out)'),
                            (ax2, x_generated, 'LDS free-running')]:
        data = data[:n_display]
        if three_d:
            ax.plot(*data[:, :3].T, linewidth=0.8)
            ax.set_zlabel('x3')
        else:
            ax.plot(data[:, 0], data[:, 1], linewidth=1.0)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title(label)
        ax.grid(alpha=0.2)
    # Shared axes are essential: independently autoscaled plots would be misleading.
    for k, (low, high) in enumerate(zip(np.min(x_true[:n_display], axis=0),
                                       np.max(x_true[:n_display], axis=0))):
        pad = max(0.1, (high-low)*0.10)
        if k == 0:
            ax1.set_xlim(low-pad, high+pad); ax2.set_xlim(low-pad, high+pad)
        elif k == 1:
            ax1.set_ylim(low-pad, high+pad); ax2.set_ylim(low-pad, high+pad)
        elif three_d and k == 2:
            ax1.set_zlim(low-pad, high+pad); ax2.set_zlim(low-pad, high+pad)
    ax3 = fig.add_subplot(133)
    n = min(n_display, len(x_generated), len(x_true))
    ax3.plot(np.arange(n), x_true[:n, 0], label='True (test)', lw=1.3)
    ax3.plot(np.arange(n), x_generated[:n, 0], label='Generated (free)', lw=1.2)
    ax3.set_xlabel('Steps after training')
    ax3.set_ylabel('Standardized x1')
    ax3.set_title('Time series (not synchronized)')
    ax3.grid(alpha=0.2)
    ax3.legend(fontsize=8)
    fig.suptitle(f'{system}: LDS-SSM cannot sustain nonlinear attractors', fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_curves(all_results: dict, history: dict[str, list[float]], figs: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (name, values) in zip(axes, history.items()):
        ax.plot(range(len(values)), values, marker='.', markersize=2)
        ax.set_title(name.replace('_', ' ').title())
        ax.set_xlabel('EM iteration (0 = initialization)')
        ax.set_ylabel('Training log likelihood')
        ax.grid(alpha=0.2)
    fig.suptitle('LDS: constrained EM training')
    fig.tight_layout()
    fig.savefig(figs / 'lds_em_learning_curve.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (name, record) in zip(axes, all_results.items()):
        horizons = [int(v) for v in record['lds_mse'].keys()]
        ax.plot(horizons, [record['lds_mse'][str(k)] for k in horizons], 'o-', label='LDS-SSM')
        ax.plot(horizons, [record['persistence_mse'][str(k)] for k in horizons], 's--', label='Persistence')
        ax.set_title(name.replace('_', ' ').title())
        ax.set_xlabel('Forecast horizon (steps)')
        ax.set_ylabel('Test MSE (standardized units)')
        ax.set_xticks(horizons)
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle('Held-out causal forecasts (test anchors use past observations only)')
    fig.tight_layout()
    fig.savefig(figs / 'lds_forecast_mse.png', dpi=150)
    plt.close(fig)


def run_lds_experiments(*, data_dir: Path = ROOT / 'data' / 'synthetic',
                        fig_dir: Path = ROOT / 'results' / 'figures',
                        metrics_dir: Path = ROOT / 'results' / 'metrics',
                        n_iter: int = 35, seed: int = 42) -> dict:
    """Train each LDS strictly on phase-1 training split; save reproducible outputs."""
    data_dir, fig_dir, metrics_dir = map(Path, (data_dir, fig_dir, metrics_dir))
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    results, histories = {}, {}
    for system, dimensions in [('van_der_pol', 2), ('lorenz', 3)]:
        path = data_dir / f'{system}_seed42.npz'
        if not path.exists():
            raise FileNotFoundError(f'{path} not found. Run phase-1 notebook first.')
        ds = load_dataset(path)
        x = np.asarray(ds['x_obs'], dtype=np.float64)
        clean = np.asarray(ds['x_true'], dtype=np.float64)
        train_end = int(ds['train_end'])
        model = LinearGaussianSSM(n_latent=dimensions, seed=seed)
        print(f'\n--- {system}: training LDS ({train_end}/{len(x)} samples) ---', flush=True)
        model.fit(x[:train_end], n_iter=n_iter, verbose=True)
        histories[system] = [float(v) for v in model.history_]

        # Free generation starts from a state estimated using TRAIN data only;
        # test observations do not initialize it.
        anchor = model.filter(x[:train_end]).filtered_mean[-1]
        z_gen, x_gen = model.rollout(anchor, len(x)-train_end, seed=seed)
        nrmse = model.forecast_mse(x, train_end)
        persist = {int(h): persistence_mse(x, train_end, int(h)) for h in nrmse}
        print('horizon      LDS-MSE    Persistence-MSE')
        for h in nrmse:
            print(f'{h:>5d}    {nrmse[h]:10.4f}    {persist[h]:10.4f}')
        plot_attractor(system, clean[train_end:], x_gen, fig_dir / f'lds_{system}.png')
        record = {
            'system': system,
            'train_end': train_end,
            'n_test': len(x) - train_end,
            'n_observed': int(x.shape[1]),
            'n_latent': dimensions,
            'n_em_iterations': len(model.history_) - 1,
            'seed': seed,
            'model': 'Linear Gaussian LDS-SSM with diagonal process and observation covariances',
            'stability_constraint': f'spectral radius <= {model.max_radius}',
            'training_log_likelihood_initial': float(model.history_[0]),
            'training_log_likelihood_final': float(model.history_[-1]),
            'spectral_radius_F': model.spectral_radius,
            'lds_mse': {str(h): val for h, val in nrmse.items()},
            'persistence_mse': {str(h): val for h, val in persist.items()},
            'note': ('Forecast scores condition on past test observations through a causal Kalman filter; '
                     'free-running plots condition ONLY on the final TRAIN filtered latent state. '
                     'No KL/chaos equivalence claims are made by this baseline.'),
            'F': model.F.tolist(), 'h': model.h.tolist(), 'B': model.B.tolist(),
            'Q_diag': model.Q.tolist(), 'R_diag': model.R.tolist()
        }
        results[system] = record
        # Keep fitted parameters compact and machine-readable for phase-3 comparisons.
        np.savez_compressed(metrics_dir / f'lds_{system}_model.npz',
                            F=model.F, h=model.h, B=model.B, Q=model.Q, R=model.R,
                            mu0=model.mu0, P0=model.P0, z_generated=z_gen,
                            x_generated=x_gen, history=np.asarray(model.history_))
        (metrics_dir / f'lds_{system}.json').write_text(
            json.dumps(record, indent=2), encoding='utf-8')
    plot_curves(results, histories, fig_dir)
    print(f'\nSaved plots: {fig_dir}\nSaved metrics/models: {metrics_dir}', flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Phase 2 LDS baseline')
    parser.add_argument('--iterations', type=int, default=35)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run_lds_experiments(n_iter=args.iterations, seed=args.seed)

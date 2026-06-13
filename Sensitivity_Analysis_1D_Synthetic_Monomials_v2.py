import numpy as np
import pandas as pd
import os
import datetime
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter
from pysindy.optimizers import STLSQ
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

noise_values = [i / 100 for i in range(0, 21, 2)]

cut_off      = 1000
sg_window    = 51
sg_polyorder = 5
random_seed  = 42

v_sim, c_sim = 0.3, 0.4
ESS_true     = v_sim / c_sim          # 0.75

# Monomial library {p, p², p³, p⁴, p⁵}, no constant term.
# In this basis the replicator equation is dp/dt = 0.15 p - 0.35 p² + 0.20 p³,
# so its support is {0, 1, 2}.
feature_names = ['p', 'p²', 'p³', 'p⁴', 'p⁵']
true_support  = {0, 1, 2}
true_key      = (0, 1, 2)

# true coefficients, known here because the data is synthetic
true_c1 =  v_sim / 2               #  0.15
true_c2 = -(v_sim + c_sim) / 2     # -0.35
true_c3 =  c_sim / 2               #  0.20

# plot colors and linestyles
COL_TEST = 'gray'         # test trajectories
COL_BEST = 'tab:orange'   # identified non-replicator structure
COL_REP  = 'tab:blue'     # replicator
COL_SENS = 'black'        # identified model in the sensitivity plot
LS_BEST  = '-.'
LS_REP   = '--'

t_full   = np.linspace(0, 50, 1000)
dt       = t_full[1] - t_full[0]
p0_train = np.round(np.linspace(0.05, 0.95, 10), 3).tolist()
p0_test  = np.round(np.linspace(0.03, 0.97, 20), 3).tolist()

TEST_PLOT_IDX = [0, 2, 5, 8, 11, 14, 17, 19]

RESULTS_ROOT = "Results"
results_dir = os.path.join(RESULTS_ROOT, 'Synthetic')
os.makedirs(results_dir, exist_ok=True)
out_path = os.path.join(results_dir, "sensitivity_analysis_1D_synthetic_monomials.xlsx")


def replicator_1d(t, p, v, c):
    return 0.5 * p * (1 - p) * (v - c * p)


def integrate(p0):
    sol = solve_ivp(replicator_1d, [t_full[0], t_full[-1]], [p0],
                    t_eval=t_full, args=(v_sim, c_sim))
    return sol.y.T


def get_roots(coeffs, support):
    # roots of Q(p) = dp/dt / p: interior root near the ESS, root near 1, any other, count in (0,1)
    max_deg = max(support)
    poly = np.zeros(max_deg + 1)
    for i, s in enumerate(sorted(support)):
        poly[s] = coeffs[i]
    all_roots  = np.roots(poly[::-1])
    real_roots = [r.real for r in all_roots if abs(r.imag) < 1e-6]

    if not real_roots:
        return np.nan, np.nan, np.nan, 0

    n_interior = sum(1 for r in real_roots if 1e-4 < r < 1 - 1e-4)

    interior = [r for r in real_roots if 1e-4 < r < 1 - 1e-4]
    root1 = min(interior, key=lambda x: abs(x - ESS_true)) if interior else np.nan

    rem1  = [r for r in real_roots if abs(r - root1) > 1e-6] if not np.isnan(root1) else real_roots
    root2 = min(rem1, key=lambda x: abs(x - 1.0)) if rem1 else np.nan

    rem2  = [r for r in rem1 if abs(r - root2) > 1e-6] if not np.isnan(root2) else rem1
    root3 = rem2[0] if rem2 else np.nan

    return root1, root2, root3, n_interior


def root_stability(p_star, coeffs, support):
    # stable if f'(p*) < 0, unstable if > 0
    if np.isnan(p_star):
        return np.nan
    sorted_sup = sorted(support)
    dQ = sum(c * s * (p_star ** (s - 1)) for c, s in zip(coeffs, sorted_sup) if s > 0)
    fp = p_star * dQ
    if abs(fp) < 1e-8:
        return 'non-hyp'
    return 'stable' if fp < 0 else 'unstable'


def struct_label(support):
    return '{' + ', '.join(feature_names[int(i)] for i in sorted(support)) + '}'


def run_analysis(noise_level):
    np.random.seed(random_seed)
    X_train_raw = []
    for p0 in p0_train:
        traj = integrate(p0)
        if noise_level > 0:
            traj = traj + np.random.normal(0, noise_level, traj.shape)
        X_train_raw.append(traj)

    np.random.seed(random_seed + 100)
    X_test = []
    for p0 in p0_test:
        traj = integrate(p0)
        if noise_level > 0:
            traj = traj + np.random.normal(0, noise_level, traj.shape)
        X_test.append(traj)

    X_test_smoothed = []
    for traj_raw in X_test:
        X_test_smoothed.append(
            savgol_filter(traj_raw[:cut_off, 0], sg_window, sg_polyorder).reshape(-1, 1))

    x_train_now, y_train_delta = [], []
    for traj_raw in X_train_raw:
        traj = traj_raw[:cut_off, 0]
        x_train_now.append(savgol_filter(traj, sg_window, sg_polyorder).reshape(-1, 1))
        y_train_delta.append(savgol_filter(traj, sg_window, sg_polyorder,
                                           deriv=1, delta=dt).reshape(-1, 1))

    x_stacked = np.vstack(x_train_now)
    y_stacked = np.vstack(y_train_delta)

    p_vals    = x_stacked.flatten()

    # monomial library p, p², p³, p⁴, p⁵, columns normalized to unit norm
    Theta     = np.column_stack([p_vals**(k+1) for k in range(5)])
    col_norms = np.linalg.norm(Theta, axis=0)
    Theta_norm = Theta / col_norms

    unique_structures = {}

    # thresholds chosen for the coefficient scale of this game (v=0.3, c=0.4)
    thresholds   = np.logspace(-2, 0, 30)
    alphas_stlsq = [0, 0.01, 0.1]
    for thr in thresholds:
        for alpha in alphas_stlsq:
            try:
                opt = STLSQ(threshold=thr, alpha=alpha, max_iter=100,
                            normalize_columns=False, unbias=False)
                opt.fit(Theta_norm, y_stacked)
                sup = np.where(opt.coef_[0] != 0)[0]
                if len(sup) < 1:
                    continue
                key = tuple(int(i) for i in sorted(sup))
                if key not in unique_structures:
                    # OLS refit on the found support for unbiased coefficients
                    c_ols, _, _, _ = np.linalg.lstsq(
                        Theta_norm[:, sup], y_stacked, rcond=None)
                    unique_structures[key] = {
                        'features': [feature_names[i] for i in sup],
                        'coeffs':   c_ols.flatten() / col_norms[sup],
                        'support':  sup
                    }
            except Exception:
                pass

    # always keep the replicator structure in the pool so its AICc is reported
    true_in_pool_stlsq = true_key in unique_structures
    n_candidates_stlsq = len(unique_structures)          # count before injection
    if not true_in_pool_stlsq:
        sup     = np.array(list(true_key))
        Theta_s = Theta_norm[:, sup]
        c_ols, _, _, _ = np.linalg.lstsq(Theta_s, y_stacked, rcond=None)
        unique_structures[true_key] = {
            'features': [feature_names[i] for i in sup],
            'coeffs':   c_ols.flatten() / col_norms[sup],
            'support':  sup,
            'injected': True,
        }

    eval_steps = len(t_full) - 1
    m_tilde    = len(p0_test)

    selection_results = []
    for data in unique_structures.values():
        features = data['features']
        coeffs   = data['coeffs']
        support  = data['support']

        def get_delta(p_val, _c=coeffs, _s=support):
            return np.dot(_c, np.array([p_val**(i+1) for i in _s]))

        errors = []
        for i in range(m_tilde):
            try:
                p_sim = [p0_test[i]]
                for _ in range(eval_steps):
                    p_sim.append(p_sim[-1] + get_delta(p_sim[-1]) * dt)
                p_gt  = list(X_test_smoothed[i][:eval_steps + 1, 0])
                p_arr = np.array(p_sim)
                if np.any(np.isnan(p_arr)) or np.any(p_arr > 1 + 1e-6) or np.any(p_arr < -1e-6):
                    errors.append(1.0)
                else:
                    errors.append(np.mean((np.array(p_gt) - p_arr) ** 2))
            except Exception:
                errors.append(1.0)

        avg_mse = np.mean(errors)
        k       = len(features)
        aicc    = (m_tilde * np.log(avg_mse + 1e-15)
                   + 2*k + 2*(k+1)*(k+2) / max(m_tilde - k - 2, 1))
        selection_results.append({
            'features':  features, 'coeffs': coeffs, 'support': support,
            'mse':       avg_mse, 'aicc': aicc,
            'stability': np.mean([e < 1.0 for e in errors]) * 100,
            'injected':  data.get('injected', False),
        })

    if not selection_results:
        return {'noise_eta': noise_level, 'error': 'no stable models found'}

    best     = min(selection_results, key=lambda x: x['aicc'])
    aicc_min = best['aicc']

    aicc_true  = next((r['aicc'] for r in selection_results
                       if tuple(sorted(int(i) for i in r['support'])) == true_key), None)
    delta_true = (aicc_true - aicc_min) if aicc_true is not None else np.nan

    id_support = set(int(i) for i in best['support'])
    tp = len(true_support & id_support)
    fp = len(id_support - true_support)
    fn = len(true_support - id_support)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0

    # coefficient error against the true replicator coefficients, over the full library
    # (only meaningful when the identified structure is the replicator)
    coeff_map = dict(zip(best['features'], best['coeffs']))
    c1_id = coeff_map.get('p',  None)
    c2_id = coeff_map.get('p²', None)
    c3_id = coeff_map.get('p³', None)
    true_coeffs_full = np.array([true_c1, true_c2, true_c3, 0.0, 0.0])
    id_coeffs_full   = np.array([coeff_map.get(f, 0.0) for f in feature_names])
    mae_c = float(np.mean(np.abs(true_coeffs_full - id_coeffs_full)))

    # fixed points and their stability
    p_star_est, r1_bnd, r2_bnd, n_interior = get_roots(best['coeffs'], best['support'])
    stab_p_star = root_stability(p_star_est, best['coeffs'], best['support'])
    stab_r1     = root_stability(r1_bnd,     best['coeffs'], best['support'])
    stab_r2     = root_stability(r2_bnd,     best['coeffs'], best['support'])
    ae_p_star   = round(abs(p_star_est - ESS_true), 4) if not np.isnan(p_star_est) else np.nan
    ae_r1       = round(abs(r1_bnd - 1.0),          4) if not np.isnan(r1_bnd)     else np.nan

    eq_str     = " + ".join([f"({c:+.6f})*{f}" for f, c in zip(best['features'], best['coeffs'])])
    sorted_res = sorted(selection_results, key=lambda x: x['aicc'])
    delta_2nd  = sorted_res[1]['aicc'] - aicc_min if len(sorted_res) > 1 else np.nan
    all_candidates = " | ".join(f"{r['features']} (d={r['aicc']-aicc_min:.1f})" for r in sorted_res)

    def simulate_model(p0, n_steps, coeffs, support):
        def f(p):
            return np.dot(coeffs, np.array([p**(i+1) for i in support]))
        p, hist = p0, [p0]
        for _ in range(n_steps):
            p = np.clip(p + f(p) * dt, 0, 1)
            hist.append(p)
        return np.array(hist)

    noise_eta = noise_level
    best_is_rep = tuple(sorted(int(i) for i in best['support'])) == true_key
    # replicator -> blue dashed, any other structure -> orange dash-dot
    model_color = COL_REP if best_is_rep else COL_BEST
    model_ls    = LS_REP  if best_is_rep else LS_BEST
    model_name  = 'Replicator' if best_is_rep else 'Identified model'

    fig, ax = plt.subplots(figsize=(10, 6))
    for traj_i in TEST_PLOT_IDX:
        p0 = X_test_smoothed[traj_i][0, 0]
        # test trajectory
        ax.plot(t_full[:cut_off], X_test_smoothed[traj_i][:, 0],
                color=COL_TEST, lw=1.3, alpha=0.85)
        # model prediction
        ax.plot(t_full,
                simulate_model(p0, eval_steps, best['coeffs'], best['support']),
                color=model_color, lw=1.6, ls=model_ls, alpha=0.95)

    ax.axhline(y=ESS_true, color='gray', ls=':', lw=1.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Time t')
    ax.set_ylabel('p')

    legend_handles = [
        Line2D([0], [0], color=COL_TEST, lw=1.3, ls='-', label='Test trajectories'),
        Line2D([0], [0], color=model_color, lw=1.6, ls=model_ls,
               label=f"{model_name}  {struct_label(best['support'])}"),
        Line2D([0], [0], color='gray', lw=1.5, ls=':', label=f'ESS = {ESS_true:.3f}'),
    ]
    ax.legend(handles=legend_handles, fontsize=10, loc='lower right')

    ax.set_title(f'Identified Model - Test Trajectory Predictions  '
                 f'($\\eta$ = {noise_eta*100:.0f}%)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plot_name = f"test_plot_noise_eta{noise_eta:.3f}.png".replace('.', 'p')
    fig.savefig(os.path.join(results_dir, plot_name), dpi=200, bbox_inches='tight')
    plt.close(fig)

    return {
        'noise_eta':            noise_eta,
        'identified_structure': str(best['features']),
        'equation':             eq_str,
        'F1':                   round(f1,              4),
        'MSE_avg_trajectory':   round(best['mse'],     8),
        'RMSE_avg_trajectory':  round(best['mse']**0.5, 6),
        'stability_%':          round(best['stability'], 1),
        'AICc_best':            round(aicc_min,        2),
        'delta_AICc_2nd':       round(delta_2nd,       2) if not np.isnan(delta_2nd)  else np.nan,
        'model_2nd':            str(sorted_res[1]['features']) if len(sorted_res) > 1 else np.nan,
        'p_star_est':           round(p_star_est,        4) if not np.isnan(p_star_est) else np.nan,
        'AE_p_star':            ae_p_star,
        'p_star_stability':     stab_p_star,
        'r1':                   round(r1_bnd,            4) if not np.isnan(r1_bnd)    else np.nan,
        'r1_error':             ae_r1,
        'r1_stability':         stab_r1,
        'r2':                   round(r2_bnd,            4) if not np.isnan(r2_bnd)    else np.nan,
        'r2_stability':         stab_r2,
        'n_interior_roots':     n_interior,
        'replicator_status':    'found' if true_in_pool_stlsq else 'injected',
        'delta_AICc_true':      round(delta_true,      2) if not np.isnan(delta_true) else np.nan,
        'c1_p':                 round(c1_id, 8) if c1_id is not None else np.nan,
        'c2_p2':                round(c2_id, 8) if c2_id is not None else np.nan,
        'c3_p3':                round(c3_id, 8) if c3_id is not None else np.nan,
        'MAE_coeffs':           round(mae_c,  6) if not np.isnan(mae_c) else np.nan,
        'precision':            round(precision,        4),
        'recall':               round(recall,           4),
        'ESS_true':             round(ESS_true,         4),
        'n_candidates_stlsq':   n_candidates_stlsq,
        'all_candidates':       all_candidates,
        'timestamp':            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'cut_off':              cut_off,
        'sg_window':            sg_window,
        'sg_polyorder':         sg_polyorder,
        'library':              'monomials {p, p², p³, p⁴, p⁵}',
    }


if __name__ == "__main__":
    rows = []
    for noise_level in noise_values:
        rows.append(run_analysis(noise_level))

    df = pd.DataFrame(rows)
    df.to_excel(out_path, index=False)
    print(f"Results saved to: {out_path}")

    # sensitivity plot: F1, coefficient error, ESS error and trajectory error vs noise
    noise_etas = [r['noise_eta']               for r in rows]
    f1_vals    = [r.get('F1',                  np.nan) for r in rows]
    mae_vals   = [r.get('MAE_coeffs',          np.nan) for r in rows]
    ae_pstar   = [r.get('AE_p_star',           np.nan) for r in rows]
    rmse_vals  = [r.get('RMSE_avg_trajectory', np.nan) for r in rows]
    structures = [r.get('identified_structure', 'unknown') for r in rows]

    def fmt_struct(s):
        try:
            inner = s.strip()[1:-1]
            terms = [t.strip().strip("'\"") for t in inner.split(',')]
            return '{' + ', '.join(terms) + '}'
        except Exception:
            return s

    unique_structs   = list(dict.fromkeys(structures))
    marker_cycle     = ['o', '^', 's', 'D', 'v', 'p', '*']
    struct_to_marker = {s: marker_cycle[i % len(marker_cycle)]
                        for i, s in enumerate(unique_structs)}

    fig, axes = plt.subplots(4, 1, figsize=(7, 10), sharex=True)
    fig.suptitle('Sensitivity Analysis (Synthetic Data)', fontsize=11, fontweight='bold')

    metrics = [
        (f1_vals,   '$F_1$',                              False, 1.0),
        (mae_vals,  r'$\mathrm{MAE}_{\boldsymbol{\Xi}}$', True,  None),
        (ae_pstar,  r'$\mathrm{AE}_{\hat{p}^*}$',         False, 0.0),
        (rmse_vals, r'$\overline{\mathrm{RMSE}}$',        False, None),
    ]

    for ax, (y, ylabel, log_scale, ref) in zip(axes, metrics):
        ax.plot(noise_etas, y, '-', color=COL_SENS, lw=1.5, alpha=0.6, zorder=2)
        for eta, yi, struct in zip(noise_etas, y, structures):
            ax.plot(eta, yi, marker=struct_to_marker[struct],
                    color=COL_SENS, ms=7, ls='none', zorder=3)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3, zorder=0)
        if log_scale:
            ax.set_yscale('log')
        if ref is not None:
            ax.axhline(ref, color='gray', ls='--', lw=1, zorder=2)

    axes[-1].set_xlabel('Noise level $\\eta$', fontsize=11)
    axes[-1].set_xticks(noise_etas)
    axes[-1].set_xticklabels([f'{e:.2f}' for e in noise_etas], rotation=45)

    legend_handles = [
        Line2D([0], [0], marker=struct_to_marker[s], color=COL_SENS,
               ls='none', ms=7, label=fmt_struct(s))
        for s in unique_structs
    ]
    axes[0].legend(handles=legend_handles, title='Identified structure',
                   fontsize=8, title_fontsize=8, loc='lower left', framealpha=0.85)

    plt.tight_layout()
    sens_plot_path = os.path.join(results_dir, 'sensitivity_analysis_4metrics.png')
    fig.savefig(sens_plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Sensitivity plot saved to: {sens_plot_path}")

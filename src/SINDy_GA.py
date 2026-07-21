import numpy as np
import pandas as pd
import os
import datetime
from pysindy.optimizers import STLSQ
import warnings
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

n_avg_values = [1, 5, 10, 15, 20, 25, 30, 35, 40]
cut_off      = 150
sg_window    = 21
sg_polyorder = 5

RESULTS_ROOT = "Results"
death_rule   = "inverse"          # "inverse" or "uniform"
base_path    = os.path.join(RESULTS_ROOT, f"GA_{death_rule}_death")

train_file = os.path.join(base_path, "ga_master_train_ens40.csv")
test_file  = os.path.join(base_path, "ga_master_test_ens40.csv")

v_sim, c_sim = 0.3, 0.4
ESS_true     = v_sim / c_sim

# Monomial library {p, p², p³, p⁴, p⁵}, no constant term.
# In this basis the replicator equation is dp/dt = 0.15 p - 0.35 p² + 0.20 p³,
# so its support is {0, 1, 2}.
feature_names = ['p', 'p²', 'p³', 'p⁴', 'p⁵']
true_support  = {0, 1, 2}
true_key      = (0, 1, 2)

TEST_PLOT_IDX = [0, 2, 5, 8, 11, 14, 17, 19]

results_dir = base_path
os.makedirs(results_dir, exist_ok=True)
out_path = os.path.join(results_dir, "sensitivity_analysis_1D_monomials.xlsx")


def load_and_average(csv_file, n_avg):
    df = pd.read_csv(csv_file)
    start_vals = sorted(set(col.rsplit('_run_', 1)[0] for col in df.columns))
    averaged = []
    for sv in start_vals:
        cols = [c for c in df.columns if c.startswith(sv + '_run_')][:n_avg]
        averaged.append(df[cols].mean(axis=1).values.reshape(-1, 1))
    return averaged


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

    # interior root closest to the true ESS
    interior = [r for r in real_roots if 1e-4 < r < 1 - 1e-4]
    root1 = min(interior, key=lambda x: abs(x - ESS_true)) if interior else np.nan

    # remaining root closest to 1
    rem1  = [r for r in real_roots if abs(r - root1) > 1e-6] if not np.isnan(root1) else real_roots
    root2 = min(rem1, key=lambda x: abs(x - 1.0)) if rem1 else np.nan

    # whatever is left
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


def run_analysis(n_avg):
    X_train_raw = load_and_average(train_file, n_avg)
    X_test_raw  = load_and_average(test_file,  n_avg)

    x_train_now, y_train_delta = [], []
    for traj_raw in X_train_raw:
        traj = traj_raw[:cut_off].flatten()
        x_train_now.append(savgol_filter(traj, sg_window, sg_polyorder).reshape(-1, 1))
        y_train_delta.append(savgol_filter(traj, sg_window, sg_polyorder,
                                           deriv=1, delta=1.0).reshape(-1, 1))

    X_test_smoothed = []
    for traj_raw in X_test_raw:
        X_test_smoothed.append(
            savgol_filter(traj_raw[:cut_off].flatten(), sg_window, sg_polyorder).reshape(-1, 1))

    x_stacked = np.vstack(x_train_now)
    y_stacked = np.vstack(y_train_delta)

    p_vals    = x_stacked.flatten()

    # monomial library p, p², p³, p⁴, p⁵, columns normalized to unit norm
    Theta     = np.column_stack([p_vals**(k+1) for k in range(5)])
    col_norms = np.linalg.norm(Theta, axis=0)
    Theta_norm = Theta / col_norms

    unique_structures = {}

    thresholds   = np.logspace(-3, -0.3, 50)
    alphas_stlsq = [0, 0.001, 0.01, 0.1]
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
    true_in_pool_stlsq  = true_key in unique_structures
    n_candidates_stlsq  = len(unique_structures)   # count before injection
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

    p0_test    = [traj[0, 0] for traj in X_test_smoothed]
    eval_steps = len(X_test_smoothed[0]) - 1
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
                p_gt  = [X_test_smoothed[i][0, 0]]
                for s in range(eval_steps):
                    p_sim.append(p_sim[-1] + get_delta(p_sim[-1]))
                    p_gt.append(X_test_smoothed[i][s + 1, 0])
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
            'features':   features, 'coeffs': coeffs, 'support': support,
            'mse':        avg_mse, 'aicc': aicc,
            'stability':  np.mean([e < 1.0 for e in errors]) * 100,
            'injected':   data.get('injected', False),
        })

    if not selection_results:
        return {'n_avg': n_avg, 'error': 'no stable models found'}

    best     = min(selection_results, key=lambda x: x['aicc'])
    aicc_min = best['aicc']

    aicc_true  = next((r['aicc'] for r in selection_results
                       if tuple(sorted(int(i) for i in r['support'])) == true_key), None)
    delta_true = (aicc_true - aicc_min) if aicc_true is not None else np.nan

    id_support = set(int(i) for i in best['support'])
    tp = len(true_support & id_support)
    fp = len(id_support - true_support)
    fn = len(true_support - id_support)
    precision  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall     = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1         = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0

    # fixed points of Q(p) = dp/dt / p and their stability
    p_star_est, r1_bnd, r2_bnd, n_interior = get_roots(best['coeffs'], best['support'])
    stab_p_star = root_stability(p_star_est, best['coeffs'], best['support'])
    stab_r1     = root_stability(r1_bnd,     best['coeffs'], best['support'])
    stab_r2     = root_stability(r2_bnd,     best['coeffs'], best['support'])

    eq_str     = " + ".join([f"({c:+.6f})*{f}" for f, c in zip(best['features'], best['coeffs'])])
    sorted_res = sorted(selection_results, key=lambda x: x['aicc'])
    delta_2nd  = sorted_res[1]['aicc'] - aicc_min if len(sorted_res) > 1 else np.nan
    all_candidates = " | ".join(f"{r['features']} (d={r['aicc']-aicc_min:.1f})" for r in sorted_res)

    def simulate_model(p0, n_steps, coeffs, support):
        p, hist = p0, [p0]
        for _ in range(n_steps):
            p = np.clip(p + np.dot(coeffs, np.array([p**(i+1) for i in support])), 0, 1)
            hist.append(p)
        return np.array(hist)

    cmap     = plt.cm.coolwarm
    gen_axis = np.arange(cut_off)

    fig, ax = plt.subplots(figsize=(10, 6))
    for traj_i in TEST_PLOT_IDX:
        p0    = X_test_smoothed[traj_i][0, 0]
        color = cmap(np.clip(p0, 0, 1))
        ax.plot(gen_axis, X_test_smoothed[traj_i][:cut_off].flatten(),
                color=color, lw=1.5, alpha=0.85)
        ax.plot(gen_axis,
                simulate_model(p0, cut_off - 1, best['coeffs'], best['support']),
                color='black', lw=1.2, ls='--', alpha=0.7)

    ax.axhline(y=ESS_true, color='gray', ls=':', lw=1.5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='p₀')

    legend_handles = [
        Line2D([0], [0], color='gray',  lw=1.5, ls='-',  label='Test (smoothed, color = p₀)'),
        Line2D([0], [0], color='black', lw=1.2, ls='--', label='Model prediction'),
        Line2D([0], [0], color='gray',  lw=1.5, ls=':',  label=f'ESS = {ESS_true:.3f}'),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc='upper right')
    ax.set_ylim(0, 1)
    ax.set_xlabel('Generation')
    ax.set_ylabel('p')

    fig.suptitle(f'Model Prediction vs Test Data  ($n_{{\\mathrm{{avg}}}}$ = {n_avg})',
                 fontsize=11, fontweight='bold')
    ax.set_title(f'Simulated model:  dp/dt = {eq_str}', fontsize=8)
    plt.tight_layout()
    plot_name = f"test_plot_navg{n_avg:02d}.png"
    fig.savefig(os.path.join(results_dir, plot_name), dpi=200, bbox_inches='tight')
    plt.close(fig)

    ae_p_star = round(abs(p_star_est - ESS_true), 4) if not np.isnan(p_star_est) else np.nan
    ae_r1     = round(abs(r1_bnd - 1.0),          4) if not np.isnan(r1_bnd)     else np.nan

    return {
        'n_avg':                n_avg,
        'identified_structure': str(best['features']),
        'equation':             eq_str,
        'F1':                   round(f1,              4),
        'MSE_avg_trajectory':   round(best['mse'],     8),
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
    for n_avg in n_avg_values:
        rows.append(run_analysis(n_avg))

    df = pd.DataFrame(rows)
    df.to_excel(out_path, index=False)
    print(f"Results saved to: {out_path}")

    # sensitivity plot: F1, ESS error and trajectory error vs the number of averages
    n_avgs   = [r['n_avg']                for r in rows]
    f1_vals  = [r.get('F1',               np.nan) for r in rows]
    ae_pstar = [r.get('AE_p_star',        np.nan) for r in rows]
    mse_vals = [r.get('MSE_avg_trajectory',np.nan) for r in rows]
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

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    fig.suptitle('Sensitivity Analysis (GA Data - Inverse-Proportional Death)',
                 fontsize=11, fontweight='bold')

    metrics = [
        (f1_vals,  '$F_1$',                    'steelblue', False, 1.0),
        (ae_pstar, r'$\mathrm{AE}_{\hat{p}^*}$', 'seagreen',  False, 0.0),
        (mse_vals, r'$\overline{\mathrm{MSE}}$',  'crimson',   False, None),
    ]

    for ax, (y, ylabel, color, log_scale, ref) in zip(axes, metrics):
        ax.plot(n_avgs, y, '-', color=color, lw=1.5, alpha=0.6, zorder=2)
        for xi, yi, struct in zip(n_avgs, y, structures):
            ax.plot(xi, yi, marker=struct_to_marker[struct],
                    color=color, ms=7, ls='none', zorder=3)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3, zorder=0)
        if log_scale:
            ax.set_yscale('log')
        if ref is not None:
            ax.axhline(ref, color='gray', ls='--', lw=1, zorder=2)

    axes[-1].set_xlabel('Ensemble averages $n_{\mathrm{avg}}$', fontsize=11)
    axes[-1].set_xticks(n_avgs)
    axes[-1].set_xticklabels([str(n) for n in n_avgs])

    legend_handles = [
        Line2D([0], [0], marker=struct_to_marker[s], color='dimgray',
               ls='none', ms=7, label=fmt_struct(s))
        for s in unique_structs
    ]
    axes[0].legend(handles=legend_handles, title='Identified structure',
                   fontsize=8, title_fontsize=8, loc='lower right', framealpha=0.85)

    plt.tight_layout()
    sens_plot_path = os.path.join(results_dir, 'sensitivity_analysis_3metrics.png')
    fig.savefig(sens_plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Sensitivity plot saved to: {sens_plot_path}")

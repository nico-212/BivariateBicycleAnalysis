"""Chapter 5 figures (A-E) built from final_results.csv.

    A / Fig 5.1  logical error rate per cycle over the p sweep, both codes
    B / Fig 5.2  static vs GNN posterior at matched settings: rate and time bars
    C / Fig 5.3  paired joint-vs-split graph comparison
    D / Fig 5.4  BP convergence (both bases) by iteration cap
    E / Fig 5.5  cost plane: amortized time per shot vs failure rate

Run from refactored_project/ (reads results.csv, writes Fig*.pdf).
Each plotted point is pinned to ONE run: seed 0, an explicit backend/layer, and,
where several runs share a configuration, the one with the largest shot budget
(the shorter same-seed runs are prefixes of it).  Circuit-sampling twins
(run_id contains "_circ_") and the paired joint/split experiment are excluded.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Wilson score interval without a statsmodels dependency (scipy ships with the
# venv; falls back to the 95% z-value if scipy is unavailable).
try:
    from scipy.stats import norm
    def _z_score(alpha=0.05):
        return float(norm.ppf(1.0 - alpha / 2.0))
except Exception:
    def _z_score(alpha=0.05):
        return 1.959963984540054

# --- Formatting ---
plt.rcParams.update({
    'font.family': 'serif',
    'axes.labelsize': 11,
    'font.size': 11,
    'legend.fontsize': 9,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'pdf.fonttype': 42
})

C_STATIC = '#d62728'
C_GNN = '#1f77b4'
C_GNN_DARK = '#08306b'
C_BASE = '#333333'
C_INF = '#FFB000'
C_PRIORS = '#7f7f7f'


def _wilson_interval(count, nobs, alpha=0.05):
    """Wilson score interval for a binomial proportion (equals
    statsmodels' proportion_confint(method='wilson'))."""
    if nobs <= 0:
        return 0.0, 1.0
    p = count / nobs
    z = _z_score(alpha)
    denom = 1.0 + z * z / nobs
    center = (p + z * z / (2.0 * nobs)) / denom
    half = z * np.sqrt(p * (1.0 - p) / nobs + z * z / (4.0 * nobs * nobs)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _pooled(rows):
    """Pool a group of runs into one failure rate with Wilson bounds [%]."""
    if rows.empty:
        return np.nan, 0.0, 0.0
    count = rows['fail_union'].sum()
    nobs = rows['shots'].sum()
    if nobs == 0 or pd.isna(count):
        return np.nan, 0.0, 0.0
    val = count / nobs
    lo, hi = _wilson_interval(count, nobs)
    return val * 100.0, (val - lo) * 100.0, (hi - val) * 100.0




def _one_per_p(rows):
    """One deterministic run per p: the run with the largest shot budget.

    Rows with physically impossible values are dropped first (guard against a
    misaligned export).  The extended runs (e.g. the n=40000/100000 baselines)
    are same-seed supersets of the shorter runs, so the largest budget is the
    most data for that configuration; ties go to the stable row order.
    """
    rows = rows[rows['p'].notna() & rows['pL_cycle_union'].notna()]
    rows = rows[(rows['pL_cycle_union'] <= 1.0) &
                (rows['fail_union'].notna()) &
                (rows['fail_union'] <= rows['shots'])]
    picked = []
    for _, g in rows.groupby('p', sort=True):
        picked.append(g.sort_values('shots', ascending=False).head(1))
    if not picked:
        return rows.iloc[0:0]
    return pd.concat(picked).sort_values('p')


def load_data(filepath='results.csv'):
    df = pd.read_csv(filepath)
    # Drop any junk/unnamed columns a malformed export may have introduced.
    df = df.loc[:, [c for c in df.columns if not str(c).startswith('Unnamed')]]
    for col in ['gnn_ms_per_shot', 'decode_ms_mean', 'wall_total_s', 'shots',
                'fail_union', 'fail_z', 'fail_x', 'p', 'cycles', 'conv_both']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df['gnn_ms_per_shot'] = df['gnn_ms_per_shot'].fillna(0)
    df['total_ms_per_shot'] = df['gnn_ms_per_shot'] + df['decode_ms_mean']
    # Amortized wall-clock time per shot, GNN forward pass included: this is
    # the "ms/shot" quoted in the chapter (matches Table 5.2).
    df['wall_ms_per_shot'] = df['wall_total_s'] * 1000.0 / df['shots']
    # Seed 0 is the number quoted in the chapter; the seed-1 replicates are the
    # stability control, so drop them here.  One config then maps to one run.
    if 'train_seed' in df.columns:
        df = df[pd.to_numeric(df['train_seed'], errors='coerce') == 0].copy()
    # The circuit-sampling ablation twins (run_id contains "_circ_") use a
    # different noise sampler than the DEM-sampled thesis runs; keep them out.
    df = df[~df['run_id'].astype(str).str.contains('_circ_', na=False)].copy()
    # Guard: no misaligned export rows may survive into the master.
    if 'pL_cycle_union' in df.columns:
        bad = int((df['pL_cycle_union'] > 1.0).sum())
        if bad:
            print(f"WARNING: {bad} row(s) with pL_cycle_union > 1 (misaligned)")
    return df


def get_wilson_intervals(row, is_per_cycle=False, cycles=None):
    """Calculates Wilson score intervals. Applies per-cycle transform if needed."""
    count = row['fail_union']
    nobs = row['shots']
    if pd.isna(count) or pd.isna(nobs) or nobs == 0:
        return 0, 0

    lo, hi = _wilson_interval(count, nobs)

    if is_per_cycle and cycles:
        val = 1 - (1 - count/nobs)**(1/cycles)
        lo = 1 - (1 - lo)**(1/cycles)
        hi = 1 - (1 - hi)**(1/cycles)
        return val - lo, hi - val
    return lo, hi


# ==========================================
# FIGURE A / 5.1: The Sweep
# ==========================================
def plot_figure_a_sweep(df):
    """Original proportions (12 in canvas, two panels), with the text and the
    markers scaled so they print at a readable size: the file is included at
    \\linewidth, so drawing large and scaling the text is what keeps the layout
    unchanged.  The legend sits outside, under the panels.  Marker shapes
    follow the cost plane (Figure 5.5): filled is OSD-3, hollow is OSD-0."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), sharey=True)

    for ax, code in zip(axes, [72, 144]):
        hdim = 16 if code == 72 else 64
        # The sweep uses one source family; the paired joint/split experiment
        # (results_pair.csv) is a separate run set and must not leak in here.
        d_code = df[(df['code_n'] == code) & (df['source_file'] != 'results_pair.csv')]

        configs = [
            ('Baseline (10k it.)',      C_BASE,     '-',  '*', 11.0, True,
             d_code[(d_code['decoder'] == 'bposd') & (d_code['backend'] == 'osd') & (d_code['osd_order'] == 7) & (d_code['max_iter'] == 10000)]),
            ('Reduced cap (1k it.)',    C_BASE,     '--', 'p', 9.0, True,
             d_code[(d_code['decoder'] == 'bposd') & (d_code['backend'] == 'osd') & (d_code['osd_order'] == 7) & (d_code['max_iter'] == 1000)]),
            ('Static (OSD-3, 100 it.)', C_STATIC,   '--', 'o', 8.5, True,
             d_code[(d_code['decoder'] == 'bposd') & (d_code['backend'] == 'osd') & (d_code['osd_order'] == 3) & (d_code['max_iter'] == 100)]),
            ('GNN (OSD-0, 100 it.)',    '#7fb3d5',  '--', '^', 8.5, False,
             d_code[(d_code['decoder'] == 'gnn') & (d_code['backend'] == 'osd') & (d_code['gnn_layers'] == 4) & (d_code['gnn_hdim'] == hdim) & (d_code['osd_order'] == 0) & (d_code['max_iter'] == 100)]),
            ('GNN (OSD-3, 100 it.)',    C_GNN_DARK, '-',  '^', 8.5, True,
             d_code[(d_code['decoder'] == 'gnn') & (d_code['backend'] == 'osd') & (d_code['gnn_layers'] == 4) & (d_code['gnn_hdim'] == hdim) & (d_code['osd_order'] == 3) & (d_code['max_iter'] == 100)]),
            ('Relay-BP',                'green',    '--', 'D', 7.5, True,
             d_code[d_code['backend'] == 'relay']),
        ]

        for label, color, ls, marker, ms, filled, rows in configs:
            rows = _one_per_p(rows)
            rows = rows[rows['fail_union'] > 1]
            if rows.empty:
                continue
            yerr = np.array([get_wilson_intervals(r, True, r['cycles'])
                             for _, r in rows.iterrows()]).T
            ax.errorbar(rows['p'], rows['pL_cycle_union'], yerr=yerr,
                        fmt=marker, ls=ls, color=color, label=label, capsize=5.5, lw=2.2,
                        markersize=ms, mfc=color if filled else 'none', mec=color)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Physical error rate $p$', fontsize=17)
        ax.set_title(f'BB{code} logical error rate per cycle', fontsize=17)
        ax.tick_params(labelsize=15)
        if code == 72:
            ax.set_ylabel('Logical error rate per cycle ($p_L$)', fontsize=17)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=16)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig('FigA_Sweep.pdf')


# ==========================================
# FIGURE B / 5.2: Matched Settings Bar Chart
# ==========================================
def plot_figure_b_matched(df):
    """Static vs GNN posterior at identical settings: failure rate and time per
    shot, BB144 at p=0.004 with 5000 shots, OSD orders 0 and 3 at 100 it."""
    d_144 = df[(df['code_n'] == 144) & (df['p'] == 0.004) &
               (df['max_iter'] == 100) & (df['shots'] == 5000)]

    groups = [
        ('Static (BP-OSD)', C_STATIC,
         [d_144[(d_144['decoder'] == 'bposd') & (d_144['backend'] == 'osd') & (d_144['osd_order'] == o)] for o in (0, 3)]),
        ('GNN + BP-OSD ($h$=64)', C_GNN,
         [d_144[(d_144['decoder'] == 'gnn') & (d_144['backend'] == 'osd') & (d_144['gnn_layers'] == 4) & (d_144['gnn_hdim'] == 64) & (d_144['osd_order'] == o)] for o in (0, 3)]),
    ]

    x = np.arange(2)
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, metric in zip(axes, ['rate', 'time']):
        for i, (label, color, masks) in enumerate(groups):
            vals, errs = [], []
            for rows in masks:
                if metric == 'rate':
                    v, lo, hi = _pooled(rows)
                    vals.append(v)
                    errs.append(max(lo, hi))
                else:
                    vals.append(rows['wall_ms_per_shot'].mean() if not rows.empty else np.nan)
                    errs.append(0.0)
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, vals, width, yerr=errs, capsize=4,
                          color=color, edgecolor='white', label=label)
            for rect, v in zip(bars, vals):
                if pd.notna(v):
                    ax.annotate(f'{v:.2f}' if metric == 'rate' else f'{v:.1f}',
                                (rect.get_x() + rect.get_width() / 2, v),
                                textcoords='offset points', xytext=(0, 3),
                                ha='center', fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(['OSD-0', 'OSD-3'])
        ax.set_title('Failure rate' if metric == 'rate' else 'Amortized time')
        ax.set_ylabel('Failure rate per shot [%]' if metric == 'rate'
                      else 'Time per shot [ms]')

    axes[0].legend()
    fig.suptitle('BB144 matched settings ($p$=0.004, 100 it., 5000 shots)', fontsize=11)
    plt.tight_layout()
    plt.savefig('FigB_Matched.pdf')


# ==========================================
# FIGURE C / 5.3: Joint vs Split Graph
# ==========================================
def plot_figure_c_joint_split(df):
    """Joint, split and static priors at the operating point, OSD-3 at 100 it.

    The joint and the split arm are the 20000-shot paired runs; the static
    control is the 5000-shot sweep run behind Table 5.2.
    """
    panels = [
        ('BB144, $p$=0.004', 144, 0.004, 64),
        ('BB72, $p$=0.004', 72, 0.004, 16),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(panels))
    width = 0.27

    series = [
        ('Joint graph', C_GNN, 'joint'),
        ('Split graph', C_STATIC, 'split'),
        ('Static priors', C_PRIORS, 'static'),
    ]
    ymax = 0.0

    for i, (label, color, arm) in enumerate(series):
        vals, errs = [], []
        for _, code, p, hdim in panels:
            if arm == 'static':
                rows = df[(df['code_n'] == code) & (df['p'] == p) &
                          (df['decoder'] == 'bposd') & (df['osd_order'] == 3) &
                          (df['max_iter'] == 100)]
            else:
                rows = df[(df['code_n'] == code) & (df['p'] == p) & (df['decoder'] == 'gnn') &
                          (df['backend'] == 'osd') & (df['gnn_layers'] == 4) &
                          (df['gnn_hdim'] == hdim) & (df['osd_order'] == 3) &
                          (df['max_iter'] == 100) & (df['shots'] == 20000)]
                is_split = rows['run_id'].astype(str).str.contains('split')
                rows = rows[is_split] if arm == 'split' else rows[~is_split]
            v, lo, hi = _pooled(rows)
            vals.append(v)
            errs.append(max(lo, hi))

        offset = (i - 1) * width
        bars = ax.bar(x + offset, vals, width, yerr=errs, capsize=4,
                      color=color, edgecolor='white', label=label)
        for rect, v, e in zip(bars, vals, errs):
            if pd.notna(v):
                ymax = max(ymax, v + e)
                ax.annotate(f'{v:.2f}%', (rect.get_x() + rect.get_width() / 2, v + e),
                            textcoords='offset points', xytext=(0, 3),
                            ha='center', va='bottom', fontsize=8)

    ax.set_ylim(0, ymax * 1.12 if ymax > 0 else 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in panels])
    ax.set_ylabel('Failure rate per shot [%]')
    ax.set_title('Joint, split and static priors, OSD-3 at 100 it.')
    ax.legend()
    plt.tight_layout()
    plt.savefig('FigC_JointSplit.pdf')


def plot_figure_c_joint_split_v2(df):
    """Full joint-vs-split sweep: both codes, every rate, both arms.

    OSD-3 at 100 iterations, min-sum scaling 0, matched shots per arm and rate.
    Per-shot union failure rate with Wilson 95% intervals; points backed by fewer
    than five failures are omitted (thesis convention), y clipped to 0.01-100%.
    """
    from matplotlib.ticker import FuncFormatter

    panels = [(72, 16, r'[[72,12,6]]'), (144, 64, r'[[144,12,12]]')]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (code, hdim, title) in zip(axes, panels):
        base = df[(df['source_file'] == 'results_pair.csv') &
                  (df['code_n'] == code) & (df['decoder'] == 'gnn') &
                  (df['backend'] == 'osd') & (df['gnn_layers'] == 4) &
                  (df['gnn_hdim'] == hdim) & (df['osd_order'] == 3) &
                  (df['max_iter'] == 100)]
        split_mask = base['run_id'].astype(str).str.contains('split')

        for label, color, marker, sel in [('Joint graph', C_GNN, 'o', ~split_mask),
                                           ('Split graph', C_STATIC, 's', split_mask)]:
            rows = base[sel].sort_values('p')
            rows = rows[rows['fail_union'] >= 5]   # <5 failures: not informative
            if rows.empty:
                continue
            x = rows['p'].to_numpy(float)
            cnt = rows['fail_union'].to_numpy(float)
            nobs = rows['shots'].to_numpy(float)
            rate = cnt / nobs * 100.0
            pos = cnt > 0

            err = np.zeros((2, int(pos.sum())))
            for i, (c, k) in enumerate(zip(cnt[pos], nobs[pos])):
                lo, hi = _wilson_interval(c, k)
                err[0, i] = rate[pos][i] - lo * 100.0
                err[1, i] = hi * 100.0 - rate[pos][i]
            ax.errorbar(x[pos], rate[pos], yerr=np.abs(err), fmt=marker + '-',
                        color=color, label=label, capsize=3, markersize=6, zorder=4)

            for xi, c, k in zip(x[~pos], cnt[~pos], nobs[~pos]):
                _, hi = _wilson_interval(c, k)
                ax.plot([xi], [hi * 100.0], marker=marker, ls='none', mfc='white',
                        mec=color, mew=1.5, ms=6, zorder=4)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(0.01, 100.0)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}" if v > 0 else ""))
        ax.set_xlabel('Physical error rate $p$')
        ax.set_title(f'{title} joint vs split graph')
        ax.grid(True, alpha=0.3)
        if code == 72:
            ax.set_ylabel('Failure rate per shot [%]')
            ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('FigC_JointSplit_v2.pdf')


def plot_figure_c_joint_split_v3(df):
    """Clarity-first joint-vs-split comparison.

    Top row:    absolute per-shot failure rate, log y with decimal percent
                labels (0.001 ... 100) and a tight range, Wilson intervals.
    Bottom row: the split/joint ratio on a linear axis -- the quantity the
                comparison is actually about (1x = no penalty).
    """
    from matplotlib.ticker import FuncFormatter

    def dec_pct(v, _):
        return f"{v:g}" if v > 0 else ""

    panels = [(72, 16, r'[[72,12,6]]'), (144, 64, r'[[144,12,12]]')]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for col, (code, hdim, title) in enumerate(panels):
        base = df[(df['source_file'] == 'results_pair.csv') &
                  (df['code_n'] == code) & (df['decoder'] == 'gnn') &
                  (df['backend'] == 'osd') & (df['gnn_layers'] == 4) &
                  (df['gnn_hdim'] == hdim) & (df['osd_order'] == 3) &
                  (df['max_iter'] == 100)]
        split_mask = base['run_id'].astype(str).str.contains('split')
        ax = axes[0, col]

        arms = {}
        for label, color, marker, sel in [('Joint graph', C_GNN, 'o', ~split_mask),
                                          ('Split graph', C_STATIC, 's', split_mask)]:
            rows = base[sel].sort_values('p')
            arms[label] = rows
            if rows.empty:
                continue
            x = rows['p'].to_numpy(float)
            cnt = rows['fail_union'].to_numpy(float)
            nobs = rows['shots'].to_numpy(float)
            rate = cnt / nobs * 100.0
            pos = cnt > 0
            err = np.zeros((2, int(pos.sum())))
            for i, (c, k) in enumerate(zip(cnt[pos], nobs[pos])):
                lo, hi = _wilson_interval(c, k)
                err[0, i] = rate[pos][i] - lo * 100.0
                err[1, i] = hi * 100.0 - rate[pos][i]
            ax.errorbar(x[pos], rate[pos], yerr=np.abs(err), fmt=marker + '-',
                        color=color, label=label, capsize=3, markersize=6, zorder=4)
            for xi, c, k in zip(x[~pos], cnt[~pos], nobs[~pos]):
                _, hi = _wilson_interval(c, k)
                ax.plot([xi], [hi * 100.0], marker=marker, ls='none', mfc='white',
                        mec=color, mew=1.5, ms=6, zorder=4)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(1e-3, 150.0)
        ax.yaxis.set_major_formatter(FuncFormatter(dec_pct))
        ax.set_xlabel('Physical error rate $p$')
        ax.set_title(f'{title}: failure rate per shot')
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel('Failure rate per shot [%]')
            ax.legend(loc='lower right')

        # --- ratio of the two arms (linear) ---
        j = {float(r.p): (float(r.fail_union), float(r.shots)) for r in arms['Joint graph'].itertuples()}
        s = {float(r.p): (float(r.fail_union), float(r.shots)) for r in arms['Split graph'].itertuples()}
        xs, ys = [], []
        for p in sorted(set(j) & set(s)):
            cj, nj = j[p]
            cs, ns = s[p]
            if cj == 0:
                continue
            xs.append(p)
            ys.append((cs / ns) / (cj / nj))

        ax2 = axes[1, col]
        ax2.plot(xs, ys, 'o-', color='#333333', markersize=6)
        for xi, yi in zip(xs, ys):
            ax2.annotate(f'{yi:.1f}x', (xi, yi), textcoords='offset points',
                         xytext=(0, 6), ha='center', fontsize=8)
        ax2.axhline(1.0, color='grey', ls='--', lw=1)
        ax2.set_xscale('log')
        ax2.set_ylim(0, max(ys) * 1.25 if ys else 1.0)
        ax2.set_xlabel('Physical error rate $p$')
        ax2.set_ylabel('Split / joint failure rate')
        ax2.set_title(f'{title}: cost of splitting the graph')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('FigC_JointSplit_v3.pdf')


# ==========================================
# FIGURE D / 5.4: BP Convergence
# ==========================================
def plot_figure_d_convergence(df):
    """Share of shots where BP settles on both bases, by iteration cap:
    static priors vs the GNN posterior (h=64) plus the OSD-7 baseline alone."""
    d = df[(df['code_n'] == 144) & (df['p'] == 0.004)]

    static = {it: d[(d['decoder'] == 'bposd') & (d['backend'] == 'osd') & (d['osd_order'] == 0) & (d['max_iter'] == it)]
              for it in (10, 30, 100)}
    gnn = {it: d[(d['decoder'] == 'gnn') & (d['backend'] == 'osd') & (d['gnn_layers'] == 4) & (d['gnn_hdim'] == 64) &
                 (d['osd_order'] == 0) & (d['max_iter'] == it)]
           for it in (10, 30, 100)}
    baseline = {it: d[(d['decoder'] == 'bposd') & (d['backend'] == 'osd') &
                     (d['osd_order'] == 7) & (d['max_iter'] == it)]
                for it in (1000, 10000)}

    def conv_pct(rows):
        if rows.empty:
            return np.nan
        return rows['conv_both'].sum() / rows['shots'].sum() * 100.0

    paired_caps = [10, 30, 100]
    x_paired = np.arange(len(paired_caps))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 4.5))

    static_vals = [conv_pct(static[it]) for it in paired_caps]
    gnn_vals = [conv_pct(gnn[it]) for it in paired_caps]
    base_vals = [conv_pct(baseline[it]) for it in (1000, 10000)]

    b1 = ax.bar(x_paired - width / 2, static_vals, width, color=C_STATIC,
                edgecolor='white', label='Static priors')
    b2 = ax.bar(x_paired + width / 2, gnn_vals, width, color=C_GNN,
                edgecolor='white', label='GNN posterior ($h$=64)')
    x_single = np.array([len(paired_caps) + 0.6, len(paired_caps) + 1.4])
    b3 = ax.bar(x_single, base_vals, 0.5, color=C_BASE, edgecolor='white',
                label='Baseline (OSD-7)')

    for bars, vals in ((b1, static_vals), (b2, gnn_vals), (b3, base_vals)):
        for rect, v in zip(bars, vals):
            if pd.notna(v):
                ax.annotate(f'{v:.1f}', (rect.get_x() + rect.get_width() / 2, v),
                            textcoords='offset points', xytext=(0, 3),
                            ha='center', fontsize=8)

    ax.set_xticks(list(x_paired) + list(x_single))
    ax.set_xticklabels(['10', '30', '100', '1000', '10000'])
    ax.set_xlabel('BP iteration cap')
    ax.set_ylabel('Shots converged on both bases [%]')
    ax.set_title('BP convergence, BB144 at $p$=0.004')
    ax.legend()
    plt.tight_layout()
    plt.savefig('FigD_Convergence.pdf')


# ==========================================
# FIGURE E / 5.5: Pareto Front
# ==========================================
def plot_figure_e_pareto(df):
    """Cost plane: amortized time per shot against failure rate for every BB144
    configuration at p=0.004 (5000 shots).  Lower left is better.

    The points are separate configurations (cap x OSD order), not a trajectory,
    so markers only: filled = OSD-3, hollow = OSD-0, colour = decoder family.
    A thin step line marks the Pareto frontier over all configurations.
    """
    from matplotlib.lines import Line2D

    d = df[(df['code_n'] == 144) & (df['p'] == 0.004) & (df['shots'] == 5000)]

    caps = [10, 30, 100]
    families = [
        ('Static (BP-OSD)', C_STATIC, 'o', '-',
         d[(d['decoder'] == 'bposd') & (d['backend'] == 'osd') &
           (d['gnn_hdim'] == 0) & (d['max_iter'].isin(caps)) & (d['osd_order'].isin([0, 3]))]),
        ('GNN, $h$=32', C_GNN, 's', '-',
         d[(d['decoder'] == 'gnn') & (d['backend'] == 'osd') & (d['gnn_layers'] == 4) & (d['gnn_hdim'] == 32) &
           (d['max_iter'].isin(caps)) & (d['osd_order'].isin([0, 3]))]),
        ('GNN, $h$=64', C_GNN_DARK, '^', '-',
         d[(d['decoder'] == 'gnn') & (d['backend'] == 'osd') & (d['gnn_layers'] == 4) & (d['gnn_hdim'] == 64) &
           (d['max_iter'].isin(caps)) & (d['osd_order'].isin([0, 3]))]),
    ]

    fig, ax = plt.subplots(figsize=(8, 6))

    pts = []
    for label, color, marker, ls, rows in families:
        rows = rows[rows['wall_ms_per_shot'].notna() & rows['pL_shot_union'].notna()]
        # one point per (cap, OSD order): prefer the largest shot budget
        rows = (rows.sort_values('shots', ascending=False)
                    .drop_duplicates(['max_iter', 'osd_order'], keep='first'))
        for _, r in rows.iterrows():
            x, y = r['wall_ms_per_shot'], r['pL_shot_union'] * 100.0
            ax.plot([x], [y], marker=marker, ls='none', color=color,
                    markerfacecolor=(color if int(r['osd_order']) == 3 else 'white'),
                    markeredgecolor=color, markersize=8, zorder=4)
            pts.append((x, y))

    # Single reference configurations
    singles = [
        ('Baseline 10k it.', 'bposd', 7, 10000, 0, C_BASE, '*', 150),
        ('Baseline 1k it.', 'bposd', 7, 1000, 0, C_BASE, '*', 90),
        ('Relay-BP', 'bposd', 7, 1000, 0, 'green', 'D', 70),
    ]
    for label, dec, osd, it, hdim, color, marker, size in singles:
        rows = d[(d['decoder'] == dec) & (d['osd_order'] == osd) &
                 (d['max_iter'] == it) & (d['backend'] == ('relay' if label == 'Relay-BP' else 'osd'))]
        if rows.empty:
            continue
        r = rows.iloc[0]
        x, y = r['wall_ms_per_shot'], r['pL_shot_union'] * 100.0
        ax.scatter([x], [y], marker=marker, color=color, s=size, zorder=5)
        pts.append((x, y))

    # Pareto frontier over all configurations (min time, min rate)
    frontier, best = [], np.inf
    for x, y in sorted(pts):
        if y < best - 1e-12:
            frontier.append((x, y))
            best = y
    if frontier:
        ax.step([p[0] for p in frontier], [p[1] for p in frontier], where='post',
                color='0.45', ls='--', lw=1.0, zorder=3)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Amortized time per shot [ms]')
    ax.set_ylabel('Failure rate per shot [%]')
    ax.set_title('BB144 cost vs accuracy at $p$=0.004 (5000 shots)')
    handles = [
        Line2D([], [], marker='o', ls='none', color=C_STATIC, label='Static (BP-OSD)'),
        Line2D([], [], marker='s', ls='none', color=C_GNN, label='GNN, $h$=32'),
        Line2D([], [], marker='^', ls='none', color=C_GNN_DARK, label='GNN, $h$=64'),
        Line2D([], [], marker='o', ls='none', markerfacecolor='none',
               markeredgecolor='0.3', label='OSD-0 (hollow)'),
        Line2D([], [], marker='o', ls='none', color='0.3', label='OSD-3 (filled)'),
        Line2D([], [], marker='*', ls='none', color=C_BASE, markersize=12, label='Baseline'),
        Line2D([], [], marker='D', ls='none', color='green', label='Relay-BP'),
        Line2D([], [], ls='--', color='0.45', label='Pareto frontier'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig('FigE_Pareto.pdf')


def plot_figure_f_latency_lines(df):
    """Where the per-shot time goes across the sweep (BB72 / BB144).

    For a matched GNN setting (h=16/64, OSD-3, 100 iterations) the amortized
    decode time (total minus the forward pass) and the GNN forward pass are
    plotted against p, with the BP-OSD-7 baseline.  Log-log: the forward pass is
    a flat per-shot floor, while the decode time rises with p, so the forward
    pass share collapses from ~90% at p=0.001 to ~10% at p=0.007 (BB144).
    """
    panels = [(72, 16, r'[[72,12,6]]'), (144, 64, r'[[144,12,12]]')]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    for ax, (code, hdim, title) in zip(axes, panels):
        d = df[(df['code_n'] == code) & (df['source_file'] != 'results_pair.csv')]
        g = _one_per_p(d[(d['decoder'] == 'gnn') & (d['backend'] == 'osd') &
                         (d['gnn_layers'] == 4) & (d['gnn_hdim'] == hdim) &
                         (d['osd_order'] == 3) & (d['max_iter'] == 100)])
        b = _one_per_p(d[(d['decoder'] == 'bposd') & (d['backend'] == 'osd') &
                         (d['osd_order'] == 7) & (d['max_iter'] == 10000)])
        if g.empty:
            continue

        gp = g['p'].to_numpy(float)
        total = g['wall_ms_per_shot'].to_numpy(float)
        fwd = g['gnn_ms_per_shot'].fillna(0).to_numpy(float)
        dec = total - fwd

        ax.fill_between(gp, dec, total, color=C_INF, alpha=0.25, lw=0, zorder=2)
        ax.plot(gp, total, 'o-', color=C_GNN_DARK, label='GNN total', zorder=4)
        ax.plot(gp, dec, 's--', color=C_GNN, label='GNN decode', zorder=4)
        ax.plot(gp, fwd, '^:', color=C_INF, label='GNN forward pass', zorder=4)
        if not b.empty:
            ax.plot(b['p'].to_numpy(float), b['wall_ms_per_shot'].to_numpy(float),
                    'D-', color=C_BASE, label='Baseline (BP-OSD-7, 10k it.)', zorder=3)

        idx = np.where(np.isclose(gp, 0.004))[0]
        if idx.size and total[idx[0]] > 0:
            i = idx[0]
            ax.annotate(f'forward pass\n{100 * fwd[i] / total[i]:.0f}% of total',
                        (gp[i], total[i]), textcoords='offset points',
                        xytext=(8, 8), fontsize=8, color=C_INF)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Physical error rate $p$')
        ax.set_title(f'{title}: time per shot')
        ax.grid(True, which='both', alpha=0.25)
        if code == 72:
            ax.set_ylabel('Amortized time per shot [ms]')
            ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig('FigF_Latency_lines.pdf')


# --- Figure F palette (matches the original figures_new/fig2_latency.pdf) ---
FIGF_OSD0 = '#1f77b4'   # blue    -- GNN posterior, OSD order 0
FIGF_OSD3 = '#d62728'   # red     -- GNN posterior, OSD order 3
FIGF_BASE = '#555555'   # gray    -- BP-OSD-7 baseline (static priors)
FIGF_INF = '#FFB000'    # amber   -- GNN inference (forward pass)
FIGF_GNN_IT = 100        # iteration cap for the two GNN bars
# Baseline iteration cap.  1000 matches the "Reduced cap (1,000 it)" arm of
# Table 5.2 and, unlike the 10k arm, fits a linear axis next to the GNN bars
# (10k reaches 3678 ms/shot on BB144 against a 135 ms GNN bar, which forces a
# log axis or a clipped axis).  The 10k numbers stay in Table 5.2.
FIGF_BASE_IT = 1000


def _figf_groups(df, code, hdim):
    """Per-p rows for GNN OSD-0, GNN OSD-3 and the BP-OSD-7 (10k it) baseline.

    NOTE: the code_n filter is required.  Without it the baseline pool merges
    both codes and _one_per_p picks the largest-shot run per p from either,
    which scrambles the gray series (BB72/BB144 values interleaved, and no
    longer monotonic in p).  The GNN series are keyed by gnn_hdim, which happens
    to be code-specific (16 -> BB72, 64 -> BB144), but the code filter is
    applied there too so the key is explicit.
    """
    d = df[(df['code_n'] == code) & (df['source_file'] != 'results_pair.csv')]
    g = {}
    for osd in (0, 3):
        g[osd] = _one_per_p(d[(d['decoder'] == 'gnn') & (d['backend'] == 'osd') &
                              (d['gnn_layers'] == 4) & (d['gnn_hdim'] == hdim) &
                              (d['osd_order'] == osd) & (d['max_iter'] == FIGF_GNN_IT)])
    b = _one_per_p(d[(d['decoder'] == 'bposd') & (d['backend'] == 'osd') &
                     (d['osd_order'] == 7) & (d['max_iter'] == FIGF_BASE_IT)])
    return g, b


def _figf_inference_floor(g):
    """Constant per-shot GNN forward-pass floor for one code.

    gnn_ms_per_shot is gnn_s / shots, so every run carries a fixed per-run
    start-up (~1.2 s on BB72) amortised over its shot budget; on the short runs
    that doubles the apparent per-shot inference (BB72: 1.28 ms at 30k shots,
    2.43 ms at 1k).  The forward pass itself is a fixed cost per shot and does
    not depend on p or on the OSD order, so the plotted floor is the per-shot
    inference at the code's largest shot budget, averaged over the two orders.
    """
    parts = [s for s in (g.get(0), g.get(3)) if s is not None and not s.empty]
    if not parts:
        return 0.0, 0
    cat = pd.concat(parts)
    nmax = int(cat['shots'].max())
    top = cat[(cat['shots'] == nmax) & cat['gnn_ms_per_shot'].notna()]
    return float(top['gnn_ms_per_shot'].mean()), nmax


def _plot_latency_grouped(df, logy, fname):
    """Grouped stacked latency bars over the p sweep, one group per p:
    GNN OSD-0 (blue) and GNN OSD-3 (red), each split into decode time and the
    amber GNN forward pass, plus the gray BP-OSD-7 baseline.  With the 10k-it
    baseline the linear axis is unusable (BB144 reaches ~3.7 s/shot at p=0.007),
    so logy=True is the legible default; logy=False clips to the GNN range and
    labels the off-scale baselines.
    """
    from matplotlib.patches import Patch

    panels = [(72, 16), (144, 64)]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.9))
    bw, offs = 0.26, {0: -0.28, 3: 0.0, 'b': 0.28}

    # Legend below the panels, as in Figure 5.1.  Inside a panel the four
    # entries cover the bars at the low-p end of the sweep.  The figure box is
    # kept at its old height and the legend takes a strip of it, so the printed
    # figure keeps the same size on the page.  matplotlib fills legend columns
    # column-major, so the list order here is chosen to pair the two GNN bars on
    # the first row and the forward pass with the baseline on the second.
    handles = [Patch(facecolor=FIGF_OSD0, edgecolor='white',
                     label=f'GNN, OSD-0 ({FIGF_GNN_IT} it.)'),
               Patch(facecolor=FIGF_INF, edgecolor='white', hatch='////',
                     label='GNN inference (forward pass, constant)'),
               Patch(facecolor=FIGF_OSD3, edgecolor='white',
                     label=f'GNN, OSD-3 ({FIGF_GNN_IT} it.)'),
               Patch(facecolor=FIGF_BASE, edgecolor='white',
                     label=f'Baseline, BP-OSD-7 ({FIGF_BASE_IT // 1000}k it.)')]

    for ax, (code, hdim) in zip(axes, panels):
        g, b = _figf_groups(df, code, hdim)
        ref = g[3] if not g[3].empty else g[0]
        ps = sorted(float(p) for p in ref['p'])
        idx = {round(p, 4): i for i, p in enumerate(ps)}
        x = np.arange(len(ps))
        gnn_hi, base_hi = 0.0, 0.0
        fwd_floor, floor_n = _figf_inference_floor(g)

        for osd, col in ((0, FIGF_OSD0), (3, FIGF_OSD3)):
            for _, r in g[osd].iterrows():
                k = round(float(r['p']), 4)
                if k not in idx:
                    continue
                total = float(r['wall_ms_per_shot'])
                # Split the measured per-shot total into the constant forward-pass
                # floor and the remainder (decode work + start-up amortisation).
                fwd = fwd_floor if (fwd_floor > 0 and total > fwd_floor) else 0.0
                dec = max(total - fwd, 0.0)
                ax.bar(x[idx[k]] + offs[osd], dec, bw, bottom=(1e-2 if logy else 0.0),
                       color=col, edgecolor='white', linewidth=0.6, zorder=3)
                if fwd > 0:
                    ax.bar(x[idx[k]] + offs[osd], fwd, bw, bottom=dec, color=FIGF_INF,
                           edgecolor='white', linewidth=0.6, hatch='////', zorder=4)
                gnn_hi = max(gnn_hi, total)

        for _, r in b.iterrows():
            k = round(float(r['p']), 4)
            if k not in idx:
                continue
            v = float(r['wall_ms_per_shot'])
            ax.bar(x[idx[k]] + offs['b'], v, bw, bottom=(1e-2 if logy else 0.0),
                   color=FIGF_BASE, edgecolor='white', linewidth=0.6, zorder=3)
            base_hi = max(base_hi, v)

        # Both series share one axis, so the taller of the two sets the range.
        # With the 1,000-it baseline this leaves every bar on scale (BB144:
        # max 385 ms/shot against a 135 ms GNN bar); nothing is clipped.
        data_hi = max(gnn_hi, base_hi)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{p * 1000:.0f}' for p in ps])
        ax.set_xlabel(r'Physical error rate, $p$ ($\times 10^{-3}$)')
        if logy:
            ax.set_yscale('log')
            ax.set_ylim(1e-2, data_hi * 2.5)
        else:
            ytop = data_hi * 1.10
            ax.set_ylim(0.0, ytop)
            # Guard, inert with the current data: if a future baseline is put on
            # an axis that cannot hold it, the cut bar is still labelled.
            for _, r in b.iterrows():
                k = round(float(r['p']), 4)
                if k in idx and float(r['wall_ms_per_shot']) > ytop:
                    ax.annotate(f"{float(r['wall_ms_per_shot']):.0f}",
                                (x[idx[k]] + offs['b'], ytop * 0.985),
                                ha='center', va='top', fontsize=6.5, color=FIGF_BASE,
                                rotation=90, zorder=6)
        ax.grid(True, axis='y', alpha=0.3)
        if code == 72:
            ax.set_ylabel('Amortized time per shot [ms]')

    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, fontsize=14)
    plt.tight_layout(rect=(0, 0.17, 1, 1))
    plt.savefig(fname)


def plot_figure_f_latency(df):
    """Fig F: linear-y grouped bars (the thesis version).

    Linear like the original figures_new/fig2_latency.pdf.  The gray baseline
    bars that run past the top of the axis are cut and labelled with their true
    value in ms, because the 10k-it baseline grows far beyond the GNN range.
    """
    _plot_latency_grouped(df, logy=False, fname='FigF_Latency.pdf')


def plot_figure_f_latency_linear(df):
    """Alias kept for callers that ask for the linear variant by name."""
    _plot_latency_grouped(df, logy=False, fname='FigF_Latency_linear.pdf')


def plot_figure_f_latency_log(df):
    """Fig F, log-y variant: keeps the full baseline bars on scale."""
    _plot_latency_grouped(df, logy=True, fname='FigF_Latency_log.pdf')


if __name__ == "__main__":
    df = load_data('results.csv')
    plot_figure_a_sweep(df)
    plot_figure_b_matched(df)
    plot_figure_c_joint_split(df)
    plot_figure_c_joint_split_v2(df)
    plot_figure_c_joint_split_v3(df)
    plot_figure_d_convergence(df)
    plot_figure_e_pareto(df)
    plot_figure_f_latency(df)
    plot_figure_f_latency_linear(df)
    plot_figure_f_latency_log(df)
    plot_figure_f_latency_lines(df)

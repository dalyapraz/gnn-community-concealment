import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections.abc import Sequence

def plot_metric_changes(
    df: pd.DataFrame,
    metrics,
    mode: str = "grid",             # "grid" | "fixed_mu" | "fixed_sigma"
    mu_values=None,
    sigma_c_values=None,
    baseline_p=None,                # numeric; if set, filter to this p
    metric_markers=None,            # {'ECS':'o','M1':'x','M2':'^'}
    color_map=None,                 # dict {σ_c: color} or list mapped to sorted σ_c
    metric_colors=None,             # dict {metric: color} for fixed_* modes
    figsize_per_plot=7,             # size for each square subplot
    ylim=None                       # tuple for all, or dict per metric
):
    # --- Filter baseline p if given ---
    d = df.copy()
    if baseline_p is not None:
        p_num = pd.to_numeric(d['p'], errors='coerce')
        d = d[p_num == float(baseline_p)]

    # --- Infer value grids ---
    if mu_values is None:
        mu_values = sorted(d['mu'].unique())
    if sigma_c_values is None:
        sigma_c_values = sorted(d['sigma_c'].unique())
    # Labels
    labels = {
        'ECS': r"$ECS$",
        'M1': r"$M_1$",
        'M2': r"$M_2$",
        'p': r"$p$",
        'sigma_c': r"$\sigma_c$",
        'mu': r"$\mu$"
    }
    # Markers
    if metric_markers is None:
        metric_markers = {m: 'o' for m in metrics}

    # Δ helper
    def delta_of(sub, metric):
        y = (sub.groupby('b_percentage')[metric].mean().sort_index()).values
        return float(np.mean(np.diff(y))) if len(y) > 1 else None

    # --- Modes ---
    if mode == "grid":
        # Colors for σ_c
        if color_map is None:
            palette = ["#000000","#E69F00","#56B4E9","#009E73",
                       "#F0E442","#0072B2","#D55E00","#CC79A7"]
            color_map = {s: palette[i % len(palette)] for i, s in enumerate(sigma_c_values)}
        elif isinstance(color_map, Sequence) and not isinstance(color_map, (str, bytes, dict)):
            color_map = {s: list(color_map)[i % len(color_map)] for i, s in enumerate(sigma_c_values)}

        rows = []
        for metric in metrics:
            for mu in mu_values:
                for sc in sigma_c_values:
                    val = delta_of(d[(d['mu']==mu) & (d['sigma_c']==sc)], metric)
                    if val is not None:
                        rows.append({'mu':mu,'sigma_c':sc,'metric':metric,'delta':val})
        delta_df = pd.DataFrame(rows)

        fig, axes = plt.subplots(1,len(metrics),figsize=((figsize_per_plot + 1) * len(metrics), figsize_per_plot),sharex=True)
        if len(metrics)==1: axes=[axes]
        for i, metric in enumerate(metrics):
            ax = axes[i]
            for sc in sigma_c_values:
                s = delta_df[(delta_df['metric']==metric)&(delta_df['sigma_c']==sc)]
                if not s.empty:
                    ax.plot(s['mu'],s['delta'],
                            marker=metric_markers[metric],
                            color=color_map.get(sc),
                            label=fr'$\sigma_c={sc}$',linewidth=2)
            # ax.set_title(f"Average rate of change in {labels[metric]}",fontsize=14)
            ax.set_xlabel(r"$\mu$", fontsize=14)
            ax.set_ylabel(f"Average rate of change in {labels[metric]}", fontsize=14)
            if isinstance(ylim,dict) and metric in ylim: 
                ax.set_ylim(ylim[metric])
            elif isinstance(ylim,tuple): 
                ax.set_ylim(ylim)
            # if i==len(metrics)-1: 
            #     ax.legend(title=r'$\sigma_c$',bbox_to_anchor=(1.05,1),loc='upper left')
            ax.grid(False)
                # Grab handles/labels from any axis (first is fine)
        # pick an Axes to read legend entries from
        first_ax = axes.flat[0] if isinstance(axes, np.ndarray) else axes

        handles, legend_labels_mpl = first_ax.get_legend_handles_labels()

        # Only create legend if there are entries
        if handles and legend_labels_mpl:
            # (optional) dedupe labels if needed
            by_label = dict(zip(legend_labels_mpl, handles))
            handles, legend_labels_mpl = list(by_label.values()), list(by_label.keys())

            # Shared, horizontal legend centered at the top
            leg = fig.legend(
                handles, legend_labels_mpl,
                loc='upper center',
                ncol=len(legend_labels_mpl),          # left→right in one row
                frameon=True,
                fontsize=14,
                markerscale=0, 
                bbox_to_anchor=(0.47, 1.05), # exactly centered above the axes area
                bbox_transform=fig.transFigure
            )
            # Thicken ONLY legend lines
            for h in leg.legend_handles:
                try:
                    h.set_linewidth(3.0)   # <- legend line width
                    h.set_marker('None')   # ensure no marker shows
                except Exception:
                    pass
        plt.tight_layout(rect=[0,0,0.90,0.95]); plt.show()
        return delta_df

    elif mode == "fixed_mu":
        if metric_colors is None:
            metric_colors = {m:"blue" for m in metrics}
        rows=[]
        for metric in metrics:
            for mu in mu_values:
                val = delta_of(d[d['mu']==mu],metric)
                if val is not None: rows.append({'mu':mu,'metric':metric,'delta':val})
        delta_df=pd.DataFrame(rows)

        fig,axes=plt.subplots(1,len(metrics),figsize=((figsize_per_plot + 1) * len(metrics), figsize_per_plot),sharex=True)
        if len(metrics)==1: axes=[axes]
        for i,metric in enumerate(metrics):
            ax=axes[i]
            s=delta_df[delta_df['metric']==metric].sort_values('mu')
            if not s.empty:
                ax.plot(s['mu'],s['delta'],
                        marker=metric_markers[metric],
                        color=metric_colors.get(metric,"black"),
                        linewidth=2)
            # ax.set_title(fr"Average rate of change in {labels[metric]} (avg over all $\sigma_c$)",fontsize=14)
            ax.set_xlabel(r"$\mu$", fontsize=14)
            ax.set_ylabel(f"Average rate of change in {labels[metric]}", fontsize=14)
            if isinstance(ylim,dict) and metric in ylim: ax.set_ylim(ylim[metric])
            elif isinstance(ylim,tuple): ax.set_ylim(ylim)
            ax.grid(False)
        plt.tight_layout(rect=[0,0,0.90,1]); plt.show()
        return delta_df

    elif mode == "fixed_sigma":
        if metric_colors is None:
            metric_colors = {m:"green" for m in metrics}
        rows=[]
        for metric in metrics:
            for sc in sigma_c_values:
                val=delta_of(d[d['sigma_c']==sc],metric)
                if val is not None: rows.append({'sigma_c':sc,'metric':metric,'delta':val})
        delta_df=pd.DataFrame(rows)

        fig,axes=plt.subplots(1,len(metrics),figsize=((figsize_per_plot + 1) * len(metrics), figsize_per_plot),sharex=True)
        if len(metrics)==1: axes=[axes]
        for i,metric in enumerate(metrics):
            ax=axes[i]
            s=delta_df[delta_df['metric']==metric].sort_values('sigma_c')
            if not s.empty:
                ax.plot(s['sigma_c'],s['delta'],
                        marker=metric_markers[metric],
                        color=metric_colors.get(metric,"black"),
                        linewidth=2)
            # ax.set_title(fr"Average rate of change in {labels[metric]} (avg over all $\mu$)",fontsize=14)
            ax.set_xlabel(r"$\sigma_c$", fontsize=14)
            ax.set_ylabel(f"Average rate of change in {labels[metric]}", fontsize=14)
            if isinstance(ylim,dict) and metric in ylim: ax.set_ylim(ylim[metric])
            elif isinstance(ylim,tuple): ax.set_ylim(ylim)
            ax.grid(False)
        plt.tight_layout(rect=[0,0,0.90,1]); plt.show()
        return delta_df

    else:
        raise ValueError("mode must be 'grid','fixed_mu','fixed_sigma'")



def plot_results_by_p(
    folder,
    file_glob="dmon_*None.csv",
    metrics=("ECS", "M1", "M2"),
    ylim_per_metric=None,          # dict: e.g. {'ECS': (0,1), 'M1': (-0.005,0.19), 'M2': (-0.02,1.05)}
    metric_markers=None,           # dict: {'ECS':'o','M1':'x','M2':'^'}
    color_map=None,                # dict {sigma_c: color}; if None -> auto palette
    mu_values=None,                # if None -> inferred from data; can pass list to select specific mus
    sigma_c_values=None,           # if None -> inferred from data
    figsize_colwidth=7,            # column width scaling
    title_suffix=r' with $\mu={mu}$ '
):
    """
    Load CSVs, group by p, and for each p:
      - Plot mean ± std for each metric across b_percentage
      - Rows: metrics; Cols: mu values
      - Lines: sigma_c values
    """

    # ---------- Load & combine ----------
    csv_files = glob.glob(f"{folder}/{file_glob}")
    if not csv_files:
        raise FileNotFoundError(f"No CSV files matched: {folder}/{file_glob}")

    df_list = [pd.read_csv(fp, keep_default_na=False) for fp in csv_files]
    df_all = pd.concat(df_list, ignore_index=True)

    # Filter out empty/NaN p
    df_all = df_all[~df_all['p'].isna() & (df_all['p'] != "")].copy()

    # Decide how to handle p values (numeric or string)
    p_numeric = pd.to_numeric(df_all['p'], errors='coerce')
    use_numeric_p = p_numeric.notna().mean() >= 0.5
    if use_numeric_p:
        df_all['p_num'] = p_numeric
        unique_p_values = sorted(df_all['p_num'].dropna().unique())
        p_selector = ('p_num', unique_p_values)
    else:
        unique_p_values = sorted(df_all['p'].unique())
        p_selector = ('p', unique_p_values)

    # Infer mu/sigma_c if not given, or filter to requested mu_values
    all_mu_values = sorted(df_all['mu'].unique())
    if mu_values is None:
        mu_values = all_mu_values
    else:
        # Filter to only the mu values that exist in the data
        mu_values = [m for m in mu_values if m in all_mu_values]
        if not mu_values:
            raise ValueError("None of the requested mu_values exist in the data")
    
    if sigma_c_values is None:
        sigma_c_values = sorted(df_all['sigma_c'].unique())

    # Colors
    if color_map is None:
        cud_colors = [
            "#000000", "#E69F00", "#56B4E9", "#009E73",
            "#F0E442", "#0072B2", "#D55E00", "#CC79A7"
        ]
        color_map = {s: cud_colors[i % len(cud_colors)] for i, s in enumerate(sigma_c_values)}

    # Labels
    labels = {
        'ECS': r"$ECS$",
        'M1': r"$M_1$",
        'M2': r"$M_2$",
        'p': r"$p$",
        'sigma_c': r"$\sigma_c$",
        'mu': r"$\mu$"
    }
    # Markers
    if metric_markers is None:
        metric_markers = {m: 'o' for m in metrics}

    figs_by_p = {}

    # ---------- Plot per p ----------
    for p_val in p_selector[1]:
        col_name = p_selector[0]
        df_p = df_all[df_all[col_name] == p_val]

        n_rows = len(metrics)
        n_cols = len(mu_values)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(figsize_colwidth * n_cols, figsize_colwidth * n_rows),
            sharex=True,
            sharey=False
        )
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.array([axes])
        elif n_cols == 1:
            axes = axes.reshape(n_rows, 1)

        for row_idx, metric in enumerate(metrics):
            for col_idx, mu in enumerate(mu_values):
                ax = axes[row_idx, col_idx]
                for sigma_c in sigma_c_values:
                    subdf = df_p[(df_p['mu'] == mu) & (df_p['sigma_c'] == sigma_c)]
                    if subdf.empty:
                        continue
                    grouped = (
                        subdf.groupby('b_percentage')
                            .agg({metric: ['mean', 'std']})
                            .reset_index()
                    )
                    if (metric, 'mean') not in grouped.columns:
                        continue

                    means = grouped[(metric, 'mean')].astype(float)
                    stds  = grouped[(metric, 'std')].astype(float).fillna(0.0)
                    xvals = (grouped['b_percentage'].astype(float) * 100.0).values

                    # Plot mean + shaded std
                    ax.plot(
                        xvals, means.values,
                        marker=metric_markers.get(metric, 'o'),
                        linestyle='-',
                        color=color_map.get(sigma_c),
                        label=fr'$\sigma_c={sigma_c}$',
                        linewidth=2
                    )
                    ax.fill_between(
                        xvals,
                        (means - stds).values,
                        (means + stds).values,
                        color=color_map.get(sigma_c),
                        alpha=0.10
                    )

                # Titles/labels (now with correct mu substitution)
                ax.set_title(f"{labels[metric]}{title_suffix.format(mu=mu)}", fontsize=16)
                
                # X-axis: only show labels on bottom row
                if row_idx == n_rows - 1:
                    ax.set_xlabel(r'Budget $\beta_b$ (% of intra-community edges)', fontsize=14)
                else:
                    ax.set_xlabel('')
                
                # Y-axis: only show labels and ticks on leftmost column
                if col_idx == 0:
                    ax.set_ylabel(labels[metric], fontsize=14)
                else:
                    ax.set_ylabel('')
                    ax.tick_params(axis='y', which='both', labelleft=False)
                
                ax.tick_params(axis='x', which='both', labelbottom=True)

                # Per-metric y-limits if provided
                if isinstance(ylim_per_metric, dict) and metric in ylim_per_metric:
                    ax.set_ylim(ylim_per_metric[metric])

                # Legend on the last column
                # if col_idx == n_cols - 1:
                #     ax.legend(title=r'$\sigma_c$', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

                ax.grid(False)
        # Grab handles/labels from any axis (first is fine)
        # pick an Axes to read legend entries from
        first_ax = axes.flat[0] if isinstance(axes, np.ndarray) else axes

        handles, legend_labels = first_ax.get_legend_handles_labels()

        # Only create legend if there are entries
        if handles and legend_labels:
            # (optional) dedupe labels if needed
            by_label = dict(zip(legend_labels, handles))
            handles, legend_labels = list(by_label.values()), list(by_label.keys())

            # Shared, horizontal legend centered at the top
            leg = fig.legend(
                handles, legend_labels,
                loc='upper center',
                ncol=len(legend_labels),          # left→right in one row
                frameon=True,
                fontsize=16,
                markerscale=0, 
                bbox_to_anchor=(0.5, 1.03) # higher above the axes area to avoid overlap
                )
            # Thicken ONLY legend lines
            for h in leg.legend_handles:
                try:
                    h.set_linewidth(3.0)   # <- legend line width
                    h.set_marker('None')   # ensure no marker shows
                except Exception:
                    pass

        # Leave room for the legend
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        # plt.suptitle(f"Results of DICE (with p = {p_val}) on LFR", fontsize=18, y=1.02)
        plt.show()

        figs_by_p[p_val] = fig

    return df_all, figs_by_p

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_metric_heatmaps_same_scale(
    folder_baseline='results',
    glob_baseline='dmon_dice_*.csv',
    folder_p='results_dice_different_p',
    glob_p='dmon_dice_*.csv',
    metrics=('M1', 'M2'),
    cmap_m1='YlGnBu',
    cmap_m2='OrRd',
    annot=True,
    fmt='.3f',
    figsize=(15, 10)
):
    # --- Load & combine ---
    csv_files_base = glob.glob(f"{folder_baseline}/{glob_baseline}")
    df_list = [pd.read_csv(f).assign(p=0.5) for f in csv_files_base]  # add p=0.5 for baseline

    csv_files_p = glob.glob(f"{folder_p}/{glob_p}")
    df_list += [pd.read_csv(f) for f in csv_files_p]                 # already has p

    df_all = pd.concat(df_list, ignore_index=True)

    # --- Aggregate over realizations ---
    grouped = (
        df_all.groupby(['mu', 'sigma_c', 'p'])
              .agg({m: 'mean' for m in metrics})
              .reset_index()
    )

    # Ensure numeric
    for col in ['mu', 'sigma_c', 'p']:
        grouped[col] = grouped[col].astype(float)
    grouped = grouped.sort_values(['mu', 'sigma_c', 'p'])

    # Layout 2x3 for (M1: 3 views) + (M2: 3 views)
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=figsize)
    axes = axes.flatten()

    plot_configs = [
        ('M1', 'p',  'sigma_c'),
        ('M1', 'p',  'mu'),
        ('M1', 'mu', 'sigma_c'),
        ('M2', 'p',  'sigma_c'),
        ('M2', 'p',  'mu'),
        ('M2', 'mu', 'sigma_c'),
    ]
    # labels mapping
    labels = {
        'M1': r"$M_1$",
        'M2': r"$M_2$",
        'p': r"$p$",
        'sigma_c': r"$\sigma_c$",
        'mu': r"$\mu$"
    }

    # --- Compute shared vmin/vmax per metric across all relevant pivots ---
    metric_limits = {}
    for metric in metrics:
        vals = []
        for m, idx, cols in plot_configs:
            if m != metric:
                continue
            pivot = grouped.pivot_table(index=idx, columns=cols, values=metric)
            vals.append(pivot.values)
        stacked = np.concatenate([v.ravel() for v in vals if v.size > 0])
        stacked = stacked[~np.isnan(stacked)]
        if stacked.size == 0:
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = float(stacked.min()), float(stacked.max())
        metric_limits[metric] = (vmin, vmax)

    # --- Plot with shared scale per metric ---

        # Make dedicated colorbar axes to the right of each row
    fig.tight_layout()
    fig.canvas.draw()  # ensure positions are computed

    # Positions of the last column axes (rightmost)
    ax_m1_last = axes[2]   # top-right
    ax_m2_last = axes[5]   # bottom-right

    # Add thin axes for colorbars next to those
    bbox1 = ax_m1_last.get_position()
    bbox2 = ax_m2_last.get_position()
    cbar_ax_m1 = fig.add_axes([bbox1.x1 + 0.01, bbox1.y0, 0.015, bbox1.height])
    cbar_ax_m2 = fig.add_axes([bbox2.x1 + 0.01, bbox2.y0, 0.015, bbox2.height])

    counts = {'M1': 0, 'M2': 0}  # how many heatmaps of each metric we've drawn

    for ax, (metric, index, columns) in zip(axes, plot_configs):
        pivot = grouped.pivot_table(index=index, columns=columns, values=metric)
        vmin, vmax = metric_limits[metric]

        # Use one colormap for all M1, another for all M2
        cmap_this = cmap_m1 if metric == 'M1' else cmap_m2

        # Show the colorbar only on the last column per metric (i.e., the 3rd plot in each row)
        # counts[metric] will be 0,1,2; we want cbar only when it equals 2
        show_cbar = (counts[metric] == 2)
        cbar_ax = cbar_ax_m1 if (metric == 'M1' and show_cbar) else (cbar_ax_m2 if (metric == 'M2' and show_cbar) else None)

        hm = sns.heatmap(
            pivot, cmap=cmap_this, annot=annot, fmt=fmt, ax=ax,
            vmin=vmin, vmax=vmax, cbar=show_cbar, cbar_ax=cbar_ax
        )
        # if show_cbar and hm.collections:
        #     hm.collections[0].colorbar.set_label(f"{labels[metric]} scale", rotation=270, labelpad=12)

        ax.set_title(f"{labels[metric]}: {labels[index]} vs {labels[columns]}", fontsize=12)
        ax.set_xlabel(labels[columns])
        ax.set_ylabel(labels[index])

        counts[metric] += 1


    # # --- Plot with shared scale per metric ---
    # for ax, (metric, index, columns) in zip(axes, plot_configs):
    #     pivot = grouped.pivot_table(index=index, columns=columns, values=metric)
    #     vmin, vmax = metric_limits[metric]
    #     # Use one colormap for all M1, another for all M2
    #     if metric == 'M1':
    #         cmap_this = cmap_m1
    #     else:
    #         cmap_this = cmap_m2
    #     sns.heatmap(
    #         pivot, cmap=cmap_this, annot=annot, fmt=fmt, ax=ax,
    #         vmin=vmin, vmax=vmax, cbar=True
    #     )
    #     ax.set_title(f"{labels[metric]}: {labels[index]} vs {labels[columns]}", fontsize=12)
    #     ax.set_xlabel(labels[columns])
    #     ax.set_ylabel(labels[index])

    plt.tight_layout()
    plt.savefig("metric_heatmaps.png", dpi=300)
    plt.show()

    return grouped, metric_limits

def plot_area_mu_sigma_heatmaps_multiple_methods(
    df_base: pd.DataFrame,
    modified_dfs: dict,                  # {"label": df_mod, "label2": df_mod2, ...}
    metrics=('M1','M2'),
    cmap='YlGnBu',
    annot=True,
    fmt='.3f',
    figsize_per_row=(14, 5)              # (width, height) per row (i.e., per method)
):
    """
    Compare multiple modified methods to a single baseline:
    For each metric and each (μ, σ_c), compute signed area over b of:
        ∫_b [ metric_mod(b) - metric_base(b) ] db
    Then plot μ × σ_c heatmaps with one row per modified method, columns by metric.
    
    Assumes df_base and each df_mod have:
      ['mu', 'sigma_c', 'b_percentage', <metrics...>, ...]
    If you need to restrict to p=0.5 or similar, pre-filter the input dataframes.
    
    Returns
    -------
    areas : pd.DataFrame with columns ['method','metric','mu','sigma_c','area']
    limits: dict {metric: (vmin, vmax)} shared across all rows for that metric
    """

    # Domains from union of baseline + all modified
    mu_values    = sorted(pd.concat([df_base['mu']] + [d['mu'] for d in modified_dfs.values()]).unique())
    sigma_values = sorted(pd.concat([df_base['sigma_c']] + [d['sigma_c'] for d in modified_dfs.values()]).unique())

    # helper: mean metric over b (avg over realizations) indexed by b in [0,1]
    def mean_curve_over_b(df, metric):
        return df.groupby('b_percentage')[metric].mean().sort_index()

    # Compute areas for every (method, metric, μ, σ_c)
    rows = []
    for method_label, df_mod in modified_dfs.items():
        for metric in metrics:
            for mu in mu_values:
                db_mu = df_base[df_base['mu'] == mu]
                dm_mu = df_mod[df_mod['mu'] == mu]
                for sc in sigma_values:
                    base = db_mu[db_mu['sigma_c'] == sc]
                    mod  = dm_mu[dm_mu['sigma_c'] == sc]
                    if base.empty or mod.empty:
                        continue

                    y_base = mean_curve_over_b(base, metric)
                    y_mod  = mean_curve_over_b(mod,  metric)

                    # align on common b grid
                    common_b = sorted(set(y_base.index).intersection(set(y_mod.index)))
                    if len(common_b) < 2:
                        continue

                    x = np.array(common_b, dtype=float)     # b in [0,1]
                    base_vals = y_base.reindex(common_b).to_numpy(float)
                    mod_vals  = y_mod.reindex(common_b).to_numpy(float)

                    # signed area (Modified - Baseline) over b
                    area = float(np.trapz(mod_vals - base_vals, x))
                    rows.append({
                        'method': method_label,
                        'metric': metric,
                        'mu': float(mu),
                        'sigma_c': float(sc),
                        'area': area
                    })

    areas = pd.DataFrame(rows)
    if areas.empty:
        raise ValueError("No areas computed. Ensure overlapping (μ, σ_c) and b grids for baseline and all methods.")

    # Uniform color scale per metric across ALL methods (rows)
    limits = {}
    for metric in metrics:
        vals = areas.loc[areas['metric'] == metric, 'area'].to_numpy()
        vals = vals[~np.isnan(vals)]
        limits[metric] = (float(vals.min()), float(vals.max())) if vals.size else (0.0, 0.0)

    # Plot: one row per method, columns by metric
    n_rows = len(modified_dfs)
    n_cols = len(metrics)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(figsize_per_row[0], figsize_per_row[1] * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    for r, (method_label, _) in enumerate(modified_dfs.items()):
        df_row = areas[areas['method'] == method_label]
        for c, metric in enumerate(metrics):
            ax = axes[r, c]
            df_m = df_row[df_row['metric'] == metric]
            pivot = df_m.pivot_table(index='mu', columns='sigma_c', values='area')
            vmin, vmax = limits[metric]
            sns.heatmap(pivot, cmap=cmap, annot=annot, fmt=fmt, ax=ax,
                        vmin=vmin, vmax=vmax, cbar=True)
            ax.set_title(f"{method_label}  |  {metric}: ∫ [ModDICE − Base DICE] db", fontsize=11)
            ax.set_xlabel("σ_c"); ax.set_ylabel("μ")

    plt.tight_layout()
    plt.show()

    return areas, limits

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_baseline_vs_multiple_modified_over_b_for_sigma(
    df_baseline: pd.DataFrame,
    modified_dfs: dict,                 # {"Method A": df_modA, "Method B": df_modB, ...}
    metrics=('ECS','M1','M2'),
    sigma_c=None,                       # required: one σ_c to display
    mu_values=None,                     # if None -> inferred from data
    p_baseline=None,                    # optional numeric filter for baseline df
    p_modified: dict | None = None,     # optional per-method p filter: {"Method A": 0.5, ...}
    baseline_style=None,                # dict for plt.plot
    modified_styles: dict | None = None,# {"Method A": {...}, "Method B": {...}}
    fill_between=True,                  # shade area (Mod - Base)
    fill_alpha=0.15,
    show_std=False,                     # draw mean ± std bands
    figsize_rowheight=5,
    ylims=None                          # dict per metric, or a tuple for all
):
    """
    Plot baseline vs MULTIPLE modified DICE curves over b for a fixed sigma_c.
    Rows: μ values; Cols: metrics. Multiple modified curves per subplot.
    Shades area between each modified curve and the baseline (on common b grid).
    
    Returns
    -------
    areas_df : DataFrame with columns
               ['metric','mu','sigma_c','method','area_signed']
               where area is ∫ (mod - base) db (b in [0,1]).
    """

    if sigma_c is None:
        raise ValueError("Please provide sigma_c to plot.")

    # Light filtering
    d0 = df_baseline.copy()
    if p_baseline is not None:
        d0 = d0[pd.to_numeric(d0['p'], errors='coerce') == float(p_baseline)]
    d0 = d0[d0['sigma_c'] == sigma_c]

    # normalize μ list
    if mu_values is None:
        mu_values = sorted(d0['mu'].unique())

    # default styles
    if baseline_style is None:
        baseline_style = dict(color="#000000", marker="o", linewidth=2, label="Baseline")
    if modified_styles is None:
        # auto-palette
        palette = ["#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        modified_styles = {}
        for i, m in enumerate(modified_dfs.keys()):
            modified_styles[m] = dict(color=palette[i % len(palette)], marker="s", linewidth=2, label=m)

    # per-method p filter (optional)
    p_modified = p_modified or {}

    # layout
    n_rows = len(mu_values)
    n_cols = len(metrics)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, figsize_rowheight * n_rows), sharex=True)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    # helper to compute mean/std over realizations per b (sorted), with b% column
    def mean_std_by_b(df, metric):
        g = df.groupby('b_percentage')[metric].agg(['mean','std']).sort_index().reset_index()
        g['b_pct'] = g['b_percentage'].astype(float) * 100.0
        return g

    # compute + plot
    area_rows = []

    for r, mu in enumerate(mu_values):
        base_mu = d0[d0['mu'] == mu]

        for c, metric in enumerate(metrics):
            ax = axes[r, c]

            # Baseline curve
            g0 = mean_std_by_b(base_mu, metric)
            if not g0.empty:
                ax.plot(g0['b_pct'], g0['mean'], **baseline_style)
                if show_std:
                    ax.fill_between(
                        g0['b_pct'], g0['mean'] - g0['std'], g0['mean'] + g0['std'],
                        color=baseline_style.get('color', '#1f77b4'), alpha=0.10
                    )

            # Modified methods
            for method_name, df_mod in modified_dfs.items():
                dm = df_mod.copy()
                pm = p_modified.get(method_name, None)
                if pm is not None:
                    dm = dm[pd.to_numeric(dm['p'], errors='coerce') == float(pm)]
                dm = dm[(dm['sigma_c'] == sigma_c) & (dm['mu'] == mu)]

                g1 = mean_std_by_b(dm, metric)
                if g1.empty:
                    continue

                # plot modified curve
                style = modified_styles.get(method_name, dict(color="#d62728", marker="s", linewidth=2, label=method_name))
                ax.plot(g1['b_pct'], g1['mean'], **style)
                if show_std:
                    ax.fill_between(
                        g1['b_pct'], g1['mean'] - g1['std'], g1['mean'] + g1['std'],
                        color=style.get('color', '#d62728'), alpha=0.10
                    )

                # Shade area between this modified and baseline
                if fill_between and not g0.empty:
                    common = np.intersect1d(g0['b_pct'].values, g1['b_pct'].values)
                    if common.size >= 2:
                        y0 = g0.set_index('b_pct').loc[common, 'mean'].values
                        y1 = g1.set_index('b_pct').loc[common, 'mean'].values
                        ax.fill_between(common, y0, y1, color=style.get('color', '#d62728'), alpha=fill_alpha)

                        # compute signed area in b∈[0,1]
                        xb = (common / 100.0).astype(float)
                        area_signed = float(np.trapz(y1 - y0, xb))
                        area_rows.append({
                            'metric': metric,
                            'mu': float(mu),
                            'sigma_c': float(sigma_c),
                            'method': method_name,
                            'area_signed': area_signed
                        })

            # Titles/labels and y-lims
            ax.set_title(f"{metric}  |  μ={mu}, σ_c={sigma_c}", fontsize=12)
            ax.set_xlabel("DICE Budget b (% of intra-community edges)", fontsize=11)
            ax.set_ylabel(metric, fontsize=11)
            if isinstance(ylims, dict) and metric in ylims:
                ax.set_ylim(ylims[metric])
            elif isinstance(ylims, tuple):
                ax.set_ylim(ylims)

            ax.grid(False)
            if c == n_cols - 1:
                ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0.)

    plt.tight_layout(rect=[0, 0, 0.92, 1])
    plt.suptitle(f"Baseline vs Multiple Modified DICE over b  (σ_c={sigma_c})", y=1.02, fontsize=14)
    plt.show()

    areas_df = pd.DataFrame(area_rows)
    return areas_df

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_area_vs_sigma_for_methods(
    df_base: pd.DataFrame,
    modified_dfs: dict,                 # {"Method A": dfA, "Method B": dfB, ...}
    metrics=('ECS','M1','M2'),
    mu_values=None,                     # if None -> infer from baseline
    sigma_c_values=None,                # if None -> infer from union of all
    p_baseline=None,                    # optional numeric filter for baseline
    p_modified: dict | None = None,     # optional per-method p filter: {"Method A": 0.5, ...}
    method_styles: dict | None = None,  # optional per-method plot styles
    figsize_rowheight=4.5,
    ylims=None,                         # dict per metric or tuple for all
    aggregate_mu: str | None = None     # None | 'mean' | 'median'
):
    """
    For each (metric, μ), compute area over b between each modified method and the baseline
    at every σ_c, then plot AREA(σ_c) as lines (one per method).


    Returns
    -------
    area_df_detail : DataFrame ['method','metric','mu','sigma_c','area']
    area_df_plot   : DataFrame used for plotting (aggregated if requested)
    """

    d0 = df_base.copy()
    if p_baseline is not None:
        d0 = d0[pd.to_numeric(d0['p'], errors='coerce') == float(p_baseline)]

    # domains
    if mu_values is None:
        mu_values = sorted(d0['mu'].unique())
    if sigma_c_values is None:
        sigma_c_values = sorted(pd.concat([d0['sigma_c']] + [d['sigma_c'] for d in modified_dfs.values()]).unique())

    # optional per-method p filter
    p_modified = p_modified or {}

    # default styles
    if method_styles is None:
        palette = ["#d62728", "#2ca02c", "#9467bd", "#18e05a", "#e377c2", "#4d4646", "#bcbd22", "#17becf"]
        method_styles = {m: dict(color=palette[i % len(palette)], marker="o", linewidth=2, label=m)
                         for i, m in enumerate(modified_dfs.keys())}

    def mean_curve_over_b(df, metric):
        """mean over realizations per b; sorted by b"""
        return df.groupby('b_percentage')[metric].mean().sort_index()

    def integrate_diff_over_b(df_base_mu_sc, df_mod_mu_sc, metric):
        """trapz over common b in [0,1] of (mod - base)"""
        if df_base_mu_sc.empty or df_mod_mu_sc.empty:
            return None
        yb = mean_curve_over_b(df_base_mu_sc, metric)
        ym = mean_curve_over_b(df_mod_mu_sc, metric)
        common_b = sorted(set(yb.index).intersection(set(ym.index)))
        if len(common_b) < 2:
            return None
        x = np.array(common_b, dtype=float)     # b in [0,1]
        base_vals = yb.reindex(common_b).to_numpy(float)
        mod_vals  = ym.reindex(common_b).to_numpy(float)
        diff = mod_vals - base_vals
        return float(np.trapz(diff, x))

    # compute areas (detail per μ)
    area_rows = []
    for mu in mu_values:
        base_mu = d0[d0['mu'] == mu]
        for sigma in sigma_c_values:
            base_mu_sc = base_mu[base_mu['sigma_c'] == sigma]
            for method_name, df_mod in modified_dfs.items():
                dm = df_mod.copy()
                pm = p_modified.get(method_name, None)
                if pm is not None:
                    dm = dm[pd.to_numeric(dm['p'], errors='coerce') == float(pm)]
                mod_mu_sc = dm[(dm['mu'] == mu) & (dm['sigma_c'] == sigma)]
                for metric in metrics:
                    area = integrate_diff_over_b(base_mu_sc, mod_mu_sc, metric)
                    if area is None:
                        continue
                    area_rows.append({
                        'method': method_name,
                        'metric': metric,
                        'mu': float(mu),
                        'sigma_c': float(sigma),
                        'area': area
                    })

    area_df_detail = pd.DataFrame(area_rows)
    if area_df_detail.empty:
        raise ValueError("No areas computed. Ensure overlapping (μ, σ_c) and b grids for baseline and all methods.")

    # --- aggregate over μ if requested ---
    if aggregate_mu in ('mean', 'median'):
        agg_fun = 'mean' if aggregate_mu == 'mean' else 'median'
        area_df_plot = (area_df_detail
                        .groupby(['method','metric','sigma_c'], as_index=False)['area']
                        .agg(agg_fun)
                        .rename(columns={'area': f'area_{aggregate_mu}'}))
        ycol = f'area_{aggregate_mu}'

        # plotting: single row (columns = metrics), lines = methods
        n_rows = 1
        n_cols = len(metrics)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, figsize_rowheight * n_rows), sharex=True)
        if n_cols == 1:
            axes = [axes]

        for c, metric in enumerate(metrics):
            ax = axes[c]
            for method_name in modified_dfs.keys():
                s = (area_df_plot[(area_df_plot['method'] == method_name) &
                                  (area_df_plot['metric'] == metric)]
                     .sort_values('sigma_c'))
                if s.empty:
                    continue
                style = method_styles.get(method_name, dict(marker='o', linewidth=2, label=method_name))
                ax.plot(s['sigma_c'], s[ycol], **style)

            ax.set_title(f"{metric}  |  ∫(mod − base) db  ({aggregate_mu} over μ)", fontsize=12)
            ax.set_xlabel("σ_c"); ax.set_ylabel("Area over b")
            if isinstance(ylims, dict) and metric in ylims:
                ax.set_ylim(ylims[metric])
            elif isinstance(ylims, tuple):
                ax.set_ylim(ylims)
            ax.grid(False)
            if c == n_cols - 1:
                ax.legend(title='Method', loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

        plt.tight_layout(rect=[0, 0, 0.92, 1])
        plt.suptitle("Area between curves over b vs σ_c  (aggregated over μ)", y=1.02, fontsize=14)
        plt.show()

    else:
        # original layout: rows = μ, cols = metrics
        area_df_plot = area_df_detail.copy()
        n_rows = len(mu_values)
        n_cols = len(metrics)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, figsize_rowheight * n_rows), sharex=True)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.array([axes])
        elif n_cols == 1:
            axes = axes.reshape(n_rows, 1)

        for r, mu in enumerate(mu_values):
            for c, metric in enumerate(metrics):
                ax = axes[r, c]
                for method_name in modified_dfs.keys():
                    s = (area_df_plot[(area_df_plot['method'] == method_name) &
                                      (area_df_plot['metric'] == metric) &
                                      (area_df_plot['mu'] == mu)]
                         .sort_values('sigma_c'))
                    if s.empty:
                        continue
                    style = method_styles.get(method_name, dict(marker='o', linewidth=2, label=method_name))
                    ax.plot(s['sigma_c'], s['area'], **style)

                ax.set_title(f"{metric}  |  μ={mu}  ∫(mod − base) db", fontsize=12)
                ax.set_xlabel("σ_c"); ax.set_ylabel("Area over b")
                if isinstance(ylims, dict) and metric in ylims:
                    ax.set_ylim(ylims[metric])
                elif isinstance(ylims, tuple):
                    ax.set_ylim(ylims)

                ax.grid(False)
                if c == n_cols - 1:
                    ax.legend(title='Method', loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

        plt.tight_layout(rect=[0, 0, 0.92, 1])
        plt.suptitle("Area between curves over b vs σ_c", y=1.02, fontsize=14)
        plt.show()

    return area_df_detail, area_df_plot


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_relative_change_mu_sigma_heatmaps_multiple_methods(
    df_base: pd.DataFrame,
    modified_dfs: dict,                   # {"label": df_mod, ...}
    metrics=('M1','M2'),
    aggregate='mean',                     # 'mean' (default) or 'integral'
    eps=1e-30,                             # avoid divide-by-zero when Base≈0
    percent=False,                        # True -> show % (×100)
    cmap=None,                            # dict per metric or single cmap for all
    annot=True,
    fmt='.3f',
    figsize_per_row=(14, 5)
):
    """
    For each metric and (μ, σ_c), compute relative change over b:
        rel(b) = (Mod(b) - Base(b)) / max(|Base(b)|, eps)

    Then aggregate over b:
      - 'mean'     : average_b rel(b)            [recommended default]
      - 'integral' : ∫ rel(b) db  (trapz on b∈[0,1])

    Plots μ × σ_c heatmaps with one row per modified method (columns = metrics).

    Assumes dataframes have: ['mu','sigma_c','b_percentage', <metrics...>, ...]
    Pre-filter to desired p before calling if needed.
    """

    # Labels for metrics
    labels = {
        'ECS': r"$ECS$",
        'M1': r"$\overline{\Delta M_1}$",
        'M2': r"$\overline{\Delta M_2}$",
        'p': r"$p$",
        'sigma_c': r"$\sigma_c$",
        'mu': r"$\mu$"
    }

    # Default colormaps per metric if not provided
    if cmap is None:
        cmap = {
            'ECS': 'YlGnBu',
            'M1': 'PuBuGn', 
            'M2': 'YlOrBr'
        }
    elif isinstance(cmap, str):
        # If single cmap provided, use it for all metrics
        cmap = {m: cmap for m in metrics}

    mu_values    = sorted(pd.concat([df_base['mu']] + [d['mu'] for d in modified_dfs.values()]).unique())
    sigma_values = sorted(pd.concat([df_base['sigma_c']] + [d['sigma_c'] for d in modified_dfs.values()]).unique())

    def mean_curve_over_b(df, metric):
        return df.groupby('b_percentage')[metric].mean().sort_index()

    rows = []
    for method_label, df_mod in modified_dfs.items():
        for metric in metrics:
            for mu in mu_values:
                db_mu = df_base[df_base['mu'] == mu]
                dm_mu = df_mod[df_mod['mu'] == mu]
                for sc in sigma_values:
                    base = db_mu[db_mu['sigma_c'] == sc]
                    mod  = dm_mu[dm_mu['sigma_c'] == sc]
                    if base.empty or mod.empty:
                        continue

                    y_base = mean_curve_over_b(base, metric)
                    y_mod  = mean_curve_over_b(mod,  metric)

                    common_b = sorted(set(y_base.index).intersection(set(y_mod.index)))
                    if len(common_b) < 2:
                        continue

                    x = np.array(common_b, dtype=float)  # b in [0,1]
                    base_vals = y_base.reindex(common_b).to_numpy(float)
                    mod_vals  = y_mod.reindex(common_b).to_numpy(float)

                    denom = np.maximum(np.abs(base_vals), eps)
                    rel = (mod_vals - base_vals) / denom
                    # print('method:', method_label, 'metric:', metric, 'mu:', mu, 'sc:', sc)
                    # print('rel:', rel)

                    if aggregate == 'integral':
                        rel_value = float(np.trapz(rel, x))
                    elif aggregate == 'mean':
                        rel_value = float(np.mean(rel))
                    else:
                        raise ValueError("aggregate must be 'mean' or 'integral'")

                    if percent:
                        rel_value *= 100.0

                    rows.append({
                        'method': method_label,
                        'metric': metric,
                        'mu': float(mu),
                        'sigma_c': float(sc),
                        'rel_value': rel_value
                    })

    rel_df = pd.DataFrame(rows)
    if rel_df.empty:
        raise ValueError("No relative values computed. Ensure overlapping (μ, σ_c) and b grids.")
    
    # shared color scale per metric
    limits = {}
    for metric in metrics:
        vals = rel_df.loc[rel_df['metric'] == metric, 'rel_value'].to_numpy()
        vals = vals[~np.isnan(vals)]
        limits[metric] = (float(vals.min()), float(vals.max())) if vals.size else (0.0, 0.0)

    n_rows = len(modified_dfs)
    n_cols = len(metrics)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(figsize_per_row[0], figsize_per_row[1] * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    # subtitle = "Average relative Δ" if aggregate == 'mean' else "∫ Δ(ModDice-Baseline) db"
    unit = " (%)" if percent else ""
    for r, (method_label, _) in enumerate(modified_dfs.items()):
        df_row = rel_df[rel_df['method'] == method_label]
        for c, metric in enumerate(metrics):
            ax = axes[r, c]
            df_m = df_row[df_row['metric'] == metric]
            pivot = df_m.pivot_table(index='mu', columns='sigma_c', values='rel_value')
            vmin, vmax = limits[metric]
            # Use metric-specific colormap
            metric_cmap = cmap.get(metric, 'YlGnBu') if isinstance(cmap, dict) else cmap
            sns.heatmap(pivot, cmap=metric_cmap, annot=annot, fmt=fmt, ax=ax,
                        vmin=vmin, vmax=vmax, cbar=True)
            # Use LaTeX labels for metrics in title
            metric_label = labels.get(metric, metric)
            # ax.set_title(f"{method_label} | {metric_label}: {subtitle}{unit}", fontsize=11)
            ax.set_title(f"{metric_label}{unit}", fontsize=12)
            ax.set_xlabel(labels['sigma_c'])
            
            # Only show y-axis label on first column (M1), hide for subsequent columns (M2, etc.)
            if c == 0:
                ax.set_ylabel(labels['mu'])
            else:
                ax.set_ylabel('')
                ax.tick_params(axis='y', labelleft=False)

    plt.tight_layout()
    plt.show()

    return rel_df, limits

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_relative_change_vs_sigma_for_methods(
    df_base: pd.DataFrame,
    modified_dfs: dict,                 # {"Method A": dfA, "Method B": dfB, ...}
    metrics=('ECS','M1','M2'),
    mu_values=None,                     # if None -> infer from baseline
    sigma_c_values=None,                # if None -> infer from union of all
    p_baseline=None,                    # optional numeric filter for baseline
    p_modified: dict | None = None,     # optional per-method p filter: {"Method A": 0.5, ...}
    method_styles: dict | None = None,  # optional per-method plot styles
    figsize_rowheight=4.5,
    ylims=None,                         # dict per metric or tuple for all
    aggregate_mu: str | None = None,    # None | 'mean' 
    eps: float = 1e-8,                  # to avoid division by ~0 baseline
    percent: bool = False               # if True, multiply by 100 for %
):
    """
    For each (metric, μ), compute the AVERAGE relative change over b between each
    modified method and the baseline at every σ_c:

        rel_avg(μ, σ_c) = mean_b [ (Mod(b) - Base(b)) / max(|Base(b)|, eps) ]

    Then plot rel_avg(σ_c) as lines (one per method).

    Returns
    -------
    rel_df_detail : DataFrame ['method','metric','mu','sigma_c','rel_avg']
    rel_df_plot   : DataFrame used for plotting (aggregated if requested)
    """

    d0 = df_base.copy()
    if p_baseline is not None:
        d0 = d0[pd.to_numeric(d0['p'], errors='coerce') == float(p_baseline)]

    # domains
    if mu_values is None:
        mu_values = sorted(d0['mu'].unique())
    if sigma_c_values is None:
        sigma_c_values = sorted(pd.concat([d0['sigma_c']] + [d['sigma_c'] for d in modified_dfs.values()]).unique())

    # per-method p filter
    p_modified = p_modified or {}

    # default styles
    if method_styles is None:
        palette = ["#d62728", "#2ca02c", "#9467bd", "#18e05a", "#e377c2", "#4d4646", "#bcbd22", "#17becf"]
        method_styles = {m: dict(color=palette[i % len(palette)], marker="o", linewidth=2, label=m)
                         for i, m in enumerate(modified_dfs.keys())}

    def mean_curve_over_b(df, metric):
        """Mean over realizations per b; index = b in [0,1]."""
        return df.groupby('b_percentage')[metric].mean().sort_index()

    # --- compute average relative change per (method, metric, μ, σ_c) ---
    rows = []
    for mu in mu_values:
        base_mu = d0[d0['mu'] == mu]
        for sigma in sigma_c_values:
            base_mu_sc = base_mu[base_mu['sigma_c'] == sigma]
            for method_name, df_mod in modified_dfs.items():
                dm = df_mod.copy()
                pm = p_modified.get(method_name, None)
                if pm is not None:
                    dm = dm[pd.to_numeric(dm['p'], errors='coerce') == float(pm)]
                mod_mu_sc = dm[(dm['mu'] == mu) & (dm['sigma_c'] == sigma)]

                for metric in metrics:
                    if base_mu_sc.empty or mod_mu_sc.empty:
                        continue
                    yb = mean_curve_over_b(base_mu_sc, metric)
                    ym = mean_curve_over_b(mod_mu_sc, metric)

                    common_b = sorted(set(yb.index).intersection(set(ym.index)))
                    if len(common_b) < 2:
                        continue

                    base_vals = yb.reindex(common_b).to_numpy(float)
                    mod_vals  = ym.reindex(common_b).to_numpy(float)

                    rel = (mod_vals - base_vals) / np.maximum(np.abs(base_vals), eps)
                    rel_avg = float(np.mean(rel))
                    if percent:
                        rel_avg *= 100.0

                    rows.append({
                        'method': method_name,
                        'metric': metric,
                        'mu': float(mu),
                        'sigma_c': float(sigma),
                        'rel_avg': rel_avg
                    })

    rel_df_detail = pd.DataFrame(rows)
    if rel_df_detail.empty:
        raise ValueError("No relative averages computed. Ensure overlapping (μ, σ_c) and b grids.")

    # --- aggregate over μ if requested ---
    if aggregate_mu == 'mean':
        agg_fun = 'mean' 
        rel_df_plot = (rel_df_detail
                       .groupby(['method','metric','sigma_c'], as_index=False)['rel_avg']
                       .agg(agg_fun)
                       .rename(columns={'rel_avg': f'rel_avg_{aggregate_mu}'}))
        ycol = f'rel_avg_{aggregate_mu}'

        # plot: single row (columns = metrics), lines = methods
        n_rows = 1
        n_cols = len(metrics)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, figsize_rowheight * n_rows), sharex=True)
        if n_cols == 1:
            axes = [axes]

        for c, metric in enumerate(metrics):
            ax = axes[c]
            for method_name in modified_dfs.keys():
                s = (rel_df_plot[(rel_df_plot['method'] == method_name) &
                                 (rel_df_plot['metric'] == metric)]
                     .sort_values('sigma_c'))
                if s.empty:
                    continue
                style = method_styles.get(method_name, dict(marker='o', linewidth=2, label=method_name))
                ax.plot(s['sigma_c'], s[ycol], **style)

            ylab = "Avg relative change (%)" if percent else "Avg relative change"
            ax.set_title(f"{metric}  |  {aggregate_mu} over μ", fontsize=12)
            ax.set_xlabel("σ_c"); ax.set_ylabel(ylab)
            if isinstance(ylims, dict) and metric in ylims:
                ax.set_ylim(ylims[metric])
            elif isinstance(ylims, tuple):
                ax.set_ylim(ylims)
            ax.grid(False)
            if c == n_cols - 1:
                ax.legend(title='Method', loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

        plt.tight_layout(rect=[0, 0, 0.92, 1])
        ttl_unit = " (%)" if percent else ""
        plt.suptitle(f"Average relative change over b vs σ_c{ttl_unit}", y=1.02, fontsize=14)
        plt.show()

    else:
        # original layout: rows = μ, cols = metrics
        rel_df_plot = rel_df_detail.copy()
        n_rows = len(mu_values)
        n_cols = len(metrics)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, figsize_rowheight * n_rows), sharex=True)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.array([axes])
        elif n_cols == 1:
            axes = axes.reshape(n_rows, 1)

        for r, mu in enumerate(mu_values):
            for c, metric in enumerate(metrics):
                ax = axes[r, c]
                for method_name in modified_dfs.keys():
                    s = (rel_df_plot[(rel_df_plot['method'] == method_name) &
                                     (rel_df_plot['metric'] == metric) &
                                     (rel_df_plot['mu'] == mu)]
                         .sort_values('sigma_c'))
                    if s.empty:
                        continue
                    style = method_styles.get(method_name, dict(marker='o', linewidth=2, label=method_name))
                    ax.plot(s['sigma_c'], s['rel_avg'], **style)

                ylab = "Avg relative change (%)" if percent else "Avg relative change"
                ax.set_title(f"{metric}  |  μ={mu}", fontsize=12)
                ax.set_xlabel("σ_c"); ax.set_ylabel(ylab)
                if isinstance(ylims, dict) and metric in ylims:
                    ax.set_ylim(ylims[metric])
                elif isinstance(ylims, tuple):
                    ax.set_ylim(ylims)
                ax.grid(False)
                if c == n_cols - 1:
                    ax.legend(title='Method', loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

        plt.tight_layout(rect=[0, 0, 0.92, 1])
        ttl_unit = " (%)" if percent else ""
        plt.suptitle(f"Average relative change (ModDice-Baseline)/|Baseline| vs σ_c{ttl_unit}", y=1.02, fontsize=14)
        plt.show()

    return rel_df_detail, rel_df_plot


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_diff_at_b_vs_sigma_for_methods(
    df_base: pd.DataFrame,
    modified_dfs: dict,                 # {"Method A": dfA, "Method B": dfB, ...}
    metrics=('ECS','M1','M2'),
    target_b=0.05,                      # evaluate at this b in [0,1]
    mu_values=None,                     # if None -> infer from baseline
    sigma_c_values=None,                # if None -> infer from union
    p_baseline=None,                    # optional numeric filter for baseline
    p_modified: dict | None = None,     # optional per-method p filter: {"Method A": 0.5, ...}
    interpolate=True,                   # linear interp if target_b not present
    aggregate_mu: str | None = None,    # None | 'mean' | 'median'  -> aggregate diffs across μ
    method_styles: dict | None = None,  # per-method plot styles
    figsize_rowheight=4.5,
    ylims=None                          # dict per metric or tuple for all
):
    """
    For each (metric, μ), compute (Modified - Baseline) at a fixed b = target_b
    for every σ_c, then plot DIFF(σ_c) as lines (one per method).

    If aggregate_mu is 'mean' or 'median', diffs are aggregated over μ and the plot
    shows a single row (columns = metrics), lines = methods vs σ_c.

    Returns
    -------
    diff_df_detail : DataFrame ['method','metric','mu','sigma_c','diff_at_b']
    diff_df_plot   : DataFrame used for plotting
    """

    d0 = df_base.copy()
    if p_baseline is not None:
        d0 = d0[pd.to_numeric(d0['p'], errors='coerce') == float(p_baseline)]

    # domains
    if mu_values is None:
        mu_values = sorted(d0['mu'].unique())
    if sigma_c_values is None:
        sigma_c_values = sorted(pd.concat([d0['sigma_c']] + [d['sigma_c'] for d in modified_dfs.values()]).unique())

    # optional per-method p filter
    p_modified = p_modified or {}

    # default styles
    if method_styles is None:
        palette = ["#d62728", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        method_styles = {m: dict(color=palette[i % len(palette)], marker="o", linewidth=2, label=m)
                         for i, m in enumerate(modified_dfs.keys())}

    def mean_series_by_b(df, metric):
        """Series indexed by b_percentage in [0,1], values = mean(metric) over realizations."""
        if df.empty:
            return pd.Series(dtype=float)
        s = df.groupby('b_percentage')[metric].mean().sort_index()
        s.index = s.index.astype(float)
        return s

    def value_at_b(series_b, b, interpolate=True):
        """Get value at b; interpolate linearly if needed/allowed."""
        if series_b.empty:
            return None
        if b in series_b.index:
            return float(series_b.loc[b])
        if not interpolate:
            return None
        x = series_b.index.values
        y = series_b.values.astype(float)
        if b < x.min() or b > x.max() or len(x) < 2:
            return None
        return float(np.interp(b, x, y))

    # --- compute diffs for every (method, metric, μ, σ_c) ---
    rows = []
    for mu in mu_values:
        base_mu = d0[d0['mu'] == mu]
        for sigma in sigma_c_values:
            base_mu_sc = base_mu[base_mu['sigma_c'] == sigma]
            for method_name, df_mod in modified_dfs.items():
                dm = df_mod.copy()
                pm = p_modified.get(method_name, None)
                if pm is not None:
                    dm = dm[pd.to_numeric(dm['p'], errors='coerce') == float(pm)]
                mod_mu_sc = dm[(dm['mu'] == mu) & (dm['sigma_c'] == sigma)]
                for metric in metrics:
                    sb = mean_series_by_b(base_mu_sc, metric)
                    sm = mean_series_by_b(mod_mu_sc, metric)
                    vb = value_at_b(sb, target_b, interpolate=interpolate)
                    vm = value_at_b(sm, target_b, interpolate=interpolate)
                    if vb is None or vm is None:
                        continue
                    rows.append({
                        'method': method_name,
                        'metric': metric,
                        'mu': float(mu),
                        'sigma_c': float(sigma),
                        'diff_at_b': float(vm - vb)
                    })

    diff_df_detail = pd.DataFrame(rows)
    if diff_df_detail.empty:
        raise ValueError("No differences computed. Check overlapping (μ, σ_c) and that target_b is in range or can be interpolated.")

    # --- aggregate over μ if requested ---
    if aggregate_mu in ('mean', 'median'):
        agg_fun = 'mean' if aggregate_mu == 'mean' else 'median'
        diff_df_plot = (diff_df_detail
                        .groupby(['method','metric','sigma_c'], as_index=False)['diff_at_b']
                        .agg(agg_fun)
                        .rename(columns={'diff_at_b': f'diff_at_b_{aggregate_mu}'}))
        ycol = f'diff_at_b_{aggregate_mu}'
        # plotting layout: single row (cols = metrics)
        n_rows = 1
        n_cols = len(metrics)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, figsize_rowheight * n_rows), sharex=True)
        if n_cols == 1:
            axes = [axes]

        for c, metric in enumerate(metrics):
            ax = axes[c]
            for method_name in modified_dfs.keys():
                s = (diff_df_plot[(diff_df_plot['method'] == method_name) &
                                  (diff_df_plot['metric'] == metric)]
                     .sort_values('sigma_c'))
                if s.empty:
                    continue
                style = method_styles.get(method_name, dict(marker='o', linewidth=2, label=method_name))
                ax.plot(s['sigma_c'], s[ycol], **style)

            ax.set_title(f"{metric}  |  Δ at b={target_b}  ({aggregate_mu} over μ)", fontsize=12)
            ax.set_xlabel("σ_c"); ax.set_ylabel("Δ (mod − base) at b")
            if isinstance(ylims, dict) and metric in ylims:
                ax.set_ylim(ylims[metric])
            elif isinstance(ylims, tuple):
                ax.set_ylim(ylims)
            ax.grid(False)
            if c == n_cols - 1:
                ax.legend(title='Method', loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

        plt.tight_layout(rect=[0, 0, 0.92, 1])
        plt.suptitle(f"Difference at fixed b vs σ_c  ({aggregate_mu} over μ)", y=1.02, fontsize=14)
        plt.show()

    else:
        # original layout: rows = μ, cols = metrics
        diff_df_plot = diff_df_detail.copy()
        n_rows = len(mu_values)
        n_cols = len(metrics)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, figsize_rowheight * n_rows), sharex=True)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.array([axes])
        elif n_cols == 1:
            axes = axes.reshape(n_rows, 1)

        for r, mu in enumerate(mu_values):
            for c, metric in enumerate(metrics):
                ax = axes[r, c]
                for method_name in modified_dfs.keys():
                    s = (diff_df_plot[(diff_df_plot['method'] == method_name) &
                                      (diff_df_plot['metric'] == metric) &
                                      (diff_df_plot['mu'] == mu)]
                         .sort_values('sigma_c'))
                    if s.empty:
                        continue
                    style = method_styles.get(method_name, dict(marker='o', linewidth=2, label=method_name))
                    ax.plot(s['sigma_c'], s['diff_at_b'], **style)

                ax.set_title(f"{metric}  |  μ={mu}  @ b={target_b}", fontsize=12)
                ax.set_xlabel("σ_c"); ax.set_ylabel("Δ (mod − base) at b")
                if isinstance(ylims, dict) and metric in ylims:
                    ax.set_ylim(ylims[metric])
                elif isinstance(ylims, tuple):
                    ax.set_ylim(ylims)

                ax.grid(False)
                if c == n_cols - 1:
                    ax.legend(title='Method', loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

        plt.tight_layout(rect=[0, 0, 0.92, 1])
        plt.suptitle("Difference at fixed b vs σ_c", y=1.02, fontsize=14)
        plt.show()

    return diff_df_detail, diff_df_plot

def plot_results_real_networks(
    method_folders,                # dict: {'Baseline': 'folder1', 'Method A': 'folder2', ...}
    network_name=None,             # str: network name for title (optional)
    file_glob="dmon_*.csv",        # pattern to match CSV files in each folder
    metrics=("ECS", "M1", "M2"),
    ylim_per_metric=None,          # e.g. {'ECS': (0,1), 'M1': (-0.005,0.19), 'M2': (-0.02,1.05)}
    method_styles=None,            # {'Method': {'color':..., 'linestyle':..., 'marker':...}}
    figsize=None,                  # overall figsize for single-row multi-metric figure
    show_std=True,                 # shaded std for each method curve
    fill_alpha=0.15                # alpha for std shading
):
    """Plot real-network results side-by-side (one row) for multiple methods.

    Creates ONE figure with columns = metrics; each subplot overlays all methods.
    A single shared legend (methods) is placed at the top center (like plot_results_by_p style).

    Parameters
    ----------
    method_folders : dict
        Mapping of method name -> folder path containing CSV files.
    network_name : str, optional
        Network name appended to each subplot title.
    file_glob : str
        Glob pattern for CSV files per folder.
    metrics : tuple[str]
        Metrics to plot (each becomes one column).
    ylim_per_metric : dict | None
        Optional y-limits per metric.
    method_styles : dict | None
        Per-method style dict; auto-generated if None.
    figsize : (w, h) | None
        Size of the whole figure; if None -> (5 * len(metrics), 5).
    show_std : bool
        Whether to shade mean ± std per method.
    fill_alpha : float
        Alpha for std band.

    Returns
    -------
    dict_dfs : dict
        Loaded DataFrames per method.
    fig : matplotlib.figure.Figure
        The combined figure object.
    """

    # --- Defaults ---
    if figsize is None:
        figsize = (5 * len(metrics), 5)

    cud_colors = [
        "#000000", "#E69F00", "#56B4E9", "#009E73",
        "#F0E442", "#0072B2", "#D55E00", "#CC79A7"
    ]

    if method_styles is None:
        method_names = list(method_folders.keys())
        linestyles = ['-', '--', '-.', ':']
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
        method_styles = {}
        for i, m in enumerate(method_names):
            method_styles[m] = {
                'color': cud_colors[i % len(cud_colors)],
                'linestyle': linestyles[i % len(linestyles)],
                'marker': markers[i % len(markers)]
            }

    labels = {
        'ECS': r"$ECS$",
        'M1': r"$M_1$",
        'M2': r"$M_2$"
    }

    # --- Load data ---
    dict_dfs = {}
    for method_name, folder in method_folders.items():
        csv_files = glob.glob(f"{folder}/{file_glob}")
        print(f"File name {csv_files}")
        if not csv_files:
            print(f"Warning: No CSV files matched in {folder} with pattern {file_glob}")
            continue
        frames = [pd.read_csv(fp, keep_default_na=False) for fp in csv_files]
        dict_dfs[method_name] = pd.concat(frames, ignore_index=True)

    if not dict_dfs:
        raise FileNotFoundError("No data loaded from any method folder")

    # --- Figure layout: one row ---
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, sharex=False, sharey=False)
    if len(metrics) == 1:
        axes = [axes]

    # Collect method legend entries once
    method_handles = {}

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for method_name, df in dict_dfs.items():
            grouped = (df.groupby('b_percentage')
                         .agg({metric: ['mean', 'std']})
                         .reset_index())
            if grouped.empty or (metric, 'mean') not in grouped.columns:
                continue
            means = grouped[(metric, 'mean')].astype(float)
            stds  = grouped[(metric, 'std')].astype(float).fillna(0.0)
            xvals = (grouped['b_percentage'].astype(float) * 100.0).values

            style = method_styles.get(method_name, {})
            line = ax.plot(
                xvals, means.values,
                marker=style.get('marker', 'o'),
                linestyle=style.get('linestyle', '-'),
                color=style.get('color', '#000000'),
                label=method_name,
                linewidth=2,
                markersize=6
            )[0]
            if show_std:
                ax.fill_between(
                    xvals,
                    (means - stds).values,
                    (means + stds).values,
                    color=style.get('color', '#000000'),
                    alpha=fill_alpha
                )
            # keep first handle per method for legend
            if method_name not in method_handles:
                method_handles[method_name] = line

        title = labels.get(metric, metric)
        if network_name:
            title = f"{title} – {network_name}"
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(r'Budget $\beta_b$ (% intra-community edges)', fontsize=12)
        ax.set_ylabel(labels.get(metric, metric), fontsize=12)
        if isinstance(ylim_per_metric, dict) and metric in ylim_per_metric:
            ax.set_ylim(ylim_per_metric[metric])
        ax.grid(False)

    # --- Shared legend (top center like plot_results_by_p) ---
    if method_handles:
        handles = list(method_handles.values())
        legend_labels = list(method_handles.keys())
        leg = fig.legend(
            handles, legend_labels,
            loc='upper center',
            ncol=len(legend_labels),
            frameon=True,
            fontsize=12,
            markerscale=0,
            bbox_to_anchor=(0.5, 1.03)
        )
        for h in leg.legend_handles:
            try:
                h.set_linewidth(3.0)
                h.set_marker('None')
            except Exception:
                pass

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return dict_dfs, fig
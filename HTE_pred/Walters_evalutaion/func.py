import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd


mpl.rcParams['font.family'] = 'Arial'
sns.set_context('notebook')
sns.set_style('whitegrid', {'font.family': ['Arial']})

def mcs_plot_auto_vlim(pc, effect_size, means, labels=True, cmap=None, cbar_ax_bbox=None,
                       ax=None, show_diff=True, cell_text_size=16, axis_text_size=12,
                       show_cbar=True, reverse_cmap=False, vlim=None, vlim_method='data_range', **kwargs):
    """
    Create a multiple comparison of means plot using a heatmap with automatic vlim determination.

    """
    for key in ['cbar', 'vmin', 'vmax', 'center']:
        if key in kwargs:
            del kwargs[key]

    if not cmap:
        cmap = "coolwarm"
    if reverse_cmap:
        cmap = cmap + "_r"

    significance = pc.copy().astype(object)
    significance[(pc < 0.001) & (pc >= 0)] = '***'
    significance[(pc < 0.01) & (pc >= 0.001)] = '**'
    significance[(pc < 0.05) & (pc >= 0.01)] = '*'
    significance[(pc >= 0.05)] = ''

    np.fill_diagonal(significance.values, '')

    # 自动确定 vlim
    if vlim is None:
        if vlim_method == 'data_range':
            max_abs = np.abs(effect_size.values).max()
            vlim = max_abs / 2 if max_abs > 0 else 1.0
        elif vlim_method == 'percentile':
            upper = np.percentile(np.abs(effect_size.values), 95)
            vlim = upper / 2 if upper > 0 else 1.0
        elif vlim_method == 'std':
            std = effect_size.values.std()
            vlim = std if std > 0 else 1.0
        else:
            raise ValueError(f"Unknown vlim_method: {vlim_method}")

    # Create a DataFrame for the annotations
    if show_diff:
        annotations = effect_size.round(3).astype(str) + significance
    else:
        annotations = significance

    hax = sns.heatmap(effect_size, cmap=cmap, annot=annotations, fmt='', cbar=show_cbar, ax=ax,
                      annot_kws={"size": cell_text_size},
                      vmin=-2*vlim if vlim else None, vmax=2*vlim if vlim else None, **kwargs)

    if labels:
        label_list = list(means.index)
        x_label_list = [x + f'\n{means.loc[x].round(2)}' for x in label_list]
        y_label_list = [x + f'\n{means.loc[x].round(2)}\n' for x in label_list]
        hax.set_xticklabels(x_label_list, size=axis_text_size, ha='center', va='top', rotation=0,
                            rotation_mode='anchor')
        hax.set_yticklabels(y_label_list, size=axis_text_size, ha='center', va='center', rotation=90,
                            rotation_mode='anchor')

    hax.set_xlabel('')
    hax.set_ylabel('')

    return hax

def tukey_hsd_with_abbreviations(df, metric, group_col, alpha=0.05, sort=False, direction_dict=None):
    """
    use pairwise_tukeyhsd for Tukey HSD test

    """
    method_abbreviations = {
        "OneHot_SMILES": "OHE",
        "Fingerprint": "MACCS",
        "RDKit_Descriptors": "RDKit",
        "Mordred": "Mordred",
        "Uni-Mol": "Uni-Mol"
    }

    df_copy = df.copy()
    df_copy[group_col] = df_copy[group_col].replace(method_abbreviations)

    if sort and direction_dict and metric in direction_dict:
        if direction_dict[metric] == 'maximize':
            df_means = df_copy.groupby(group_col).mean(
                numeric_only=True).sort_values(metric, ascending=False)
        elif direction_dict[metric] == 'minimize':
            df_means = df_copy.groupby(group_col).mean(
                numeric_only=True).sort_values(metric, ascending=True)
        else:
            raise ValueError("无效的方向。期望 'maximize' 或 'minimize'。")
    else:
        df_means = df_copy.groupby(group_col).mean(numeric_only=True)

    # run Tukey HSD 
    tukey = pairwise_tukeyhsd(endog=df_copy[metric],
                              groups=df_copy[group_col],
                              alpha=alpha)

    result_tab = pd.DataFrame(data=tukey._results_table.data[1:],
                              columns=tukey._results_table.data[0])
    methods = df_means.index
    df_means_diff = pd.DataFrame(index=methods, columns=methods, data=0.0)
    pc = pd.DataFrame(index=methods, columns=methods, data=1.0)
    for _, row in result_tab.iterrows():
        group1 = row['group1']
        group2 = row['group2']
        meandiff = row['meandiff']
        p_adj = row['p-adj']

        df_means_diff.loc[group1, group2] = meandiff
        df_means_diff.loc[group2, group1] = -meandiff
        pc.loc[group1, group2] = p_adj
        pc.loc[group2, group1] = p_adj

    df_means_diff = df_means_diff.astype(float)
    result_tab["group1_mean"] = result_tab["group1"].map(df_means[metric])
    result_tab["group2_mean"] = result_tab["group2"].map(df_means[metric])

    result_tab.index = result_tab['group1'] + ' - ' + result_tab['group2']

    return result_tab, df_means, df_means_diff, pc


def ci_plot_with_abbreviations(result_tab, ax_in, name, auto_xlim=True, xlim_factor=1.2):
    """
    Create a confidence interval plot for the given result table with support for method abbreviations and automatic xlim.

    Parameters:
    result_tab (pd.DataFrame): DataFrame containing the results with columns 'meandiff', 'lower', and 'upper'.
    ax_in (matplotlib.axes.Axes): The axes on which to plot the confidence intervals.
    name (str): The title of the plot.
    auto_xlim (bool): Whether to automatically determine the x-axis limits. Default is True.
    xlim_factor (float): Factor to multiply the max absolute value when determining xlim. Default is 1.2.

    Returns:
    None
    """
    result_err = np.array([result_tab['meandiff'] - result_tab['lower'],
                           result_tab['upper'] - result_tab['meandiff']])
    sns.set(rc={'figure.figsize': (6, 2)})
    sns.set_context('notebook')
    sns.set_style('whitegrid')
    ax = sns.pointplot(x=result_tab.meandiff, y=result_tab.index, marker='o', linestyle='', ax=ax_in)
    ax.errorbar(y=result_tab.index, x=result_tab['meandiff'], xerr=result_err, fmt='o', capsize=5)
    ax.axvline(0, ls="--", lw=3)
    ax.set_xlabel("Mean Difference")
    ax.set_ylabel("")
    ax.set_title(name)
    
    if auto_xlim:
        max_abs = max(abs(result_tab['meandiff'].min()), abs(result_tab['meandiff'].max()))
        max_abs = max(max_abs, abs(result_tab['lower'].min()), abs(result_tab['upper'].max()))
        xlim = max_abs * xlim_factor
        ax.set_xlim(-xlim, xlim)
    else:
        ax.set_xlim(-0.2, 0.2)


def make_boxplots_parametric(df, metric_ls):
    """
    Create boxplots for each metric using repeated measures ANOVA.
    """

    metric_ls = [m for m in metric_ls if m.lower() not in ['prec', 'recall']]

    method_abbreviations = {
        "OneHot_SMILES": "OHE",
        "Fingerprint": "MACCS",
        "RDKit_Descriptors": "RDKit",
        "Mordred": "Mordred",
        "Uni-Mol": "Uni-Mol"
    }

    df_abbrev = df.copy()
    df_abbrev['method_abbrev'] = df_abbrev['method'].map(
        method_abbreviations).fillna(df_abbrev['method'])

    n_metrics = len(metric_ls)
    n_cols = 2
    n_rows = math.ceil(n_metrics / n_cols)

    figure, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows),
                                sharex=False, sharey=False, dpi=600)
    axes = axes.flatten()

    for i, stat in enumerate(metric_ls):
        model = AnovaRM(data=df_abbrev, depvar=stat,
                        subject='cv_cycle', within=['method_abbrev']).fit()
        p_value = model.anova_table['Pr > F'].iloc[0]

        ax = sns.boxplot(y=stat, x="method_abbrev", hue="method_abbrev", ax=axes[i],
                         data=df_abbrev, palette="Set2", legend=False)

        ax.set_title(f"{stat.upper()}  (p={p_value:.1e})", fontfamily='Arial')
        ax.set_xlabel("")
        ax.set_ylabel(stat.upper(), fontfamily='Arial')

        ax.set_xticks(list(range(len(ax.get_xticklabels()))))
        ax.set_xticklabels([lbl.get_text() for lbl in ax.get_xticklabels()],
                           rotation=0, fontfamily='Arial')

    for i in range(n_metrics, len(axes)):
        axes[i].axis('off')

    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    plt.tight_layout()


def make_mcs_plot_grid_auto_vlim(df, stats, group_col, alpha=.05,
                                 figsize=(12, 12), direction_dict={}, show_diff=True,
                                 cell_text_size=10, axis_text_size=12, title_text_size=16, 
                                 sort_axes=False, vlim_method='data_range'):
    """
    Create a fixed 2x2 grid of multiple comparison plots using Tukey HSD test results.
    """
    stats = [s for s in stats if s.lower() not in ['prec', 'recall']]

    # 固定 2 行 2 列
    nrow, ncol = 2, 2
    fig, ax = plt.subplots(nrow, ncol, figsize=figsize, dpi=1440)
    ax = ax.flatten()

    # 设置回归任务的默认方向
    for key in ['r2', 'rho', 'mae', 'mse']:
        direction_dict.setdefault(key, 'maximize' if key in ['r2', 'rho'] else 'minimize')
    direction_dict = {k.lower(): v for k, v in direction_dict.items()}

    for i, stat in enumerate(stats):
        stat = stat.lower()
        if stat not in direction_dict:
            raise ValueError(f"Stat '{stat}' is missing in direction_dict.")

        reverse_cmap = (direction_dict[stat] == 'minimize')

        _, df_means, df_means_diff, pc = tukey_hsd_with_abbreviations(
            df, stat, group_col, alpha, sort_axes, direction_dict
        )

        hax = mcs_plot_auto_vlim(
            pc, effect_size=df_means_diff, means=df_means[stat],
            show_diff=show_diff, ax=ax[i], cbar=True,
            cell_text_size=cell_text_size, axis_text_size=axis_text_size,
            reverse_cmap=reverse_cmap, vlim_method=vlim_method
        )
        hax.set_title(stat.upper(), fontsize=title_text_size, fontfamily='Arial')

    # 隐藏多余的子图
    for i in range(len(stats), nrow * ncol):
        ax[i].set_visible(False)

    plt.tight_layout()



def make_ci_plot_grid_with_abbreviations(df_in, metric_list, group_col="method", alpha=0.05, 
                                        auto_xlim=True, xlim_factor=1.2, height_per_plot=3.5):
    """
    """
    metric_list = [m for m in metric_list if m.lower() not in ['prec', 'recall']]

    figure, axes = plt.subplots(len(metric_list), 1,
                                figsize=(10, height_per_plot * len(metric_list)), sharex=False, dpi=1000)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for i, metric in enumerate(metric_list):
        df_tukey, _, _, _ = tukey_hsd_with_abbreviations(df_in, metric, group_col=group_col, alpha=alpha)
        ci_plot_with_abbreviations(df_tukey, ax_in=axes[i], name=metric,
                                   auto_xlim=auto_xlim, xlim_factor=xlim_factor)

    figure.suptitle("Comparison of Tukey HSD",
                     fontfamily='Arial') # y=1.02,
    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout()

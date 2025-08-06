import os

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def process_data_frame(df, df_baseline):
    df_1 = df.copy(deep=True)
    df_2 = df.copy(deep=True)

    # df_1 = df_1 / xavier
    for tgt_task in df_1.axes[1].to_list():
        if df_1.iloc[0][tgt_task] == 0:
            continue
        df_1[tgt_task] = df_1[tgt_task] / df_1.iloc[0][tgt_task]

    # df_2 = df_2 / baseline
    for tgt_task in df_2.axes[1].to_list():
        df_2[tgt_task] = df_2[tgt_task] / df_baseline[tgt_task]

    return df_1, df_2


def display_n_save_heatmap(df, fig_w, fig_h, heatmap_kw, axis_font_size, title):
    plt.figure(figsize=(fig_w, fig_h))
    sns.heatmap(df, **heatmap_kw)
    plt.xticks(fontsize=axis_font_size)
    plt.yticks(fontsize=axis_font_size)
    plt.title(title)
    plt.savefig('visualization/figures/{}.svg'.format(title), format='svg')
    plt.show()


if __name__ == "__main__":
    # Visualization parameters
    fig_w = 15
    fig_h = 17
    heatmap_kw = {
        # 'vmin': 0.8,
        # 'vmax': 1.2,
        'cmap': 'Purples',
        'annot': True,
        'fmt': '.2f',
        'square': True,
        'cbar': False,
        'annot_kws': {"fontsize": 15}}
    axis_font_size = 22

    print(os.getcwd())
    os.chdir('../')
    print(os.getcwd())

    # Read the csv file
    frozen_prompt_mat = pd.read_csv('Logs/csv/FrozenPromptTransfer-New.csv', index_col=0)
    projection_mat = pd.read_csv('Logs/csv/ProjectionTransfer-New.csv', index_col=0)
    baseline_mat = pd.read_csv('Logs/csv/Baselines.csv', index_col=0)
    mmd_mat = pd.read_csv('analysis/ground_truth/MMD_TRAIN.csv', index_col=0)

    frozen_prompt_mat_xavier, frozen_prompt_mat_baseline = process_data_frame(frozen_prompt_mat, baseline_mat.iloc[0])
    projection_mat_xavier, projection_mat_baseline = process_data_frame(projection_mat, baseline_mat.iloc[1])

    display_n_save_heatmap(frozen_prompt_mat_xavier, fig_w, fig_h, heatmap_kw, axis_font_size,
                           title='heatmap_frozen_xavier')
    display_n_save_heatmap(frozen_prompt_mat_baseline, fig_w, fig_h, heatmap_kw, axis_font_size,
                           title='heatmap_frozen_baseline')

    display_n_save_heatmap(projection_mat_xavier, fig_w, fig_h, heatmap_kw, axis_font_size,
                           title='heatmap_projection_xavier')
    display_n_save_heatmap(projection_mat_baseline, fig_w, fig_h, heatmap_kw, axis_font_size,
                           title='heatmap_projection_baseline')

    heatmap_kw['cmap'] = 'Purples_r'
    heatmap_kw['vmin'] = 0
    # heatmap_kw['vmax'] = 2
    display_n_save_heatmap(mmd_mat.T, fig_w, fig_h, heatmap_kw, axis_font_size,
                           title='heatmap_modality_gap')

    mmd_np = mmd_mat.T.values
    # projection_np = np.round(projection_mat_baseline.values, decimals=2)
    projection_np = projection_mat_baseline.values
    for threshold in [3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5]:
        filtered = projection_np[mmd_np > threshold]

        print('Threshold: {}; Percentage: {}'.format(threshold, np.sum(filtered < 1) / len(filtered)))


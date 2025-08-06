import os
import pandas as pd

from transferability.utils import *

def parse_gt_csv(split):

    # Read the csv file
    mmd_gt = pd.read_csv('analysis/ground_truth/MMD_{}.csv'.format(split.upper()), index_col=0)

    # Get the name for source and target tasks
    target_tasks = mmd_gt.axes[0].to_list()
    source_tasks = mmd_gt.axes[1].to_list()

    # Iterate through the target tasks
    meta_dict = {}
    for target_task in target_tasks:
        meta_dict[target_task] = {}

        # Iterate through the source tasks
        for source_task in source_tasks:
            if source_task == 'xavier':
                continue
            meta_dict[target_task][source_task] = - mmd_gt.loc[target_task, source_task]

    return meta_dict


if __name__ == '__main__':

    print(os.getcwd())
    os.chdir('../')
    print(os.getcwd())

    # Manual parameters
    tranfer_task = 'ProjectionTransfer'  # ProjectionTransfer, FrozenPromptTransfer
    src_model = 'roberta'
    split = 'train'

    # Get the ground-truth dictionaries for mmd results
    mmd_meta_gt = parse_gt_csv(split)

    for target_task in list(mmd_meta_gt.keys()):

        # Get the ground-truth dictionary for projection transfer results
        ground_truth_dict, _ = get_ground_truth(tranfer_task, src_model, target_task)

        # Print the prediction results for this target task
        print('On {}'.format(target_task.upper()))
        print_prediction_results(results_dict=mmd_meta_gt[target_task],
                                 ground_truth_dict=ground_truth_dict)

        # Print the evaluation scores
        kendall = kendalls_coefficient(mmd_meta_gt[target_task], ground_truth_dict)
        spearman = spearmans_rank_correlation(mmd_meta_gt[target_task], ground_truth_dict)
        print("Kendall's coefficient score: {:.2f}".format(kendall * 100))
        print("Spearman's rank correlation: {:.2f}".format(spearman * 100))
        print()
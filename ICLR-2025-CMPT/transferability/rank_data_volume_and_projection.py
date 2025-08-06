import os
import pandas as pd

from transferability.utils import *

gt_dict = {'imdb': 25000,
           'sst2': 67349,
           'laptop': 3045,
           'restaurant': 3041,
           'movie': 1600,
           'tweet': 45389,
           'mnli': 392702,
           'qnli': 104743,
           'snli': 549367,
           'deontology': 18164,
           'justice': 21791,
           'qqp': 363846,
           'mrpc': 3668}

if __name__ == '__main__':

    print(os.getcwd())
    os.chdir('../')
    print(os.getcwd())

    # Manual parameters
    tranfer_task = 'ProjectionTransfer'  # ProjectionTransfer, FrozenPromptTransfer
    src_model = 'roberta'
    tgt_tasks = ['caltech101', 'cifar', 'dtd', 'oxford_flowers102', 'oxford_iiit_pet', 'sun397', 'svhn',
                 'patch_camelyon', 'resisc45', 'eurosat', 'diabetic_retinopathy',
                 'dmlab', 'kitti', 'smallnorb_azi', 'smallnorb_ele', 'dsprites_loc', 'dsprites_ori', 'clevr_dist', 'clevr_count']

    for tgt_task in tgt_tasks:
        # Get the ground-truth dictionary for projection transfer results
        ground_truth_dict, _ = get_ground_truth(tranfer_task, src_model, tgt_task)

        # Print the prediction results for this target task
        print('On {}'.format(tgt_task.upper()))
        print_prediction_results(results_dict=gt_dict, ground_truth_dict=ground_truth_dict)

        # Print the evaluation scores
        kendall = kendalls_coefficient(gt_dict, ground_truth_dict)
        spearman = spearmans_rank_correlation(gt_dict, ground_truth_dict)
        print("Kendall's coefficient score: {:.2f}".format(kendall * 100))
        print("Spearman's rank correlation: {:.2f}".format(spearman * 100))
        print()

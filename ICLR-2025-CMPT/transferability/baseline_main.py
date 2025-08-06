import os

from transferability.utils import *

if __name__ == '__main__':
    print(os.getcwd())
    os.chdir('../')
    print(os.getcwd())

    # Parameters
    device = 'cuda'

    src_model = 'roberta'
    src_tasks = ('imdb', 'sst2', 'laptop', 'restaurant', 'movie', 'tweet',
                 'mnli', 'qnli', 'snli',
                 'deontology', 'justice',
                 'qqp', 'mrpc')

    tgt_tasks = ['caltech101', 'cifar', 'dtd', 'oxford_flowers102', 'oxford_iiit_pet', 'sun397', 'svhn',
                 'patch_camelyon', 'resisc45', 'eurosat', 'diabetic_retinopathy',
                 'dmlab', 'kitti', 'smallnorb_azi', 'smallnorb_ele', 'dsprites_loc', 'dsprites_ori', 'clevr_dist', 'clevr_count']
    metric = 'COSINE_AVERAGE'  # MODEL_ACTIVATION, COSINE_AVERAGE

    norm = False
    whiten = False
    transfer_task = 'ProjectionTransfer'  # ProjectionTransfer, FrozenPromptTransfer
    involve_mmd = False

    for tgt_task in tgt_tasks:

        # Get the ground truth result
        ground_truth_dict, _ = get_ground_truth(transfer_task=transfer_task,
                                                src_model=src_model,
                                                tgt_task=tgt_task)

        # Read the target prompt
        tgt_prompt = load_target_prompt(tgt_task, norm=norm, whiten=whiten, device=device)

        # Get the estimation results
        results_dict = get_transferability_results(src_model,
                                                   src_tasks,
                                                   tgt_prompt,
                                                   norm,
                                                   whiten,
                                                   projection=None,
                                                   device=device,
                                                   metric=metric)

        # If you want to combine the results from MMD
        if involve_mmd:
            from transferability.rank_mmd_and_projection import parse_gt_csv

            mmd_meta_gt = parse_gt_csv('train')
            mmd_gt = mmd_meta_gt[tgt_task]

            assert mmd_gt.keys() == results_dict.keys()

            for source_task in mmd_gt.keys():
                # results_dict_final[source_task] = results_dict_final[source_task] + sigmoid(mmd_gt[source_task])
                results_dict[source_task] = results_dict[source_task] + mmd_gt[source_task]

        # Print the estimation results
        print_prediction_results(results_dict=results_dict, ground_truth_dict=ground_truth_dict, target_task=tgt_task)

        # Calculate the ranking scores
        kendall = kendalls_coefficient(results_dict, ground_truth_dict)
        spearman = spearmans_rank_correlation(results_dict, ground_truth_dict)
        print("Kendall's coefficient score: {:.2f}".format(kendall * 100))
        print("Spearman's rank correlation: {:.2f}".format(spearman * 100))

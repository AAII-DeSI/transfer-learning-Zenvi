import torch
import torch.optim as optim

from transferability.utils import *
from transferability.metrics.loss_functions import DistanceAverageEmbedding, DistanceInstanceEmbedding

from train.train_utils import print_sys_info, set_random_seed

def train(config, target_task, device):

    # Get the ground truth transfer performance
    ground_truth_dict, _ = get_ground_truth(config['basic']['transfer_task'],
                                            config['Source']['model'],
                                            target_task)

    # Load all the source prompts
    src_prompts = load_source_prompts_multi(source_tasks=config['Source']['tasks'],
                                            norm=config['basic']['norm'],
                                            whiten=config['basic']['whiten'],
                                            src_model=config['Source']['model'],
                                            device=device)

    # Load the target prompt
    tgt_prompt = load_target_prompt(target_task=target_task,
                                    norm=config['basic']['norm'],
                                    whiten=config['basic']['whiten'],
                                    device=device)

    results_dict_list = []
    for seed in (42, 44, 100):

        print('On seed: {}...'.format(seed))

        set_random_seed(seed, reproducibility=True)

        # Create the universal projector
        universal_projector = UniversalProjector(device=device,
                                                 opt_dim=config['Projector']['opt_dim'],
                                                 use_src=config['Projector']['use_src'],
                                                 src_mix=config['Projector']['src_mix'],
                                                 src_noise=config['Projector']['src_noise'],
                                                 use_tgt=config['Projector']['use_tgt'],
                                                 dropout=config['Projector']['dropout'])
        universal_projector.to(device)

        optimizer = optim.Adam(universal_projector.parameters(), lr=config['basic']['lr'], weight_decay=config['basic']['wd'])
        loss_fn = eval(config['Loss']['function'])(config['Loss']['metric'], config['Loss']['reduction'])

        for epoch in range(100):

            # # Shuffle the source prompts
            # src_prompts_shuffled = src_prompts.get_batch()

            # Forward pass
            src_prompts_projected, tgt_prompt_projected = universal_projector(src_prompts, tgt_prompt)

            loss = loss_fn(src_prompts_projected, tgt_prompt_projected)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate the universal projector, we only evaluate it after training for an entire seed
        results_dict = get_transferability_results(src_model=config['Source']['model'],
                                                   src_tasks=config['Source']['tasks'],
                                                   tgt_prompt=tgt_prompt,
                                                   norm=config['basic']['norm'],
                                                   whiten=config['basic']['whiten'],
                                                   projection=universal_projector,
                                                   device=device,
                                                   metric=config['Loss']['evaluation_metric'])
        results_dict_list.append(results_dict)

    # Now, merge 'results_dict_list' into a single list
    results_dict_final = {}
    for source_task in config['Source']['tasks']:
        t_score = 0
        for results_dict in results_dict_list:
            t_score += results_dict[source_task]
        results_dict_final[source_task] = t_score/len(results_dict_list)

    # If you want to combine the results from MMD
    if config['MMD']['involve']:
        from transferability.rank_mmd_and_projection import parse_gt_csv
        mmd_meta_gt = parse_gt_csv('train')
        mmd_gt = mmd_meta_gt[target_task]

        assert mmd_gt.keys() == results_dict_final.keys()

        for source_task in mmd_gt.keys():
            lambda_t = config['MMD']['lambda_t']
            lambda_m = config['MMD']['lambda_m']
            mmd_processor_type = config['MMD']['mmd_processor_type']
            results_dict_final[source_task] = lambda_t * results_dict_final[source_task] - lambda_m * mmd_processor(mmd_gt[source_task], type=mmd_processor_type)

    # Print the transferability estimation results, along with the ground truths
    print_prediction_results(results_dict_final, ground_truth_dict, target_task)

    # How well is the transferability estimated?
    kendall = kendalls_coefficient(results_dict_final, ground_truth_dict)
    spearman = spearmans_rank_correlation(results_dict_final, ground_truth_dict)
    print("Kendall's coefficient score: {:.2f}".format(kendall * 100))
    print("Spearman's rank correlation: {:.2f}".format(spearman * 100))

    return kendall, spearman, results_dict_final


def print_config(config):

    print('<{:-^100s}>'.format('Basic Parameters'))
    for k, v in config['basic'].items():
        print('{}: {}'.format(k, v))

    print('<{:-^100s}>'.format('Projector Parameters'))
    for k, v in config['projector'].items():
        print('{}: {}'.format(k, v))

    print('<{:-^100s}>'.format('Training Parameters'))
    for k, v in config['train'].items():
        print('{}: {}'.format(k, v))

if __name__ == '__main__':

    print(os.getcwd())
    os.chdir('../')
    print(os.getcwd())

    # Manual parameters
    device = print_sys_info(0)

    # Read the config file
    config = yaml.load(open('transferability/configs/UniversalProjector.yaml'), yaml.FullLoader)
    # print_config(config)

    # Main function
    kendalls, spearmans = [], []
    for target_task in config['Target']['tasks']:
        kendall, spearman, results_dict_final = train(config, target_task, device)
        kendalls.append(kendall)
        spearmans.append(spearman)

    print(torch.mean(torch.Tensor(kendalls)))
    print(torch.mean(torch.Tensor(spearmans)))
    print(torch.mean(torch.Tensor(list(results_dict_final.values()))))
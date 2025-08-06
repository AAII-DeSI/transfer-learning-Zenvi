import os
import yaml
import csv
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from transferability.metrics import on, spot

task_dict = {'imdb': 'IMDB',
             'sst2': 'SST2',
             'laptop': 'laptop',
             'restaurant': 'restaurant',
             'movie': 'movierationales',
             'tweet': 'tweetevalsentiment',
             'mnli': 'MNLI',
             'qnli': 'QNLI',
             'snli': 'snli',
             'deontology': 'ethicsdeontology',
             'justice': 'ethicsjustice',
             'qqp': 'QQP',
             'mrpc': 'MRPC'}

model_dict = {'bert': 'Bert',
              'roberta': 'Roberta',
              't5': 'T5'}


def mmd_processor(x, type=None):
    if type == 'log':
        return np.log(-x)
    elif type == 'sinh':
        return np.sinh(-x)
    else:
        return -x


def load_target_prompt(target_task, norm=False, whiten=False, prompt_init='xavier', device='cpu'):
    base_dir = 'VPT_100_prompt/'

    # Load the prompt embeddings that are trained on target image datasets. Using only xavier init without projection
    best_accuracy = 0
    best_path = ''
    for seed in os.listdir(base_dir):

        for prompt_dir in os.listdir(os.path.join(base_dir, seed)):

            if prompt_dir == '{}_{}_none'.format(prompt_init, target_task):

                for ckpt in os.listdir(os.path.join(base_dir, seed, prompt_dir)):

                    if 'prompt' in ckpt:

                        accuracy = eval(ckpt.split('(')[1].rstrip(')').replace('_', '.'))

                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_path = os.path.join(base_dir, seed, prompt_dir, ckpt)

    print('<{:-^100s}>'.format('Target prompt information'))
    print('The path to the best target prompt:', best_path.replace('\\', '/'))  # Just looks good
    print('The accuracy the target prompt achieves on {}:'.format(target_task), best_accuracy)

    target_prompt = torch.load(best_path, map_location=device)

    if norm:
        target_prompt = F.normalize(target_prompt, p=2, dim=1)

    if whiten:
        whitener = nn.LayerNorm(target_prompt.shape[1], device=device)

        with torch.no_grad():
            target_prompt = whitener(target_prompt)

    return target_prompt


def load_source_prompt(src_task, norm=False, whiten=False, src_model='roberta', device='cpu'):
    path = os.path.join('task_prompt_emb',
                        task_dict[src_task.lower()] + 'Prompt' + model_dict[src_model.lower()],
                        'task_prompt')

    source_prompt = torch.load(path, map_location=device)
    # source_prompt = source_prompt[1:]

    if norm:
        source_prompt = F.normalize(source_prompt, p=2, dim=1)

    if whiten:
        whitener = nn.LayerNorm(source_prompt.shape[1], device=device)

        with torch.no_grad():
            source_prompt = whitener(source_prompt)

    return source_prompt


def load_source_prompts_multi(source_tasks, norm=False, whiten=False, src_model='roberta', device='cpu'):
    """
    Load multiple source prompts given the source tasks and source model.
    """

    source_prompt_ori = []

    for source_task in source_tasks:
        source_prompt = load_source_prompt(source_task, norm, whiten, src_model, device)
        source_prompt_ori.append(source_prompt.unsqueeze(0))

    source_prompt_ori = torch.cat(source_prompt_ori, dim=0)

    return source_prompt_ori.to(device)


def kendalls_coefficient(results_dict: dict, ground_truth_dict: dict):
    # The two dictionaries should have exactly the same keys
    for key in results_dict.keys():
        assert key in ground_truth_dict.keys()

    results_list = [x[1] for x in sorted(list(results_dict.items()), key=lambda x: x[0], reverse=False)]
    ground_truth_list = [x[1] for x in sorted(list(ground_truth_dict.items()), key=lambda x: x[0], reverse=False)]

    assert len(results_list) == len(ground_truth_list)

    signs = 0
    num_iters = 0
    for i in range(len(results_list) - 1):  # 0, 1, 2, 3

        for j in range(i + 1, len(results_list)):  # 1, 2, 3, 4

            sign_gt = ground_truth_list[i] - ground_truth_list[j]
            if sign_gt != 0:
                sign_gt = 1 if sign_gt > 0 else -1

            sign_rs = results_list[i] - results_list[j]
            if sign_rs != 0:
                sign_rs = 1 if sign_rs > 0 else -1

            signs += sign_gt * sign_rs
            num_iters += 1

    return signs / num_iters


def spearmans_rank_correlation(results_dict: dict, ground_truth_dict: dict):
    # The two dictionaries should have exactly the same keys
    for key in results_dict.keys():
        assert key in ground_truth_dict.keys()

    # Make sure the two 'dictionaries' have exactly the same order
    results_list = sorted(list(results_dict.items()), key=lambda x: x[0], reverse=False)
    ground_truth_list = sorted(list(ground_truth_dict.items()), key=lambda x: x[0], reverse=False)

    # Rank them
    results_order = [sorted(results_list, key=lambda x: x[1], reverse=True).index(e) for e in results_list]
    ground_truth_order = [sorted(ground_truth_list, key=lambda x: x[1], reverse=True).index(e) for e in
                          ground_truth_list]

    # Calculate d^2
    dd = [(results_order[i] - ground_truth_order[i]) ** 2 for i in range(len(results_order))]

    rho = 1 - ((6 * sum(dd)) / (len(results_order) * (len(results_order) ** 2 - 1)))

    return rho


def print_prediction_results(results_dict, ground_truth_dict, target_task=''):
    # Print the bigger information
    print('<{:-^100s}>'.format(target_task))

    results = sorted(list(results_dict.items()), key=lambda x: x[1], reverse=True)
    ground_truth = sorted(list(ground_truth_dict.items()), key=lambda x: x[1], reverse=True)

    # Your table head
    table_head = '{:<4s}\t{:<10s}\t{:<5s}\t{:<12s}\t{:<8s}'.format('Rank', 'Predicted', 'Score', 'Ground Truth',
                                                                   'Accuracy')
    table_rows = '{:<4}\t{:<10}\t{:<5.2f}\t{:<12}\t{:<8}'

    # Print table head
    print(table_head)

    # Print table content
    for idx in range(len(results)):

        print(table_rows.format(idx, results[idx][0], results[idx][1], ground_truth[idx][0], ground_truth[idx][1]),
              end='')
        if results[idx][0] == ground_truth[idx][0]:
            print('BINGO!', end='')
        print()


def get_ground_truth(transfer_task, src_model, tgt_task):
    if transfer_task in ('ProjectionTransfer', 'FrozenPromptTransfer'):

        f = open('transferability/ground_truths/{}.yaml'.format(transfer_task))
        y = yaml.load(f, yaml.FullLoader)

        ground_truth_dict = y['{}_to_{}'.format(src_model, tgt_task)]

        for key, value in ground_truth_dict.items():
            ground_truth_dict[key] = eval(value)
        if 'xavier' in ground_truth_dict.keys():
            ground_truth_dict.pop('xavier')

        return ground_truth_dict, None

    elif transfer_task == 'AttentionTransfer':

        f = open('transferability/ground_truths/AttentionTransfer.yaml')
        y = yaml.load(f, yaml.FullLoader)

        meta_dict = y['{}_to_{}'.format(src_model, tgt_task)]

        ground_truth_dict, actu_length_dict = {}, {}

        for key, value in meta_dict.items():

            if '??.??' in value:
                ground_truth_dict[key] = 0.00
                actu_length_dict[key] = 0
                continue

            meta_dict[key] = list(map(eval, meta_dict[key]))
            ground_truth_dict[key] = max(meta_dict[key])
            actu_length_dict[key] = 100 - meta_dict[key].index(max(meta_dict[key])) * 10

        return ground_truth_dict, actu_length_dict

    else:
        raise NotImplementedError


def get_transferability_results(src_model,
                                src_tasks,
                                tgt_prompt,
                                norm=False,
                                whiten=False,
                                projection=None,
                                device='cpu',
                                metric='COSINE_AVERAGE'):
    transferability_metric_dict = {'COSINE_AVERAGE': spot.cosine_average,
                                   'COSINE_MAX': spot.cosine_max,
                                   'COSINE_INDIVIDUAL': spot.cosine_individual,
                                   'EUCLIDEAN_AVERAGE': spot.euclidean_average,
                                   'MODEL_ACTIVATION': on.model_stimulation_similarity}

    metric_func = transferability_metric_dict[metric]

    if projection is not None:
        projection.eval()

    results_dict = {}
    for src_task in src_tasks:

        src_prompt = load_source_prompt(src_task, norm, whiten, src_model, device=device)

        if projection is not None:
            with torch.no_grad():
                src_prompt_proj, tgt_prompt_proj = projection(src_prompt, tgt_prompt)
        else:
            src_prompt_proj, tgt_prompt_proj = src_prompt, tgt_prompt

        t = metric_func(src_prompt_proj, tgt_prompt_proj, device=device)

        results_dict[src_task] = t.tolist()

    return results_dict


def evaluate(src_model,
             src_tasks,
             tgt_prompt,
             norm,
             whiten,
             ground_truth_dict,
             projection=None,
             device='cpu',
             metric='Cosine Similarity of Averaged Token'):
    results_dict = get_transferability_results(src_model=src_model,
                                               src_tasks=src_tasks,
                                               tgt_prompt=tgt_prompt,
                                               norm=norm,
                                               whiten=whiten,
                                               projection=projection,
                                               device=device,
                                               metric=metric)

    # Now, score it
    kendall = kendalls_coefficient(results_dict, ground_truth_dict)
    spearmans = spearmans_rank_correlation(results_dict, ground_truth_dict)

    return kendall, spearmans, results_dict


def map_transferability_to_length(results_dict, map_func):
    length_dict = {}
    for k, v in results_dict.items():
        length_dict[k] = map_func(v)

    return length_dict


class UniversalProjector(nn.Module):

    def __init__(self,
                 device,
                 opt_dim,

                 # Source
                 use_src=True,
                 src_mix=True,
                 src_noise=0.,

                 # Target
                 use_tgt=False,
                 tgt_mix=True,  # Future function
                 tgt_noise=0.,  # Future function

                 dropout=0.1):

        super(UniversalProjector, self).__init__()

        self.device = device

        # Source parameters
        self.src_mix = src_mix
        self.src_noise = src_noise
        if use_src:
            self.src_proj = nn.Sequential(nn.Linear(768, opt_dim), nn.Dropout(dropout))
        else:
            self.src_proj = nn.Identity()


        # Target parameters
        if use_tgt:
            self.tgt_proj = nn.Sequential(nn.Linear(768, opt_dim), nn.Dropout(dropout))
        else:
            self.tgt_proj = nn.Identity()

    def forward(self, src_prompts, tgt_prompts):

        if self.training:
            # Generate source noise
            if self.src_noise > 0:
                noise_std = self.src_noise * src_prompts.std()
                noise = torch.normal(mean=0, std=noise_std, size=src_prompts.shape)
            else:
                noise = torch.zeros(src_prompts.shape)
            noise = noise.to(self.device)

            batch_size = src_prompts.shape[0]

            if self.src_mix:

                # 1. (13, 100, 768) to (1300, 768)
                src_prompts_mixed = src_prompts.view(src_prompts.shape[0] * src_prompts.shape[1], -1)

                # 2. randperm
                mix_ids = torch.randperm(src_prompts_mixed.shape[0])
                src_prompts_mixed = src_prompts_mixed[mix_ids]

                # 3. (1300, 768) to (13, 100, 768)
                src_prompts_mixed = src_prompts_mixed.view(src_prompts.shape[0], src_prompts.shape[1], -1)

                ids = torch.randperm(src_prompts.shape[0])[:batch_size]
                src_prompts = src_prompts_mixed[ids] + noise

            else:
                ids = torch.randperm(src_prompts.shape[0])[:batch_size]
                src_prompts = src_prompts[ids] + noise

        return self.src_proj(src_prompts), self.tgt_proj(tgt_prompts)


class SourcePromptDataset:

    def __init__(self, src_prompts, device, mix=False, noise=0.0):

        self.src_prompts = src_prompts
        self.mix = mix
        self.noise = noise
        self.device = device

        if self.noise > 0:
            self.noise_std = self.noise * self.src_prompts.std()

    def get_batch(self, batch_size=None):

        if self.noise > 0:
            noise = torch.normal(mean=0, std=self.noise_std, size=self.src_prompts.shape)
        else:
            noise = torch.zeros(self.src_prompts.shape)

        noise = noise.to(self.device)

        if batch_size is None:
            batch_size = self.src_prompts.shape[0]

        if self.mix:

            # 1. (13, 100, 768) to (1300, 768)
            src_prompts_mixed = self.src_prompts.view(self.src_prompts.shape[0] * self.src_prompts.shape[1], -1)

            # 2. randperm
            mix_ids = torch.randperm(src_prompts_mixed.shape[0])
            src_prompts_mixed = src_prompts_mixed[mix_ids]

            # 3. (1300, 768) to (13, 100, 768)
            src_prompts_mixed = src_prompts_mixed.view(self.src_prompts.shape[0], self.src_prompts.shape[1], -1)

            ids = torch.randperm(self.src_prompts.shape[0])[:batch_size]
            return src_prompts_mixed[ids] + noise

        else:
            ids = torch.randperm(self.src_prompts.shape[0])[:batch_size]
            return self.src_prompts[ids] + noise


if __name__ == '__main__':
    # Test the universal projector
    p = UniversalProjector(type='non-linear',
                           outer_dim=768,
                           inner_dim=384)

    src_prompts = torch.rand((13, 100, 768))

    src_pmt_dataset = SourcePromptDataset(src_prompts, mix=True, noise=0.1)

    x = src_pmt_dataset.get_batch()

    src_prompts_projected = p(x)

    print(src_prompts_projected.shape)

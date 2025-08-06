import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import math
from tqdm import tqdm
from functools import reduce
from operator import mul

import torch
from torch.utils.data import DataLoader

from train import train_utils
from analysis.metrics import MaximumMeanDiscrepancyLoss, Distance
from analysis.utils import *

class InfoWriter:

    def __init__(self, source_tasks):

        self.table = '{:>20s}'.format('Target | Source')

        self.csv = ''

        for source_task in source_tasks:
            self.table += '\t{:^10s}'.format(source_task)
            self.csv += ',{}'.format(source_task)

    def update_row(self, target_task, mmd_list):

        self.table += '\n{:>20s}'.format(target_task)
        self.csv += '\n{}'.format(target_task)

        for value in mmd_list:
            s = '{:.4f}'.format(value)

            self.table += '\t{:^10s}'.format(s)
            self.csv += ',{}'.format(value)

    def print(self):
        print(self.table)

    def save_csv(self, name):

        csv_path = 'analysis/ground_truth/{}.csv'.format(name)

        if os.path.exists(csv_path):
            os.remove(csv_path)

        with open(csv_path, 'wt') as c:
            c.write(self.csv)


def mmd_inference(patch_embeds_dataloader, src_task, loss_func, prompt):
    with torch.no_grad():

        with tqdm(total=5 * len(patch_embeds_dataloader)) as pbar:
            pbar.set_description(src_task.upper())

            step = 0
            loss = 0
            for epoch in range(5):

                for batch, x in enumerate(patch_embeds_dataloader):
                    x = x.to(device)

                    loss += loss_func(prompt, x)

                    step += 1

                    pbar.update(1)

    return loss / step


def main(device, target_tasks, reduction, src_model, metric='mmd', split='test', write_csv=False):
    assert metric in ('mmd', 'Euclidean', 'Cosine')
    assert split in ('train', 'test')

    # Create metric loss function
    if metric == 'mmd':
        loss_func = MaximumMeanDiscrepancyLoss()
    else:
        loss_func = Distance(device, metric)

    # To write the final information in a table form
    info = InfoWriter(list(task_dict.keys()))

    # Iterate through the target tasks
    for target_task in target_tasks:

        print('On {}:'.format(target_task.upper()))

        # Create dataset
        train_loader, _, test_loader, _ = train_utils.dataset_loading(dataset=target_task, batch_size=64)

        # Extract patch embeddings
        patch_embeds = extract_patch_embeds(device, eval('{}_loader'.format(split)), target_task)

        # Convert the patch embeddings into a torch dataset
        patch_embeds_dataset = EmbeddingDataset(patch_embeds, reduction=reduction)
        patch_embeds_dataloader = DataLoader(patch_embeds_dataset, batch_size=100, shuffle=True, drop_last=True)

        # Iterate through the source tasks
        loss_list = []
        for src_task in list(task_dict.keys()):

            # Load the source prompt
            if src_task == 'xavier':  # Random prompt initialized by xavier initialization
                patch_size = (16, 16)
                val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + 768))
                prompt = torch.FloatTensor(100, patch_embeds.shape[-1]).uniform_(-val, val).to(device)
            else:
                path = os.path.join('task_prompt_emb',
                                    task_dict[src_task.lower()] + 'Prompt' + model_dict[src_model.lower()],
                                    'task_prompt')
                prompt = torch.load(path).to(device)

            if metric == 'mmd':
                avg_loss = mmd_inference(patch_embeds_dataloader, src_task, loss_func, prompt)
            else:
                with torch.no_grad():
                    avg_loss = loss_func(prompt, patch_embeds, src_task)

            loss_list.append(avg_loss)

        info.update_row(target_task, loss_list)
        info.print()

    if write_csv:
        info.save_csv('{}_{}'.format(metric.upper(), split.upper()))


if __name__ == '__main__':
    print(os.getcwd())
    os.chdir('../')
    print(os.getcwd())

    device = train_utils.print_sys_info(0)
    # target_tasks = ['cifar', 'caltech101', 'dtd', 'oxford_flowers102', 'oxford_iiit_pet', 'sun397', 'svhn',
    #                 'patch_camelyon', 'resisc45', 'eurosat', 'diabetic_retinopathy',
    #                 'dmlab', 'kitti', 'smallnorb_azi', 'smallnorb_ele', 'dsprites_loc', 'dsprites_ori', 'clevr_dist', 'clevr_count']
    target_tasks = ['dtd']
    reduction = 'none'
    src_model = 'roberta'
    metric = 'mmd'  # 'mmd', 'Euclidean', 'Cosine'
    split = 'train'
    write_csv = False

    main(device, target_tasks, reduction, src_model, metric=metric, split=split, write_csv=write_csv)

'''
Q1: What is a universal projector?
A1: A linear projector that projects the source prompts to the space of the target prompt.
    All the source tasks share the same projector.

Q2: How to train a universal projector?
A2: The main idea is to minimize the Euclidean Distance or Cosine Similarity between the projected source prompts and
    target prompt.
    But when it comes to the details, there could be too many variations. Let's just try the easiest first and see how
    it goes.

Q3: How to evaluate a universal projector?
A3: This is simple, just use your transferability metrics.
'''
import os
import yaml
import random
import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from transferability.utils import *
from transferability.metrics.loss_functions import DistanceAverageEmbedding, DistanceInstanceEmbedding
from transferability.metrics.spot import cosine_average
from train.train_utils import print_sys_info, set_random_seed

def cosine_between_each_source_and_target(src_prompts, tgt_prompt, projector):

    cosines = []

    for src_prompt in src_prompts:

        src_prompt_proj, tgt_prompt_proj = projector(src_prompt, tgt_prompt)

        cosine = cosine_average(src_prompt_proj, tgt_prompt_proj)

        cosines.append(cosine)

    return torch.Tensor(cosines).mean()

def train():

    kendalls = []
    spearmans = []
    final_losses = []
    avg_cosines = []
    for seed in [42, 44, 100]:

        set_random_seed(seed, reproducibility=True)

        # Create the universal projector
        universal_projector = UniversalProjector(device=device,
                                                 opt_dim=config['Projector']['opt_dim'],
                                                 use_src=config['Projector']['use_src'],
                                                 src_mix=config['Projector']['src_mix'],
                                                 src_noise=config['Projector']['src_noise'],
                                                 use_tgt=config['Projector']['use_tgt'],
                                                 dropout=config['Projector']['dropout'])
        universal_projector.train()
        universal_projector.to(device)

        optimizer = optim.Adam(universal_projector.parameters(), lr=lr, weight_decay=wd)


        for epoch in range(100):

            # Forward pass
            src_prompts_proj, tgt_prompt_proj = universal_projector(src_prompts, tgt_prompt)

            loss = loss_fn(src_prompts_proj, tgt_prompt_proj)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        universal_projector.eval()
        kendall, spearman, _ = evaluate(src_model=config['Source']['model'],
                                        src_tasks=config['Source']['tasks'],
                                        tgt_prompt=tgt_prompt,
                                        norm=config['basic']['norm'],
                                        whiten=config['basic']['whiten'],
                                        ground_truth_dict=ground_truth_dict,
                                        projection=universal_projector,
                                        device=device,
                                        metric=config['Loss']['evaluation_metric'])
        final_loss = loss.detach().cpu().numpy()
        avg_cosine = cosine_between_each_source_and_target(src_prompts, tgt_prompt, universal_projector)

        kendalls.append(kendall*100)
        spearmans.append(spearman*100)
        final_losses.append(final_loss)
        avg_cosines.append(avg_cosine)

    kendalls = np.array(kendalls)
    spearmans = np.array(spearmans)
    final_losses = np.array(final_losses)
    avg_cosines = np.array(avg_cosines)

    mean_kendalls, std_kendalls = kendalls.mean(), kendalls.std()
    mean_spearmans, std_spearmans = spearmans.mean(), spearmans.std()
    mean_losses, std_losses = final_losses.mean(), final_losses.std()
    mean_avg_cosines, std_avg_cosines = avg_cosines.mean(), avg_cosines.std()

    kendalls_dict = {'best': '{:.2f}±{:.2f}'.format(mean_kendalls, std_kendalls)}
    spearmans_dict = {'best': '{:.2f}±{:.2f}'.format(mean_spearmans, std_spearmans)}
    losses_dict = {'best': '{:.2f}±{:.2f}'.format(mean_losses, std_losses)}
    avg_cosines_dict = {'best': '{:.2f}±{:.2f}'.format(mean_avg_cosines, std_avg_cosines)}

    return kendalls_dict, spearmans_dict, losses_dict, avg_cosines_dict

class Info:

    def __init__(self, wd_grid):

        self.kendall = '<{:-^100s}>\n'.format("Kendall: mean(best)±std(best)")
        self.spearman = '<{:-^100s}>\n'.format("Spearman: mean(best)±std(best)")
        self.loss = '<{:-^100s}>\n'.format("Loss: mean(best)±std(best)")
        self.avg_cosines = '<{:-^100s}>\n'.format("Average Cosine: mean(best)±std(best)")

        self.kendall += '{:^9s}'.format('lr | wd')
        self.spearman += '{:^9s}'.format('lr | wd')
        self.loss += '{:^9s}'.format('lr | wd')
        self.avg_cosines += '{:^9s}'.format('lr | wd')

        self.__add_wd_info(wd_grid)

    def __add_wd_info(self, wd_grid):

        for wd in wd_grid:
            self.kendall += '\t{:^12s}'.format(str(wd))
            self.spearman += '\t{:^12s}'.format(str(wd))
            self.loss += '\t{:^12s}'.format(str(wd))
            self.avg_cosines += '\t{:^12s}'.format(str(wd))

    def add_lr_info(self, lr):

        self.kendall += '\n{:^9s}'.format(str(lr))
        self.spearman += '\n{:^9s}'.format(str(lr))
        self.loss += '\n{:^9s}'.format(str(lr))
        self.avg_cosines += '\n{:^9s}'.format(str(lr))

    def add_score_infor(self, kendalls_dict, spearmans_dict, losses_dict, avg_cosines_dict):

        self.kendall += '\t{:^12s}'.format(kendalls_dict['best'])
        self.spearman += '\t{:^12s}'.format(spearmans_dict['best'])
        self.loss += '\t{:^12s}'.format(losses_dict['best'])
        self.avg_cosines += '\t{:^12s}'.format(avg_cosines_dict['best'])

    def display_info(self):

        print(self.kendall, '\n')
        print(self.spearman, '\n')
        print(self.loss, '\n')
        print(self.avg_cosines, '\n')

if __name__ == '__main__':

    print(os.getcwd())
    os.chdir('../')
    print(os.getcwd())

    # Parameters
    device = print_sys_info(0)

    # Load the config yaml
    config = yaml.load(open('transferability/configs/UniversalProjector-GridSearch.yaml'), yaml.FullLoader)

    # Define the loss function for training the projector
    loss_fn = eval(config['Loss']['function'])(metric=config['Loss']['metric'], reduction=config['Loss']['reduction'])

    # Get the ground-truth transferability results
    ground_truth_dict, _ = get_ground_truth(config['basic']['transfer_task'], config['Source']['model'], config['Target']['task'])

    # Load all the source prompts (13, 100, 768)
    src_prompts = load_source_prompts_multi(config['Source']['tasks'],
                                            norm=config['basic']['norm'],
                                            whiten=config['basic']['whiten'],
                                            src_model=config['Source']['model'],
                                            device=device)

    # Load the target prompt (100, 768)
    tgt_prompt = load_target_prompt(config['Target']['task'],
                                    norm=config['basic']['norm'],
                                    whiten=config['basic']['whiten'],
                                    device=device)

    # Print info
    print_info = Info(config['basic']['wd_grid'])

    # Start grid search
    print('<{:-^100s}>'.format('Start Grid Search'))

    with tqdm(total=len(config['basic']['lr_grid']) * len(config['basic']['wd_grid'])) as pbar:
        pbar.set_description('Progress')

        for lr in config['basic']['lr_grid']:

            print_info.add_lr_info(lr)

            for wd in config['basic']['wd_grid']:

                kendalls_dict, spearmans_dict, losses_dict, avg_cosines_dict = train()

                print_info.add_score_infor(kendalls_dict, spearmans_dict, losses_dict, avg_cosines_dict)

                pbar.update(1)

    print_info.display_info()
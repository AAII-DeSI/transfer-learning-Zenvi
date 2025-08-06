import os
import yaml

import torch

from train import train_utils
from model.model_ViT import MyViT as model


class Log:

    def __init__(self, src_projs, dataset):

        self.logdir = 'Logs/Grid_Search/AttentionTransfer/{}.txt'.format(dataset)
        if not os.path.exists('Logs/Grid_Search/AttentionTransfer/'):
            os.makedirs('Logs/Grid_Search/AttentionTransfer/')

        self.info = '{:>6s}'.format('Source')

        for src_proj in src_projs:
            self.info += '\t{:^13s}'.format(src_proj)

    def update_row_head(self, src_task):

        self.info += '\n{:>6s}'.format(src_task)

    def update_row_content(self, best_tests):

        s = '{:.2f}±{:.2f}'.format(torch.mean(best_tests), torch.std(best_tests))
        self.info += '\t{:^13s}'.format(s)

    def write(self):

        with open(self.logdir, 'at') as file:
            file.write(self.info)


search_map = {
    # 'caltech101': ('mnli', 'snli'),
    # 'cifar': ('mnli', 'snli', 'qqp'),
    'dtd': ('mnli', 'snli'),
    # 'oxford_flowers102': ('mnli', 'snli'),
    # 'oxford_iiit_pet': ('mnli', 'snli'),
    # 'sun397': ('snli', ),
    # 'svhn': ('mnli', 'snli', 'qqp'),
    # 'patch_camelyon': ('mnli', 'snli', 'qqp'),
    # 'resisc45': ('mnli', 'snli',),
    # 'eurosat': ('mnli', 'qnli', 'snli'),
    # 'diabetic_retinopathy': ('qnli', 'qqp', 'mnli', 'snli'),
    # 'dmlab': ('qnli', 'qqp', 'mnli', 'snli'),
    # 'kitti': ('qnli', 'qqp', 'mnli', 'snli'),
    # 'kitti': ('qnli', )
    # 'smallnorb_azi': ('qnli', 'qqp', 'mnli', 'snli', 'sst2'),
    # 'smallnorb_ele': ('qnli', 'qqp', 'mnli', 'snli'),
    # 'dsprites_loc': ('mnli', 'snli', 'qqp', 'imdb'),
    # 'dsprites_ori': ('qnli', 'qqp', 'mnli', 'snli', 'tweet'),
    # 'clevr_dist': ('qnli', 'qqp', 'mnli', 'snli'),
    # 'clevr_count': ('qnli', 'qqp', 'mnli', 'snli'),
}

if __name__ == '__main__':

    # --------------------------------------- Parameters Need Your Modification ----------------------------------------
    device = train_utils.print_sys_info(0)
    # ------------------------------------------------------------------------------------------------------------------

    config_file = 'configs/grid_search/AttentionTransfer/all.yaml'
    config = yaml.load(open(config_file), yaml.FullLoader)

    for dataset in search_map.keys():

        print('Performing grid search on {}'.format(dataset))

        # Create target dataset
        train_loader, val_loader, _, num_classes = train_utils.dataset_loading(dataset=dataset, batch_size=64,
                                                                               train_mode='800')

        # Create log for this target dataset
        log = Log(config['model']['src_projs'], dataset)

        src_tasks = search_map[dataset]

        for src_task in src_tasks:

            log.update_row_head(src_task)

            for src_proj in config['model']['src_projs']:

                # Deal with the learning rate and target prompt length
                lr = config['train']['base_lr'] * 64 / 256 if config['train']['modify_lr'] else config['train'][
                    'base_lr']
                tgt_len = 100 - eval(src_proj.split('-')[1]) if config['model']['tgt_len'] == 'c_source' else \
                config['model']['tgt_len']

                best_tests = []
                for seed in (42, 44, 100):
                    train_utils.set_random_seed(seed, reproducibility=True)
                    print('<{:-^100s}>'.format('Seed and Grid Information'),
                          'Current Seed: {}'.format(seed),
                          'Current Source Task: {}'.format(src_task),
                          'Current Source Projection: {}'.format(src_proj),
                          sep='\n')

                    # Build the model
                    vit = model(device=device,
                                trainable_para=config['model']['trainable_para'],
                                output_dim=num_classes,

                                src_len=config['model']['src_len'],
                                src_init=src_task,
                                src_model=config['model']['src_model'],
                                src_proj=src_proj,

                                tgt_len=tgt_len,
                                tgt_init=config['model']['tgt_init'],
                                tgt_proj=config['model']['tgt_proj'],
                                tgt_involvement=config['model']['tgt_involvement'],
                                tgt_task=dataset,

                                cls_pos=config['model']['cls_pos'])

                    # Build the optimizer
                    optimizer = train_utils.make_optimizer(model=vit,
                                                           conf_weight_decay=config['train']['weight_decay'],
                                                           conf_base_lr=lr,
                                                           conf_optimizer=config['train']['optimizer_type'])

                    # Start Training
                    _, _, best_test, _ = train_utils.train(train_loader=train_loader,
                                                           test_loader=val_loader,
                                                           model=vit,
                                                           lr_decay=config['train']['lr_decay'],
                                                           optimizer=optimizer,
                                                           tensorboard_path=None,
                                                           epochs=config['train']['epochs'],
                                                           device=device,
                                                           save_prompt=config['train']['save_prompt'],
                                                           seed=seed)

                    # Print critical information for this seed
                    print('Finished training for seed={}; Summary:'.format(seed))
                    print('Best Test: {:.2f}'.format(best_test * 100))

                    # Add information to the lists
                    best_tests.append(best_test * 100)

                # Print information in (mean ± std) format
                best_tests = torch.Tensor(best_tests)

                print('Finished training for source {} ({})! Final statistics: '.format(src_task, src_proj))
                print('Best Test: {:.2f}±{:.2f}'.format(torch.mean(best_tests), torch.std(best_tests)))

                # Write information to log
                log.update_row_content(best_tests)
                print(log.info)
        log.write()

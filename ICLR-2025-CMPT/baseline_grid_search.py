from train import train_utils as utils
from model.model_ViT import MyViT as model

import os
import yaml
import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn

def check_and_print_config(config):

    assert type(config['dataset']['name']) == list and len(config['dataset']['name']) >= 1

    if type(config['dataset']['batch_size']) == int:
        config['dataset']['batch_size'] = [config['dataset']['batch_size']] * len(config['dataset']['name'])
    elif type(config['dataset']['batch_size']) == list:
        assert len(config['dataset']['batch_size']) == len(config['dataset']['name'])
    else:
        raise NotImplementedError

    if type(config['model']['src_len']) == int:
        config['model']['src_len'] = [config['model']['src_len']] * len(config['dataset']['name'])
    elif type(config['model']['src_len']) == list:
        assert len(config['model']['src_len']) == len(config['dataset']['name'])
    else:
        raise NotImplementedError

    print('<{:-^100s}>'.format('DATASET PARAMETERS'))
    for k, v in config['dataset'].items():
        print('{}: {}'.format(k, v))

    print('<{:-^100s}>'.format('EXPERIMENT PARAMETERS'))
    for k, v in config['train'].items():
        print('{}: {}'.format(k, v))

    print('<{:-^100s}>'.format('MODEL PARAMETERS'))
    for k, v in config['model'].items():
        print('{}: {}'.format(k, v))

def set_random_seed(reproducibility):

    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    if reproducibility:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True

class Log:

    def __init__(self, logdir):

        self.logdir = logdir

    def write(self, s):

        with open(self.logdir, 'at') as file:
            file.write(s)



if __name__ == '__main__':

    # --------------------------------------- Parameters Need Your Modification ----------------------------------------
    device = utils.print_sys_info(0)
    config_file = 'configs/grid_search/VPT_100/all.yaml'
    # ------------------------------------------------------------------------------------------------------------------

    config = yaml.load(open(config_file), yaml.FullLoader)
    check_and_print_config(config)

    for idx in range(len(config['dataset']['name'])):

        dataset = config['dataset']['name'][idx]
        batch_size = config['dataset']['batch_size'][idx]

        train_loader, val_loader, _, num_classes = utils.dataset_loading(dataset=dataset, batch_size=batch_size, train_mode='800')

        # --------------------------------------------- Log Parameters -------------------------------------------------
        logdir = 'Logs/Grid_Search/{}/{}-sup.txt'.format(config['log']['dir_lvl_2'], dataset)
        if not os.path.exists('Logs/Grid_Search/{}/'.format(config['log']['dir_lvl_2'])):
            os.makedirs('Logs/Grid_Search/{}/'.format(config['log']['dir_lvl_2']))
        print('Log will be saved to {}...'.format(logdir))
        log = Log(logdir)

        # write the firs line of the log
        log.write('{:^12s}'.format('wd | base_lr'))
        for base_lr in config['train']['base_lr_grid']:
            log.write('{:^15s}'.format(str(base_lr)))

        # ------------------------------------------------ Training ----------------------------------------------------
        for wd in config['train']['weight_decay_grid']:
            log.write('\n{:^12s}'.format(str(wd)))

            for base_lr in config['train']['base_lr_grid']:

                lr = base_lr * batch_size / 256 if config['train']['modify_lr'] else base_lr

                best_tests = []

                for seed in (42, 44, 100):

                    # Set the seed
                    set_random_seed(reproducibility=True)
                    print('<{:-^100s}>'.format('Seed and Hyperparameter Information'),
                          'Current Seed: {}'.format(seed),
                          'Current Learning Rate: {}'.format(lr),
                          'Current Weight Decay: {}'.format(wd),
                          sep='\n')

                    vit = model(device=device,
                                trainable_para=config['model']['trainable_para'],
                                output_dim=num_classes,

                                src_len=config['model']['src_len'][idx],
                                src_init=config['model']['src_task'],
                                src_model=config['model']['src_model'],
                                src_proj=config['model']['src_proj'],

                                tgt_len=config['model']['tgt_len'],
                                tgt_init=config['model']['tgt_init'],
                                tgt_proj=config['model']['tgt_proj'],
                                tgt_involvement=config['model']['tgt_involvement'],
                                tgt_task=config['dataset']['name'],

                                cls_pos=config['model']['cls_pos'])

                    optimizer = utils.make_optimizer(model=vit,
                                                     conf_weight_decay=wd,
                                                     conf_base_lr=lr,
                                                     conf_optimizer=config['train']['optimizer_type'])

                    # Start Training
                    _, _, best_test, _ = utils.train(train_loader=train_loader,
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

                print('Finished training for seed 42, 44, 100! Final statistics: ')
                print('Best Test: {:.2f}±{:.2f}'.format(torch.mean(best_tests), torch.std(best_tests)))

                # Write information to log
                best_tests_s = '{:.2f}±{:.2f}'.format(torch.mean(best_tests), torch.std(best_tests))
                log.write('{:^15s}'.format(best_tests_s))

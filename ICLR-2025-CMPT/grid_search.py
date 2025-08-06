import os
import yaml
import torch

from train import train_utils
from model.model_ViT import MyViT as model


class Log:

    def __init__(self, src_task, dataset, optimizer_type, proj, dir_lvl_2, base_lr_grid):

        # file_name = '{}-to-{}_{}_{}'.format(src_task, dataset, optimizer_type, proj)
        file_name = '{}-{}{}-{}'.format(dataset.upper(), src_task.upper(), proj.split('-')[1], optimizer_type)
        logdir = 'Logs/Grid_Search/{}/{}.txt'.format(dir_lvl_2, file_name)
        if not os.path.exists('Logs/Grid_Search/{}/'.format(dir_lvl_2)):
            os.makedirs('Logs/Grid_Search/{}/'.format(dir_lvl_2))
        print('Log will be saved to {}...'.format(logdir))

        self.logdir = logdir

        self.write_first_row(base_lr_grid)

    def _write(self, s):
        with open(self.logdir, 'at') as file:
            file.write(s)

    def write_first_row(self, base_lr_grid):
        self._write('{:^12s}'.format('wd | base_lr'))
        for base_lr in base_lr_grid:
            self._write('\t{:^11s}'.format(str(base_lr)))

    def write_row_head(self, wd):

        self._write('\n{:^12s}'.format(str(wd)))

    def write_row_content(self, best_tests):

        s = '{:.2f}±{:.2f}'.format(torch.mean(best_tests), torch.std(best_tests))
        self._write('\t{:^11s}'.format(s))


if __name__ == '__main__':

    # --------------------------------------- Parameters Need Your Modification ----------------------------------------
    device = train_utils.print_sys_info(1)
    task = 'AttentionTransfer'  # AttentionTransfer, ProjectionTransfer, ArtificialPrompt
    source_model = 'roberta'
    target_task = 'oxford_iiit_pet'
    optimizer_type = 'adam'
    # ------------------------------------------------------------------------------------------------------------------

    config_file = 'configs/grid_search/{}/{}_{}_{}.yaml'.format(task, source_model, target_task, optimizer_type)
    config = yaml.load(open(config_file), yaml.FullLoader)

    train_utils.print_config(config)

    # Create dataset
    train_loader, val_loader, _, num_classes = train_utils.dataset_loading(dataset=config['dataset']['name'],
                                                                           batch_size=config['dataset']['batch_size'],
                                                                           train_mode='800')

    # Log Parameters
    log = Log(src_task=config['model']['src_task'],
              dataset=config['dataset']['name'],
              optimizer_type=config['train']['optimizer_type'],
              proj=config['model']['src_proj'],
              dir_lvl_2=config['log']['dir_lvl_2'],
              base_lr_grid=config['train']['base_lr_grid'])

    for wd in config['train']['weight_decay_grid']:
        log.write_row_head(wd)

        for base_lr in config['train']['base_lr_grid']:

            lr = base_lr * config['dataset']['batch_size'] / 256 if config['train']['modify_lr'] else base_lr

            best_tests = []
            for seed in (42, 44, 100):

                # Set the seed
                train_utils.set_random_seed(seed, reproducibility=True)
                print('<{:-^100s}>'.format('Seed and Hyperparameter Information'),
                      'Current Seed: {}'.format(seed),
                      'Current Learning Rate: {}'.format(lr),
                      'Current Weight Decay: {}'.format(wd),
                      sep='\n')

                vit = model(device=device,
                            trainable_para=config['model']['trainable_para'],
                            output_dim=num_classes,

                            src_len=config['model']['src_len'],
                            src_init=config['model']['src_task'],
                            src_model=config['model']['src_model'],
                            src_proj=config['model']['src_proj'],

                            tgt_len=config['model']['tgt_len'],
                            tgt_init=config['model']['tgt_init'],
                            tgt_proj=config['model']['tgt_proj'],
                            tgt_involvement=config['model']['tgt_involvement'],
                            tgt_task=config['dataset']['name'],

                            cls_pos=config['model']['cls_pos'])

                optimizer = train_utils.make_optimizer(model=vit,
                                                       conf_weight_decay=wd,
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
                                                       seed=seed,
                                                       evaluate_last=config['train']['evaluate_last'])

                # Print critical information for this seed
                print('Finished training for seed={}; Summary:'.format(seed))
                print('Best Test: {:.2f}'.format(best_test * 100))

                # Add information to the lists
                best_tests.append(best_test * 100)

            # Print information in (mean ± std) format
            best_tests = torch.Tensor(best_tests)

            print('Finished training for lr={}, wd={}! Final statistics: '.format(lr, wd))
            print('Best Test: {:.2f}±{:.2f}'.format(torch.mean(best_tests), torch.std(best_tests)))

            # Write information to log
            log.write_row_content(best_tests)

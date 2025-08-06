import os
import yaml
import torch

from train import train_utils
from model.model_ViT import MyViT as model


class InfoPrinter:

    def __init__(self, src_projs):
        self.info = '{:>10s}'.format('Source')

        for src_proj in src_projs:
            self.info += '\t{:^13s}'.format(src_proj)

    def update_row_head(self, src_task):
        self.info += '\n{:>10s}'.format(src_task)

    def update_row_content(self, best_tests):
        s = '{:.2f}±{:.2f}'.format(torch.mean(best_tests), torch.std(best_tests))
        self.info += '\t{:^13s}'.format(s)

    def print(self):
        print(self.info)


class Log:

    def __init__(self, dir_lvl_1, src_model, dataset, src_task, src_proj):
        dir_lvl_2 = '{}-{}'.format(src_model.upper(), dataset.upper())
        file_name = '{}-{}'.format(src_task, src_proj)
        self.logdir = 'Logs/{}/{}/{}.txt'.format(dir_lvl_1, dir_lvl_2, file_name)

        if not os.path.exists('Logs/{}/{}'.format(dir_lvl_1, dir_lvl_2)):
            os.makedirs('Logs/{}/{}'.format(dir_lvl_1, dir_lvl_2))

        print('Log will be saved to {}...'.format(self.logdir))

    def write_one_seed(self, best_train, last_train, best_test, last_test):
        with open(self.logdir, 'at') as file:
            file.write('Seed {}: \n'.format(seed))
            file.write('Best Train: {:.2f} \n'.format(best_train * 100))
            file.write('Last Train: {:.2f} \n'.format(last_train * 100))
            file.write('Best  Test: {:.2f} \n'.format(best_test * 100))
            file.write('Last  Test: {:.2f} \n'.format(last_test * 100))

    def write_all_seeds(self, best_trains, last_trains, best_tests, last_tests):
        with open(self.logdir, 'at') as file:
            file.write('Overall: \n')
            file.write('Best Train: {:.2f}±{:.2f} \n'.format(torch.mean(best_trains), torch.std(best_trains)))
            file.write('Last Train: {:.2f}±{:.2f} \n'.format(torch.mean(last_trains), torch.std(last_trains)))
            file.write('Best  Test: {:.2f}±{:.2f} \n'.format(torch.mean(best_tests), torch.std(best_tests)))
            file.write('Last  Test: {:.2f}±{:.2f} \n'.format(torch.mean(last_tests), torch.std(last_tests)))


def tensorboard_path_generator(use_tensorboard, dir_lvl_1, src_model, dataset, src_task, src_proj, seed):
    dir_lvl_2 = '{}-{}'.format(src_model.upper(), dataset.upper())
    file_name = '{}-{}'.format(src_task, src_proj)

    if use_tensorboard:
        tensorboard_path = 'TensorBoards/Seed={}/{}/{}/{}'.format(seed, dir_lvl_1, dir_lvl_2, file_name)
        print('Using Tensorboard Path: {}'.format(tensorboard_path))
        if os.path.exists(tensorboard_path):
            print('WARNING!!! TENSORBOARD PATH: {} ALREADY EXISTS!'.format(tensorboard_path))
    else:
        tensorboard_path = None
        print('Tensorboard will not be used.')

    return tensorboard_path


if __name__ == '__main__':

    # --------------------------------------- Parameters Need Your Modification ----------------------------------------
    device = train_utils.print_sys_info(0)
    task = 'AttentionTransfer'  # AttentionTransfer, FrozenPromptTransfer, ProjectionTransfer
    source_model = 'roberta'
    target_task = 'dtd'
    evaluate_last = False
    # ------------------------------------------------------------------------------------------------------------------

    config_file = 'configs/main/{}/{}_{}.yaml'.format(task, source_model, target_task)
    config = yaml.load(open(config_file), yaml.FullLoader)
    train_utils.print_config(config)

    # Create dataset
    train_loader, _, test_loader, num_classes = train_utils.dataset_loading(dataset=config['dataset']['name'],
                                                                            batch_size=config['dataset']['batch_size'])

    # This is used to print the final performance table
    info_printer = InfoPrinter(config['model']['src_projs'])

    for src_task in config['model']['src_tasks']:
        info_printer.update_row_head(src_task)

        for src_proj in config['model']['src_projs']:

            # Deal with the target prompt length. Note that target prompt will only appear in attention transfer
            if 'attention-' in src_proj:
                tgt_len = 100 - eval(src_proj.split('-')[1]) if config['model']['tgt_len'] == 'c_source' else \
                    config['model']['tgt_len']
            else:
                tgt_len = config['model']['tgt_len']

            # Log parameters
            log = Log(dir_lvl_1=config['log']['dir_lvl_1'],
                      src_model=config['model']['src_model'],
                      dataset=config['dataset']['name'],
                      src_task=src_task,
                      src_proj=src_proj)

            # Skip the run if the log file exists
            if os.path.exists(log.logdir):
                print('Log {} already occupied, skipping the run for {}-{}'.format(log.logdir, src_task, src_proj))
                info_printer.update_row_content(torch.Tensor([0, 0, 0]))
                continue

            best_trains, last_trains, best_tests, last_tests = [], [], [], []
            for seed in (42, 44, 100):
                # Set the seed
                train_utils.set_random_seed(seed, reproducibility=True)
                print('<{:-^100s}>'.format('SEED AND TENSORBOARD INFORMATION'))
                print('Current Seed: {}'.format(seed))

                # Generate tensorboard path
                tensorboard_path = tensorboard_path_generator(use_tensorboard=config['log']['use_tensorboard'],
                                                              dir_lvl_1=config['log']['dir_lvl_1'],
                                                              src_model=config['model']['src_model'],
                                                              dataset=config['dataset']['name'],
                                                              src_task=src_task,
                                                              src_proj=src_proj,
                                                              seed=seed)

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
                            tgt_task=config['dataset']['name'],

                            cls_pos=config['model']['cls_pos'])

                # Deal with the learning rate
                base_lr = config['train']['base_lr']
                lr = base_lr * config['dataset']['batch_size'] / 256 if config['train']['modify_lr'] else base_lr

                # Build the optimizer
                optimizer = train_utils.make_optimizer(model=vit,
                                                       conf_weight_decay=config['train']['weight_decay'],
                                                       conf_base_lr=lr,
                                                       conf_optimizer=config['train']['optimizer_type'])

                # Start Training
                best_train, last_train, best_test, last_test = train_utils.train(train_loader=train_loader,
                                                                                 test_loader=test_loader,
                                                                                 model=vit,
                                                                                 lr_decay=config['train']['lr_decay'],
                                                                                 optimizer=optimizer,
                                                                                 tensorboard_path=tensorboard_path,
                                                                                 epochs=config['train']['epochs'],
                                                                                 device=device,
                                                                                 save_prompt=config['train']['save_prompt'],
                                                                                 seed=seed,
                                                                                 evaluate_last=evaluate_last)

                # Print critical information for this seed
                print('Finished training for seed={}; Summary:'.format(seed),
                      'Best Train: {:.2f}'.format(best_train * 100),
                      'Last Train: {:.2f}'.format(last_train * 100),
                      'Best Test: {:.2f}'.format(best_test * 100),
                      'Last Test: {:.2f}'.format(last_test * 100),
                      'tensorboard path: {}'.format(tensorboard_path),
                      sep='\n')

                # Write information to log
                log.write_one_seed(best_train, last_train, best_test, last_test)

                # Add information to the lists
                best_trains.append(best_train * 100)
                last_trains.append(last_train * 100)
                best_tests.append(best_test * 100)
                last_tests.append(last_test * 100)

            # When all seeds are finished, convert the lists to torch tensors
            best_trains = torch.Tensor(best_trains)
            last_trains = torch.Tensor(last_trains)
            best_tests = torch.Tensor(best_tests)
            last_tests = torch.Tensor(last_tests)

            # Write information to log
            log.write_all_seeds(best_trains, last_trains, best_tests, last_tests)

            # Add info to the overall print info
            info_printer.update_row_content(best_tests)

            # Print the final information
            info_printer.print()

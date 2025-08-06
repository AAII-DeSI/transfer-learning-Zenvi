import sys
import os
import math
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from train import dataset_utils

import random
import numpy as np


def print_config(config):
    print('<{:-^100s}>'.format('DATASET PARAMETERS'))
    for k, v in config['dataset'].items():
        print('{}: {}'.format(k, v))

    print('<{:-^100s}>'.format('EXPERIMENT PARAMETERS'))
    for k, v in config['train'].items():
        print('{}: {}'.format(k, v))

    print('<{:-^100s}>'.format('MODEL PARAMETERS'))
    for k, v in config['model'].items():
        print('{}: {}'.format(k, v))


def set_random_seed(seed, reproducibility):
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


def print_sys_info(cuda_index):
    '''
    Print environmental information and set cuda device.
    '''

    print('<{:-^100s}>'.format('ENVIRONMENT INFORMATION'))
    # Check out the GPU and torch version status.

    print("Torch Version: ", torch.__version__)
    print("Cuda is available? ", torch.cuda.is_available())

    if torch.cuda.is_available():

        print("Cuda Version: ", torch.version.cuda)

        device = 'cuda'
        if cuda_index >= 0:
            torch.cuda.set_device(cuda_index)

        print("How many Cuda devices?", torch.cuda.device_count())
        print("Current Cuda device?", torch.cuda.current_device())

    else:
        device = 'cpu'

    print("Python Version: ", sys.version)
    return device


def dataset_loading(dataset: str = 'cifar',
                    batch_size: int = 32,
                    image_size: tuple = (224, 224),
                    normalize: str = 'imagenet',
                    train_mode: str = '1000',
                    shuffle: bool = True):
    num_classes_dict = {'cifar': 100,
                        'caltech101': 102,
                        'dtd': 47,
                        'dtd-jpg': 47,
                        'oxford_flowers102': 102,
                        'oxford_iiit_pet': 37,
                        'oxford_iiit_pet-jpg': 37,
                        'sun397': 397,
                        'svhn': 10,
                        'svhn-jpg': 10,
                        'patch_camelyon': 2,
                        'resisc45': 45,
                        'eurosat': 10,
                        'diabetic_retinopathy': 5,
                        'dmlab': 6,
                        'kitti': 4,
                        'smallnorb_azi': 18,
                        'smallnorb_ele': 9,
                        'smallnorb_ele-jpg': 9,
                        'dsprites_loc': 16,
                        'dsprites_ori': 16,
                        'clevr_dist': 6,
                        'clevr_count': 8,
                        'clevr_count-jpg': 8}

    dataset_root = 'data/vtab-1k/{}'.format(dataset)

    # Sanity check
    assert dataset in num_classes_dict.keys()
    assert train_mode in ('1000', '800')
    assert os.path.exists(dataset_root)

    # Statistics dictionaries of the four datasets
    means = {'imagenet': (0.485, 0.456, 0.406)}
    stds = {'imagenet': (0.229, 0.224, 0.225)}

    # Statistics retrieval
    mean = means[normalize]
    std = stds[normalize]

    # Image transforms
    image_transform = transforms.Compose([transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean, std)])

    # Datasets
    train_set = dataset_utils.VtabDataset(root=dataset_root, split=train_mode, transform=image_transform)
    val_set = dataset_utils.VtabDataset(root=dataset_root, split='200', transform=image_transform)
    test_set = dataset_utils.VtabDataset(root=dataset_root, split='test', transform=image_transform)

    # Data Loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, num_workers=2)

    return train_loader, val_loader, test_loader, num_classes_dict[dataset]


def make_optimizer(model, conf_weight_decay, conf_base_lr, conf_optimizer, conf_momentum=0.9):
    # Record all the trainable parameters
    params = []
    for key, value in model.named_parameters():
        if value.requires_grad:
            params.append((key, value))

    _params = []
    for key, value in params:

        lr = conf_base_lr
        weight_decay = conf_weight_decay

        # if 'opt_layer.0.bias' in key:
        #     weight_decay = 0

        if 'bias' in key:
            weight_decay = 0

        _params += [{"params": [value],
                     "lr": lr,
                     "weight_decay": weight_decay,
                     "name": key}]

    if conf_optimizer == 'adam':
        optimizer = optim.Adam(_params, lr=conf_base_lr, weight_decay=conf_weight_decay)
    else:
        optimizer = optim.SGD(_params, lr=conf_base_lr, weight_decay=conf_weight_decay, momentum=conf_momentum)

    # Print critical information for the optimizer
    print('<{:-^100s}>'.format('Optimizer Information'),
          'Type: {}'.format(conf_optimizer),
          sep='\n')
    for i, param_group in enumerate(optimizer.param_groups):
        print("{}: lr={}, wd={}".format(param_group['name'], param_group['lr'], param_group['weight_decay']))

    return optimizer


def train(train_loader,
          test_loader,
          model,
          lr_decay,
          optimizer,
          tensorboard_path=None,
          epochs=100,
          device='cuda',
          save_prompt=False,
          seed=42,
          evaluate_last=False,
          print_detail=False):
    # Loss
    loss_fn = nn.CrossEntropyLoss()

    # Switch all the modules to train mode, and load them to training devices
    model.to(device=device)
    model.train()

    # Tensorboard writer
    if tensorboard_path is not None:
        writer = SummaryWriter(tensorboard_path)
    else:
        writer = None
    print('<{:-^100s}>'.format('START TRAINING'))

    # To record the best training and testing accuracy
    best_train_acc, best_test_acc = 0, 0
    train_acc, test_acc = 0, 0

    # If we use a decay rule for the learning rate (Modify this if you need a new decay rule)
    if lr_decay:
        lr_scheduler = WarmupCosineSchedule(optimizer, warmup_steps=10, t_total=epochs)
    else:
        lr_scheduler = None

    # Start training
    for epoch in range(1, epochs + 1):

        # For display
        total_images = len(train_loader.dataset)
        trained_images = 0
        running_loss = torch.scalar_tensor(0)
        correct, total = 0, 0

        print('Epoch {}'.format(epoch))
        if lr_decay:
            if print_detail:
                for i, param_group in enumerate(optimizer.param_groups):
                    print("Learning rate for {}: {}".format(param_group['name'], param_group['lr']))
            else:
                print('Learning Rate: {}'.format(lr_scheduler.get_last_lr()[0]))


        # Progress bar
        with tqdm(total=len(train_loader)) as pbar:
            pbar.set_description('Training epoch {}'.format(epoch))

            for batch, (x, y) in enumerate(train_loader, 0):

                # Inputs and labels
                x, y = x.to(device), y.to(device)

                # Forward Pass
                logits = model(x)

                # Output-Level Loss
                loss = loss_fn(logits, y)

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Training statistics to be printed
                running_loss += loss.detach().cpu().item() * y.size(0)

                # For the current training batch, how many samples were correctly predicted
                _, predicted = logits.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
                trained_images += len(x)

                # Write the losses to Tensorboard
                if tensorboard_path is not None:
                    writer.add_scalar('Output Loss', loss.detach().cpu().item(),
                                      (epoch - 1) * total_images + trained_images)

                # Update progress bar
                pbar.update(1)

        # Update learning rate
        if lr_decay:
            lr_scheduler.step()

        if evaluate_last:
            if epoch == epochs:
                # Get the performance on the test set
                test_acc = test(test_loader=test_loader, model=model, device=device)
        else:
            test_acc = test(test_loader=test_loader, model=model, device=device)

        # Save the prompt
        if save_prompt:
            model.save_proj_prompt(root='{}_prompt'.format(save_prompt),
                                   accuracy="{:.2f}".format(test_acc * 100),
                                   seed=seed)

        train_acc = correct / total

        print("Finished training for {}/{} epochs,".format(epoch, epochs),
              "average loss: {:.2f},".format(running_loss / total_images),
              "current train accuracy: {:.2f},".format(train_acc * 100),
              "test accuracy {:.2f},".format(test_acc * 100))
        time.sleep(0.5)

        if train_acc >= best_train_acc:
            best_train_acc = train_acc
        if test_acc >= best_test_acc:
            best_test_acc = test_acc

        # Write the accuracies to Tensorboard
        if tensorboard_path is not None:
            writer.add_scalar('Training Acc', train_acc, (epoch - 1))
            writer.add_scalar('Testing Acc', test_acc, (epoch - 1))

        if epoch % 100 == 0:
            print('{}E: best train: {:.2f}; last train: {:.2f}'.format(epoch, best_train_acc * 100, train_acc * 100))
            print('{}E: best test: {:.2f}; last test: {:.2f}'.format(epoch, best_test_acc * 100, test_acc * 100))
            time.sleep(0.5)

    return best_train_acc, train_acc, best_test_acc, test_acc


def test(test_loader, model, device):
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        # Progress bar
        with tqdm(total=len(test_loader)) as pbar:
            pbar.set_description('Testing')

            for batch, (x, y) in enumerate(test_loader):
                # Get inputs
                x, y = x.to(device), y.to(device)

                # Inference
                logits = model(x)

                # Accurcay
                _, predicted = logits.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

                # Update progress bar
                pbar.update(1)

    model.train()

    return correct / total


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps`.
        Decreases learning rate from 1. to 0. over remaining
            `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate
            follows cosine function after warmup.
    """

    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step + 1) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


if __name__ == '__main__':
    import os

    print(os.getcwd())
    os.chdir('../')
    print(os.getcwd())

    train_loader, val_loader, test_loader, num_classes = dataset_loading(dataset='dtd',
                                                                         batch_size=1,
                                                                         image_size=(224, 224),
                                                                         normalize='imagenet',
                                                                         train_mode='1000')

    x, y = next(iter(train_loader))

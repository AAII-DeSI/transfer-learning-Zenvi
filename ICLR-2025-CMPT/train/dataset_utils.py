import os
import numpy as np
import torch

from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.datasets.folder import ImageFolder, default_loader


class VtabDataset(Dataset):

    def __init__(self, root: str, split: str = '1000', transform=None):

        self.root = root
        self.transform = transform

        if split == '1000':
            txt_file = os.path.join(self.root, 'train800val200.txt')
        elif split == '800':
            txt_file = os.path.join(self.root, 'train800.txt')
        elif split == '200':
            txt_file = os.path.join(self.root, 'val200.txt')
        elif split == 'test':
            txt_file = os.path.join(self.root, 'test.txt')
        else:
            raise NotImplementedError

        self.img_paths = []
        self.img_labels = []
        with open(txt_file, 'r') as f:
            for line in f:
                img_name = line.split(' ')[0]
                label = int(line.split(' ')[1])
                self.img_paths.append(os.path.join(root, img_name))
                self.img_labels.append(label)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        image = Image.open(self.img_paths[idx])

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = np.array(image)

        label = self.img_labels[idx]

        return image, label


# class VtabDataset(ImageFolder):
#     def __init__(self, root, split='1000', transform=None):
#         self.dataset_root = root
#         self.loader = default_loader
#         self.target_transform = None
#         self.transform = transform
#
#         if split == '1000':
#             txt_file = os.path.join(self.dataset_root, 'train800val200.txt')
#         elif split == '800':
#             txt_file = os.path.join(self.dataset_root, 'train800.txt')
#         elif split == '200':
#             txt_file = os.path.join(self.dataset_root, 'val200.txt')
#         elif split == 'test':
#             txt_file = os.path.join(self.dataset_root, 'test.txt')
#         else:
#             raise NotImplementedError
#
#         # train_list_path = os.path.join(self.dataset_root, 'train800.txt')
#         # test_list_path = os.path.join(self.dataset_root, 'val200.txt')
#
#         self.samples = []
#
#         with open(txt_file, 'r') as f:
#             for line in f:
#                 img_name = line.split(' ')[0]
#                 label = int(line.split(' ')[1])
#                 self.samples.append((os.path.join(root, img_name), label))


if __name__ == '__main__':
    print(os.getcwd())
    os.chdir('../')
    print(os.getcwd())

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                    ])

    cifar = VtabDataset(root='data/vtab-1k/cifar', split='test', transform=transform)

    loader = torch.utils.data.DataLoader(cifar, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

    train_features, train_labels = next(iter(loader))

import os

import torch

from model.utils_ViT import *
import train.train_utils as utils

class MyViT(nn.Module):

    def __init__(self,

                 # Basic
                 device: str = 'cuda',
                 trainable_para: tuple=('opt',),
                 output_dim: int = 10,

                 # Source Prompt Parameters
                 src_len: int = 0,
                 src_init: str = 'xavier',  # random, xavier
                 src_model: str = 'roberta',  # roberta, bert, t5
                 src_proj: str = 'none',

                 # Target Prompt Parameters
                 tgt_len: int = 0,
                 tgt_init: str = 'xavier',
                 tgt_proj: str = 'none',
                 tgt_involvement: str = 'concat',
                 tgt_task: str = 'cifar',

                 # Encoder-Specific Parameters
                 cls_pos: str = 'after prompt',  # 'before prompt', 'after prompt'
                 ):

        assert src_len >= 0
        assert cls_pos in ('before prompt', 'after prompt')

        super(MyViT, self).__init__()

        # Meta parameters
        self.device = device
        self.cls_pos = cls_pos
        self.src_len = src_len

        # Load the pretrained ViT model
        self.vit_model, vit_patch_embedder, vit_cls, vit_pos_embeddings = load_vit()

        self.hidden_dim = self.vit_model.config.hidden_size

        # Prompt Parameters
        if self.src_len > 0:

            self.src_init = src_init
            self.src_model = src_model
            self.src_proj = src_proj

            self.tgt_task = tgt_task

            prompt_embeddings = InputPrompts(device=self.device,

                                             src_len=self.src_len,
                                             src_init=self.src_init,
                                             src_model=self.src_model,
                                             src_proj=self.src_proj,
                                             src_dim=self.hidden_dim,

                                             tgt_len=tgt_len,
                                             tgt_init=tgt_init,
                                             tgt_proj=tgt_proj,
                                             tgt_dim=self.hidden_dim,
                                             tgt_involvement=tgt_involvement,
                                             tgt_task=self.tgt_task)
        else:
            prompt_embeddings = None

        # Rebuild the ViT embeddings
        embeddings = ViTEmbeddings(device=self.device,
                                   cls_token=vit_cls,
                                   position_embeddings=vit_pos_embeddings,
                                   prompt_embeddings=prompt_embeddings,
                                   patch_embedder=vit_patch_embedder,
                                   cls_pos=self.cls_pos)
        self.vit_model.embeddings = embeddings

        # Build the output classifier
        self.opt_layer = nn.Sequential(
            # nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, output_dim, bias=True),
            # nn.Dropout(0.1),
                                       )

        # Set trainable parameters of the entire ViT model
        set_trainable_parameters(self, trainable_para)

        # Print your information
        self.__print_model_info()

    def forward(self, pixel_values, return_hidden_state=False):

        # Forward pass
        last_hidden_state = self.vit_model(pixel_values).last_hidden_state  # 32*(100+1+196)*768

        # Get the last hidden state corresponding to [CLS] token
        cls_hidden_state = last_hidden_state[:, 0, :] if self.cls_pos == 'before prompt' else last_hidden_state[:, self.src_len, :]  # 32*768

        # Pass the hidden state to the output classifier
        logits = self.opt_layer(cls_hidden_state)

        if return_hidden_state:
            return logits, cls_hidden_state
        else:
            return logits

    def save_proj_prompt(self, root, accuracy: str, seed: int):

        save_dir = '{}/seed{}/{}_{}_{}'.format(root,
                                               seed,
                                               self.src_init.lower(),
                                               self.tgt_task.lower(),
                                               self.src_proj)

        if not os.path.exists('{}/seed{}'.format(root, seed)):
            os.mkdir('{}/seed{}'.format(root, seed))

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # Sanity check
        modules = os.listdir(save_dir)
        assert len(modules) == 2 or len(modules) == 0, 'The folder either should be an empty one or should contain two files'

        if len(modules) == 2:

            # The two files should have consistency
            assert eval(modules[0].split('(')[1].rstrip(')').replace('_', '.')) == eval(modules[1].split('(')[1].rstrip(')').replace('_', '.'))
            prev_acc = eval(modules[0].split('(')[1].rstrip(')').replace('_', '.'))

            # If the modules to be saved cannot outperform existing modules, then just jump out
            if prev_acc > eval(accuracy):
                print('Existing modules already have an accuracy of {}! Will not save the modules with an accuracy of {}'.format(prev_acc, accuracy))
                return
            # If they can, then delete the existing modules
            else:
                for file in modules:
                    os.remove(os.path.join(save_dir, file))

        # If there's no file in the folder, then just save the prompt and output predictor
        prompt_path = os.path.join(save_dir, 'prompt({})'.format(accuracy.replace('.', '_')))
        output_path = os.path.join(save_dir, 'output({})'.format(accuracy.replace('.', '_')))

        self.eval()
        with torch.no_grad():
            prompt = self.vit_model.embeddings.prompt_embeddings.get_prompt(1, device=self.device)[0]
        self.train()

        torch.save(prompt, prompt_path)
        torch.save(self.opt_layer, output_path)


    def __print_model_info(self):

        print('<{:-^100s}>'.format('BASIC PARAMETERS'),
              'Predefined image size: {}'.format(self.vit_model.embeddings.patch_embeddings.image_size),
              'Predefined number of image channels: {}'.format(self.vit_model.embeddings.patch_embeddings.num_channels),
              'Number of patches: {}'.format(self.vit_model.embeddings.patch_embeddings.num_patches),
              'Patch size: {}'.format(self.vit_model.embeddings.patch_embeddings.patch_size),
              'Patch embedder: {}'.format(self.vit_model.embeddings.patch_embeddings.projection),
              sep='\n')

        print('<{:-^100s}>'.format('OUTPUT LAYER'),
              self.opt_layer,
              sep='\n')

        print('<{:-^100s}>'.format('TRAINABLE PARAMETERS'))
        trainable_parameters = 0
        for name, param in self.named_parameters():
            if param.requires_grad == True:
                print('{:>20s} : {:<10s}'.format(name, str(tuple(param.shape))))
                trainable_parameters += param.numel()
        print('Total trainable parameters: {}'.format(trainable_parameters))


if __name__ == "__main__":

    print(os.getcwd())
    os.chdir('../')
    print(os.getcwd())

    # ---------------------------------------------- Dataset Parameters ------------------------------------------------
    dataset = 'cifar'

    # Create dataset
    train_loader, _, test_loader, num_classes = utils.dataset_loading(dataset=dataset,
                                                                      batch_size=64,
                                                                      normalize='imagenet',
                                                                      train_mode='1000')


    # Initialize your customized ViT model
    vit = MyViT(device='cpu',
                trainable_para=('source projector', 'opt'),
                output_dim=num_classes,

                src_len=100,
                src_init='qqp',
                src_model='roberta',
                src_proj='type1',

                tgt_len=0,
                tgt_init='xavier',
                tgt_proj='none',
                tgt_involvement='concat',
                tgt_task=dataset,

                cls_pos='after prompt')

    acc = utils.test(test_loader, vit, device='cpu')
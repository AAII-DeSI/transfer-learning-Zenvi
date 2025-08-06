import os
import math

from copy import deepcopy
from typing import Optional
from operator import mul
from functools import reduce

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn.modules.utils import _pair

import collections.abc


def load_vit(name="google/vit-base-patch16-224-in21k"):
    from transformers import ViTModel

    vit_model = ViTModel.from_pretrained(name)

    vit_patch_embedder = vit_model.embeddings.patch_embeddings

    vit_cls = vit_model.embeddings.cls_token

    vit_pos_embeddings = vit_model.embeddings.position_embeddings

    return vit_model, vit_patch_embedder, vit_cls, vit_pos_embeddings


class ViTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.
    """

    def __init__(self,
                 device,
                 cls_token,
                 position_embeddings,
                 prompt_embeddings=None,
                 patch_embedder=None,
                 img_size: int = 32,
                 num_channels: int = 3,
                 patch_size: int = 4,
                 cls_pos: str = 'after prompt',  # 'before prompt', 'after prompt'
                 ) -> None:

        assert cls_pos in ('before prompt', 'after prompt')

        super().__init__()

        self.device = device

        self.cls_token = cls_token
        self.position_embeddings = position_embeddings
        self.prompt_embeddings = prompt_embeddings

        self.cls_pos = cls_pos

        # Obtain the hidden dimension
        assert self.cls_token.shape[-1] == self.position_embeddings.shape[-1]
        # if self.prompt_embeddings is not None:
        #     assert self.cls_token.shape[-1] == self.prompt_embeddings.tgt_dim
        hidden_size = self.cls_token.shape[-1]

        self.patch_embeddings = patch_embedder
        # Rebuild the patch embedder when it is not specified
        if self.patch_embeddings is None:
            self.patch_embeddings = ViTPatchEmbeddings(img_size, patch_size, num_channels, hidden_size)
        self.num_patches = self.patch_embeddings.num_patches

        self.cls_token.to(self.device)
        self.position_embeddings.to(self.device)
        self.patch_embeddings.to(self.device)

    def forward(self,
                pixel_values: torch.Tensor,
                bool_masked_pos: Optional[torch.BoolTensor] = None,
                interpolate_pos_encoding: bool = False,
                ) -> torch.Tensor:
        """
        Note: the prompt embeddings will not be added with the positional embeddings;
        Yet in the case of NLP prompts, the prompt embeddings are added with the positional embeddings.
        """

        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)

        # Concat the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # Add positional encoding to each token
        embeddings = embeddings + self.position_embeddings

        # Concat the prompt embeddings to the embeddings
        embeddings = self.__cat_prompts(embeddings)

        return embeddings

    def __cat_prompts(self, embeddings):

        if self.prompt_embeddings is not None:

            prompt_embeddings = self.prompt_embeddings.get_prompt(embeddings.shape[0], device=self.device)

            if self.cls_pos == 'before prompt':
                return torch.cat((embeddings[:, :1, :], prompt_embeddings, embeddings[:, 1:, :]), dim=1)
            elif self.cls_pos == 'after prompt':
                return torch.cat((prompt_embeddings, embeddings), dim=1)
            else:
                raise NotImplementedError

        else:

            return embeddings


class ViTPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, image_size, patch_size, num_channels, hidden_size):
        super().__init__()

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


class InputPrompts(nn.Module):
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

    def __init__(self,

                 # Meta parameters
                 device: str = 'cuda',

                 # Source parameters
                 src_len: int = 100,
                 src_init: str = 'xavier',
                 src_model: str = 'roberta',
                 src_proj: str = 'none',
                 src_dim: int = 768,

                 # Target parameters
                 tgt_len: int = 0,
                 tgt_init: str = 'xavier',
                 tgt_proj: str = 'none',
                 tgt_dim: int = 768,
                 tgt_involvement: str = 'concat',  # 'add', 'concat'
                 tgt_task: str = 'cifar'):

        # Sanity check
        assert src_proj in ('type1', 'none') or 'attention' in src_proj or 'new' in src_proj
        assert tgt_proj in ('type1', 'none') or 'attention' in tgt_proj
        assert tgt_involvement in ('add', 'concat')
        assert src_len > 0
        assert tgt_len >= 0

        super(InputPrompts, self).__init__()

        # Meta parameters
        self.device = device

        # Source parameters
        self.src_model = src_model

        # Target parameters
        self.tgt_len = tgt_len
        self.tgt_task = tgt_task

        # Build source prompt
        self.src_embedding, self.src_len, self.src_dim = self.__build_prompt(src_init, src_len, src_dim)
        self.src_embedding.to(self.device)
        self.src_tokens_ids = torch.arange(self.src_len).long().to(self.device)

        # Build source projector
        self.src_projector = self.__build_projector(src_proj, src_dim, tgt_dim)
        self.src_projector.to(self.device)

        if self.tgt_len > 0:

            # Supplementary target parameters
            self.tgt_dim = tgt_dim
            self.tgt_init = tgt_init
            self.tgt_involvement = tgt_involvement

            # Build target prompt
            self.tgt_embedding, self.tgt_len, _ = self.__build_prompt(tgt_init, tgt_len, tgt_dim)
            self.tgt_embedding.to(self.device)
            self.tgt_tokens_ids = torch.arange(self.tgt_len).long().to(self.device)

            # Build target projector
            # 如果target prompt需要projector，那么它一定是初始化在source dimention的
            self.tgt_projector = self.__build_projector(tgt_proj, src_dim, tgt_dim)
            self.tgt_projector.to(self.device)

            # Initialize the target projector with the source projector
            if tgt_proj == src_proj == 'type1' and self.src_dim == self.tgt_dim:
                self.tgt_projector.weight.data.copy_(self.src_projector.weight.data)
                self.tgt_projector.bias.data.copy_(self.src_projector.bias.data)

        # Print the information
        self.__print_info(src_init, src_model)

    def get_prompt(self, bsz, device):

        # Retrieve the source prompts
        src_prompts = self.src_embedding(self.src_tokens_ids).unsqueeze(0).to(device)  # 1*100*768

        # Project the source prompts
        src_prompts = self.src_projector(src_prompts)

        if self.tgt_len > 0:

            # Retrieve the target prompts
            tgt_prompts = self.tgt_embedding(self.tgt_tokens_ids).unsqueeze(0).to(device)  # 1*100*768

            # Project the target prompts
            tgt_prompts = self.tgt_projector(tgt_prompts)

            # Merge the target prompts with the source prompts by either addition or concatenation
            if self.tgt_involvement == 'add':
                prompts = src_prompts + tgt_prompts
            elif self.tgt_involvement == 'concat':
                prompts = torch.cat((src_prompts, tgt_prompts), dim=1)  # 1*200*768
            else:
                raise NotImplementedError

        else:
            prompts = src_prompts

        prompts = prompts.expand(bsz, -1, -1)

        return prompts

    def __build_prompt(self, init, length, dim):

        """
        The inputs length and dim only work on xavier initialization.
        For other types of initialization, these two parameters are determined by the loaded prompt.
        """

        # Xavier initialization
        if init == 'xavier':
            embedding = nn.Embedding(length, dim)
            patch_size = _pair(16)
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + dim))
            nn.init.uniform_(embedding.weight, -val, val)

        # Normal initialization
        elif 'normal' in init:
            embedding = nn.Embedding(length, dim)
            mean, std = eval(init.split('-')[1]), eval(init.split('-')[2])
            embedding.weight.data = torch.normal(mean=mean, std=std, size=(length, dim))

        # Artificial initialization
        elif init == 'artificial':

            path = os.path.join('analysis/artificial_prompts/{}'.format(self.tgt_task))

            weight = torch.load(path)
            length, dim = weight.shape[0], weight.shape[1]

            embedding = nn.Embedding(length, dim)
            embedding.weight = nn.Parameter(weight)

        elif 'divided' in init:

            if self.tgt_task.endswith('-jpg'):
                tgt_task = self.tgt_task.rstrip('-jpg')
            else:
                tgt_task = self.tgt_task

            path = os.path.join('analysis/divided_prompts/{}-{}-{}'.format(tgt_task, init.split('-')[1], init.split('-')[2]))

            if 'proj' not in init:
                weight = torch.load(path)
            else:
                weight = torch.load(path)
                weight = self.src_projector(weight.to(self.device))

            length, dim = weight.shape[0], weight.shape[1]
            embedding = nn.Embedding(length, dim)
            embedding.weight = nn.Parameter(weight)

        # Initialize the prompt embeddings by loading from pretrained prompt embeddings
        else:

            path = os.path.join('task_prompt_emb',
                                self.task_dict[init.lower()] + 'Prompt' + self.model_dict[self.src_model.lower()],
                                'task_prompt')

            weight = torch.load(path)
            length, dim = weight.shape[0], weight.shape[1]

            embedding = nn.Embedding(length, dim)
            embedding.weight = nn.Parameter(weight)

        return embedding, length, dim

    def __build_projector(self, projection, in_dim, out_dim):

        if projection == 'type1':
            projector = nn.Linear(in_dim, out_dim, bias=True)
            return projector

        elif 'attention' in projection:

            if '-' in projection:
                new_prompt_len = eval(projection.split('-')[1])
            else:
                assert projection == 'attention'
                new_prompt_len = self.src_len

            projector = AttentionProjector(output_len=new_prompt_len, in_dim=in_dim, out_dim=out_dim)
            return projector

        elif 'new' in projection:

            tgt_len = eval(projection.split('-')[1])

            projector = NewAttentionProjector(tgt_len=tgt_len, in_dim=in_dim, out_dim=out_dim)
            return projector

        elif projection == 'none':
            projector = nn.Identity()
            return projector

        else:
            raise NotImplementedError

    def __print_info(self, src_task, src_model):

        print('<{:-^100s}>'.format('Source Prompt Info'),
              'Length: {}'.format(self.src_len),
              'Dimension: {}'.format(self.src_dim),
              'Source model: {}'.format(src_model),
              'Initialization: {}'.format(src_task),
              sep='\n')
        print('Source projector: \n', self.src_projector)

        if self.tgt_len > 0:
            print('<{:-^100s}>'.format('Target Prompt Info'),
                  'Length: {}'.format(self.tgt_len),
                  'Dimension: {}'.format(self.tgt_dim),
                  'Initialization: {}'.format(self.tgt_init),
                  'Involvement: {}'.format(self.tgt_involvement),
                  sep='\n')
            print('Target projector: \n', self.tgt_projector)


class AttentionProjector(nn.Module):

    def __init__(self,
                 output_len: int = 50,
                 in_dim: int = 768,
                 out_dim: int = 768,
                 initializer_range: float = 0.02,
                 dropout: float = 0.):
        super(AttentionProjector, self).__init__()

        self.output_len = output_len
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Query vectors used to calculate the attention scores with key vectors projected from source prompts
        self.query = nn.Parameter(torch.randn(self.output_len, self.in_dim))

        # Projection matrices that directly works on source prompts
        self.W_k = nn.Linear(self.in_dim, self.in_dim)
        self.W_v = nn.Linear(self.in_dim, self.out_dim)

        self.dropout = nn.Dropout(dropout)

        # Handle the initialization
        # self.W_k.weight.data = nn.init.trunc_normal_(self.W_k.weight.data.to(torch.float32),
        #                                              mean=0.0,
        #                                              std=initializer_range).to(self.W_k.weight.dtype)
        #
        # self.W_v.weight.data = nn.init.trunc_normal_(self.W_v.weight.data.to(torch.float32),
        #                                              mean=0.0,
        #                                              std=initializer_range).to(self.W_v.weight.dtype)
        #
        # self.query.data = nn.init.trunc_normal_(self.query.data.to(torch.float32),
        #                                         mean=0.0,
        #                                         std=initializer_range).to(self.query.dtype)

    def forward(self, src_prompts):
        # 1*100*768 to 100*768
        src_prompts = src_prompts.squeeze(0)

        # src_prompts = self.dropout(src_prompts)

        # Obtain the keys and values
        keys = self.W_k(src_prompts)  # 100*768
        values = self.W_v(src_prompts)  # 100*768

        # Calculate the attention score. The resulting matrix should have a dimension of 50*100
        # attention_score = torch.softmax(torch.matmul(self.query, keys.t()) / torch.sqrt(torch.scalar_tensor(self.in_dim)), dim=1)  # 50*100
        attention_score = torch.softmax(torch.matmul(self.query, keys.t()), dim=1)  # 50*100

        attention_score = self.dropout(attention_score)

        attention_score = attention_score.unsqueeze(dim=-1).expand(-1, -1, self.out_dim)  # 50*100*768

        projected_src_prompts = torch.sum(torch.mul(attention_score, values), dim=1)  # 50*768

        return projected_src_prompts.unsqueeze(0)  # 1*50*768

# class AttentionProjector(nn.Module):
#
#     """
#     This is the simplified version of the attention projector
#     """
#
#     def __init__(self,
#                  output_len: int = 50,
#                  in_dim: int = 768,
#                  out_dim: int = 768,
#                  dropout: float = 0.1):
#
#         super(AttentionProjector, self).__init__()
#
#         self.output_len = output_len
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#
#         self.A = nn.Parameter(torch.ones(output_len, 100) / 100)  # 50*100
#
#         self.W_v = nn.Linear(in_dim, out_dim)
#
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, src_prompts):
#
#         # 1*100*768 to 100*768
#         src_prompts = src_prompts.squeeze(0)
#
#         # Obtain the keys and values
#         V = self.W_v(src_prompts)  # 100*768
#
#         # Calculate the attention score. The resulting matrix should have a dimension of 50*100
#
#         attention_score = torch.softmax(self.A, dim=1)  # 50*100
#
#         attention_score = self.dropout(attention_score)
#
#         attention_score = attention_score.unsqueeze(dim=-1).expand(-1, -1, self.out_dim)  # 50*100*768
#
#         projected_src_prompts = torch.sum(torch.mul(attention_score, V), dim=1)  # 50*768
#
#         return projected_src_prompts.unsqueeze(0)  # 1*50*768

# class AttentionProjector(nn.Module):
#
#     def __init__(self, output_len: int = 50, in_dim: int = 768, out_dim: int=768):
#
#         super(AttentionProjector, self).__init__()
#
#         self.output_len = output_len
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#
#         # Query vectors used to calculate the attention scores with key vectors projected from source prompts
#         self.query = nn.Parameter(torch.randn(self.output_len, self.in_dim))  # 100 * 768
#
#         # Projection matrices that directly works on source prompts
#         self.W_q = nn.Linear(self.in_dim, self.in_dim)
#         self.W_k = nn.Linear(self.in_dim, self.in_dim)
#         self.W_v = nn.Linear(self.in_dim, self.out_dim)
#
#     def forward(self, src_prompts):
#
#         query_dp = deepcopy(self.query.data)
#
#         # 1*100*768 to 100*768
#         src_prompts = src_prompts.squeeze(0)
#
#         # Obtain the keys and values
#         Q = self.W_q(query_dp)  # 100*768
#         K = self.W_k(torch.cat((src_prompts, self.query), dim=0))  # 200*768
#         V = self.W_v(torch.cat((src_prompts, self.query), dim=0))  # 200*768
#
#         # Calculate the attention score. The resulting matrix should have a dimension of 50*100
#         attention_score = torch.softmax(torch.matmul(Q, K.t()) / torch.sqrt(torch.scalar_tensor(self.in_dim)), dim=1)  # 100*200
#         attention_score = attention_score.unsqueeze(dim=-1).expand(-1, -1, self.out_dim)  # 100*200*768
#
#         projected_src_prompts = torch.sum(torch.mul(attention_score, V), dim=1)  # 100*768
#
#         return projected_src_prompts.unsqueeze(0)  # 1*100*768

class NewAttentionProjector(nn.Module):

    def __init__(self, tgt_len: int = 50, output_len: int = 100, in_dim: int = 768, out_dim: int = 768):

        super(NewAttentionProjector, self).__init__()

        assert tgt_len <= output_len // 2
        assert output_len % 2 == 0

        self.query_comp_len = output_len - (2 * tgt_len)

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.tgt_prompt = nn.Parameter(torch.randn(tgt_len, self.out_dim))  # 10 * 768

        if self.query_comp_len > 0:
            self.query_comp = nn.Parameter(torch.randn(self.query_comp_len, self.out_dim))  # 80 * 768

        self.W_q = nn.Linear(self.out_dim, self.in_dim)
        self.W_k = nn.Linear(self.in_dim, self.in_dim)
        self.W_v = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, src_prompts):

        tgt_prompt_dp = deepcopy(self.tgt_prompt.data)

        src_prompts = src_prompts.squeeze(0)

        if self.query_comp_len > 0:
            Q = torch.cat((self.W_q(tgt_prompt_dp), self.query_comp), dim=0)
        else:
            Q = self.W_q(tgt_prompt_dp)
        # Q = self.W_q(tgt_prompt_dp)  # 50, 768
        # K = self.W_k(torch.cat((src_prompts, tgt_prompt_dp), dim=0))  # 150, 768
        K = self.W_k(src_prompts)
        # V = self.W_v(torch.cat((src_prompts, tgt_prompt_dp), dim=0))
        V = self.W_v(src_prompts)

        # attention_score = torch.softmax(torch.matmul(Q, K.t()) / torch.sqrt(torch.scalar_tensor(self.in_dim)), dim=1)  # 50, 150
        attention_score = torch.softmax(torch.matmul(Q, K.t()), dim=1)  # 50, 150
        attention_score = attention_score.unsqueeze(dim=-1).expand(-1, -1, self.out_dim)  # 50, 150, 768
        projected_src_prompts = torch.sum(torch.mul(attention_score, V), dim=1)  # 50, 768

        final_prompts = torch.cat((projected_src_prompts, self.tgt_prompt), dim=0)  # 100, 768

        return final_prompts.unsqueeze(0)


def set_trainable_parameters(MyViTModel, trainable_para):
    # Freeze all parameters
    for name, param in MyViTModel.named_parameters():
        param.requires_grad = False

    # Unlock prompt embeddings for training
    if 'source prompt' in trainable_para:
        for name, param in MyViTModel.named_parameters():
            if 'prompt_embeddings.src_embedding' in name:
                param.requires_grad = True

    # Unlock target prompt embeddings for training
    if 'target prompt' in trainable_para:
        for name, param in MyViTModel.named_parameters():
            if 'prompt_embeddings.tgt_embedding' in name:
                param.requires_grad = True

    # Unlock source projector for training
    if 'source projector' in trainable_para:
        for name, param in MyViTModel.named_parameters():
            if 'prompt_embeddings.src_projector' in name:
                param.requires_grad = True

    # Unlock target projector for training
    if 'target projector' in trainable_para:
        for name, param in MyViTModel.named_parameters():
            if 'prompt_embeddings.tgt_projector' in name:
                param.requires_grad = True

    # Unlock output predictor for training
    if 'opt' in trainable_para:
        for name, param in MyViTModel.named_parameters():
            if 'opt_layer' in name:
                param.requires_grad = True

    # Unlock layer norm affine parameters for training
    if 'ln' in trainable_para:
        for name, param in MyViTModel.named_parameters():
            if 'layernorm' in name:
                param.requires_grad = True

    # Unlock positional encoding for training
    if 'wpe' in trainable_para:
        for name, param in MyViTModel.named_parameters():
            if 'position_embeddings' in name:
                param.requires_grad = True


if __name__ == "__main__":
    import os

    print(os.getcwd())
    os.chdir('../')
    print(os.getcwd())

    device = 'cpu'

    # ------------------------------------------------- Prompt Check ---------------------------------------------------
    """
    This is to check that your prompt functions accordingly.
    """

    # Initialize prompt embeddings
    prompt_embeddings = InputPrompts(device=device,

                                     src_len=100,
                                     src_init='snli',
                                     src_model='roberta',
                                     src_proj='attention-70',
                                     src_dim=768,

                                     tgt_len=30,
                                     tgt_init='normal-0.01126-0.51789',
                                     tgt_proj='none',
                                     tgt_dim=768,
                                     tgt_involvement='concat',  # 'add', 'concat'
                                     tgt_task='svhn')

    prompt = prompt_embeddings.get_prompt(bsz=64, device=device)
    # ------------------------------------------------------------------------------------------------------------------

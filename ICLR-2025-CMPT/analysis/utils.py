import os
import random
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

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


def loss_between_prompt_and_patchset(device, loss, source_prompt, patch_embeds_dataloader):
    """
    This function calculates the loss (typically MMD) between the source prompt and the patch embeds. Note: the patch
    embeds come in as a pytorch's dataloader (patch_embeds_dataloader), with a batch size equal to the number of
    embeddings in source prompt.
    """

    with torch.no_grad():
        with tqdm(total=len(patch_embeds_dataloader)) as pbar:
            pbar.set_description('Calculating loss between source prompt and patch embeds')

            step = 0
            l = 0

            for batch, x in enumerate(patch_embeds_dataloader):
                x = x.to(device)

                l += loss(source_prompt.to(device), x)

                step += 1

                pbar.update(1)

    return l / step


def load_source_prompt(src_task, src_model):
    """
    Load 1 source prompt given the source task and source model.
    """

    path = os.path.join('task_prompt_emb',
                        task_dict[src_task.lower()] + 'Prompt' + model_dict[src_model.lower()],
                        'task_prompt')

    prompt = torch.load(path)

    return prompt


def load_source_prompts_multi(source_tasks, source_model):
    """
    Load multiple source prompts given the source tasks and source model.
    """

    source_prompt_ori = []

    for source_task in source_tasks:
        source_prompt = load_source_prompt(source_task, source_model)
        source_prompt_ori.append(source_prompt)

    source_prompt_ori = torch.cat(source_prompt_ori, dim=0)

    return source_prompt_ori


def extract_patch_embeds(device, test_loader, dataset):
    embedder = load_vit_embedder(device)

    with torch.no_grad():
        patch_embeds = []

        embedder.eval()

        with tqdm(total=len(test_loader)) as pbar:
            pbar.set_description('Extracting patch embeddings from {}...'.format(dataset.upper()))

            for batch, (x, _) in enumerate(test_loader):
                # Get inputs
                x = x.to(device)

                # Inference
                patch_embed = embedder(x)

                patch_embed = patch_embed[:, 1:, :]

                patch_embeds.append(patch_embed.to('cpu'))

                pbar.update(1)

        patch_embeds = torch.cat(patch_embeds, dim=0)

    return patch_embeds


def extract_source_embddings(src_dataset, tokenizer, embedder, reduction, num_per_class=0):

    if num_per_class > 0:
        new_src_dataset_dict = {}
        for src_data in src_dataset:
            if src_data.label not in new_src_dataset_dict.keys():
                new_src_dataset_dict[src_data.label] = [src_data]
            else:
                new_src_dataset_dict[src_data.label].append(src_data)

        src_dataset = []
        for key in new_src_dataset_dict.keys():
            src_dataset += random.sample(new_src_dataset_dict[key], num_per_class)


    embeddings = []

    with tqdm(total=len(src_dataset)) as pbar:
        pbar.set_description('Extracting source embeddings')

        for instance in src_dataset:

            if hasattr(instance, 'text_a'):
                text_a = ' ' + instance.text_a
            else:
                text_a = ''

            if hasattr(instance, 'text_b'):
                text_b = ' ' + instance.text_b
            else:
                text_b = ''

            text = text_a + text_b

            token = torch.Tensor(tokenizer(text)['input_ids']).long()

            with torch.no_grad():
                embedding = embedder(token)

            if reduction == 'mean':
                embedding = torch.mean(embedding, dim=0, keepdim=True)

            embeddings.append(embedding)

            pbar.update(1)

    embeddings = torch.cat(embeddings, dim=0)

    return embeddings


def load_vit_embedder(device, model_name='google/vit-base-patch16-224-in21k'):
    from transformers import ViTModel

    vit_model = ViTModel.from_pretrained(model_name)

    embedder = vit_model.embeddings

    embedder.to(device)

    return embedder


class EmbeddingDataset(Dataset):

    def __init__(self, embeds, reduction='mean'):

        if len(embeds.shape) == 3:

            # Reduce the patch embeddings from 10000*196*768 to x*768
            if reduction == 'mean':
                self.embeds = torch.mean(embeds, dim=1)
            elif reduction == 'none':
                self.embeds = embeds.view(-1, embeds.shape[-1])
            else:
                raise NotImplementedError

        elif len(embeds.shape) == 2:
            self.embeds = embeds

        else:
            raise NotImplementedError

    def __len__(self):
        return self.embeds.shape[0]

    def __getitem__(self, idx):
        return self.embeds[idx]

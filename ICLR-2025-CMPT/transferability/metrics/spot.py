'''
Transferability measurements used in paper: SPoT: Better Frozen Model Adaptation through Soft Prompt Transfer
ArXiv: https://arxiv.org/abs/2110.07904
SPoT defines the transferability between tasks through the cosine similarity between their prompts.
There are two ways of calculating the cosine similarity:
1. COSINE SIMILARITY OF AVERAGE TOKENS: compute the cosine similarity between the average pooled representations of the
prompt tokens.
2. PER-TOKEN AVERAGE COSINE SIMILARITY: compute the average cosine similarity between every prompt token pair.
'''

import torch
import torch.nn.functional as F


def cosine_average(src_prompt, tgt_prompt, device='cpu'):
    # Simple!
    return F.cosine_similarity(torch.mean(src_prompt, dim=0, keepdim=True),
                               torch.mean(tgt_prompt, dim=0, keepdim=True)).mean()


def cosine_max(src_prompt, tgt_prompt, device='cpu'):
    return F.cosine_similarity(torch.max(src_prompt, dim=0)[0].unsqueeze(0),
                               torch.max(tgt_prompt, dim=0)[0].unsqueeze(0)).mean()


def cosine_individual(src_prompt, tgt_prompt, device='cpu'):
    # Simple!
    return F.cosine_similarity(src_prompt.unsqueeze(1), tgt_prompt.unsqueeze(0), dim=-1).mean()

def euclidean_average(src_prompt, tgt_prompt, device='cpu'):
    return  - F.pairwise_distance(torch.mean(src_prompt, dim=0, keepdim=True),
                                  torch.mean(tgt_prompt, dim=0, keepdim=True)).mean()


if __name__ == '__main__':
    src_prompt = torch.rand((100, 768))
    tgt_prompt = torch.rand((100, 768))

    a1 = cosine_average(src_prompt, tgt_prompt)
    b1 = cosine_max(src_prompt, tgt_prompt)
    c1 = cosine_individual(src_prompt, tgt_prompt)

    a2 = euclidean_average(src_prompt, tgt_prompt)

    print('COSINE MEAN: {}'.format(a1))
    print('COSINE MAX: {}'.format(b1))
    print('COSINE INDIVIDUAL: {}'.format(c1))
    print('EUCLIDEAN MEAN: {}'.format(a2))

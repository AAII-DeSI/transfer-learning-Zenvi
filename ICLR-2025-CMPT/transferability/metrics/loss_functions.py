import torch
import torch.nn as nn

class DistanceAverageEmbedding(nn.Module):

    def __init__(self, metric='Euclidean', reduction='mean'):

        assert metric in ('Euclidean', 'Cosine')
        assert reduction in ('mean', 'max')

        super(DistanceAverageEmbedding, self).__init__()

        self.metric = metric
        self.reduction = reduction

        if self.metric == 'Euclidean':
            self.metric_function = nn.PairwiseDistance()
        else:
            self.metric_function = nn.CosineSimilarity()

    def forward(self, src_prompts, tgt_prompt):

        # src_prompts: (13, 100, 768) through reduction to (13, 768)
        # tgt_prompt: (100, 768) through reduction to (768, )

        if self.reduction == 'mean':
            src_prompts = src_prompts.mean(dim=1)
            tgt_prompt = tgt_prompt.mean(dim=0)
        elif self.reduction == 'max':
            src_prompts = src_prompts.max(dim=1)[0]
            tgt_prompt = tgt_prompt.max(dim=0)[0]

        loss = self.metric_function(src_prompts, tgt_prompt)
        loss = torch.sum(loss) / loss.shape[0]

        if self.metric == 'Euclidean':
            return loss
        else:
            return -loss

class DistanceInstanceEmbedding(nn.Module):

    def __init__(self, metric='Euclidean', reduction='mean'):

        assert metric in ('Euclidean', 'Cosine')

        super(DistanceInstanceEmbedding, self).__init__()

        self.metric = metric

        if self.metric == 'Euclidean':
            self.metric_function = nn.PairwiseDistance()
        else:
            self.metric_function = nn.CosineSimilarity(dim=-1)

    def forward(self, src_prompts, tgt_prompt):

        loss = self.metric_function(src_prompts.unsqueeze(2), tgt_prompt.unsqueeze(0).unsqueeze(0)).mean()

        return loss

if __name__ == '__main__':

    src_prompts = torch.rand((13, 100, 768))
    tgt_prompt = torch.rand((100, 768))

    loss_fn = DistanceInstanceEmbedding(metric='Cosine')

    loss = loss_fn(src_prompts, tgt_prompt)

    print(loss)
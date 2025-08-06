import torch
import torch.nn as nn

from tqdm import tqdm


class MaximumMeanDiscrepancyLoss(nn.Module):

    def __init__(self, kernel_mul=2.0, kernel_num=5):

        super(MaximumMeanDiscrepancyLoss, self).__init__()

        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = None

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

        num_samples = int(source.shape[0] + target.shape[0])

        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.shape[0]), int(total.shape[0]), int(total.shape[1]))
        total1 = total.unsqueeze(1).expand(int(total.shape[0]), int(total.shape[0]), int(total.shape[1]))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (num_samples ** 2 - num_samples)

        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                       fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss

class Distance(nn.Module):

    def __init__(self, device, metric='Euclidean'):

        assert metric in ('Euclidean', 'Cosine')

        super(Distance, self).__init__()

        self.metric = metric
        self.device = device

        if self.metric == 'Euclidean':
            self.metric_function = nn.PairwiseDistance()
        else:
            self.metric_function = nn.CosineSimilarity(dim=-1)

    def forward(self, source_prompt, patch_embeds, src_task):

        # source_prompt: (100, 768)
        # patch_embeds: (1000, 196, 768)

        d = 0

        with tqdm(total=patch_embeds.shape[0]) as pbar:
            pbar.set_description('{} {}'.format(src_task, self.metric))

            for patch_embed in patch_embeds:

                source_prompt, patch_embed = source_prompt.to(self.device), patch_embed.to(self.device)

                d += self.metric_function(source_prompt.unsqueeze(1), patch_embed.unsqueeze(0)).mean()

                pbar.update(1)

        return d/patch_embeds.shape[0]


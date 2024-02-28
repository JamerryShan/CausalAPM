# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# x = torch.randn(5, 5)
# target = torch.tensor([0, 2, 3, 1, 4])
# one_hot = F.one_hot(target).float()

# print(one_hot)

# x = torch.tensor([0.4167, 0.2000, 0.4375, 0.5714, 0.5769, 0.5714, 0.1667, 0.4000, 0.2703,
#         0.3333, 0.5000, 0.0964, 0.1429, 0.1875, 0.8182, 0.4167, 0.5000, 0.2308,
#         0.2258, 0.4545, 0.6000, 0.1143, 0.4815, 0.5000, 0.5385, 0.3636, 0.2333,
#         0.2069, 0.2647, 0.5556, 0.4000, 0.5238]).view(-1, 1)

# y = torch.ones_like(x) - x

# print(x)
# print(y)

# z = torch.cat((x, y), dim=-1)

# print(z.shape[0])


class CLUBv2(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    def __init__(self, x_dim, y_dim, lr=1e-3, beta=0):
        super(CLUBv2, self).__init__()
        self.hiddensize = y_dim
        self.version = 2
        self.beta = beta

    def mi_est_sample(self, x_samples, y_samples):
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        positive = torch.zeros_like(y_samples)
        negative = - (y_samples - y_samples[random_index]) ** 2 / 2.
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        # return upper_bound/2.
        return upper_bound

    def mi_est(self, x_samples, y_samples):  # [nsample, 1]
        positive = torch.zeros_like(y_samples)

        prediction_1 = y_samples.unsqueeze(1)  # [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)  # [1,nsample,dim]
        negative = - ((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2.   # [nsample, dim]
        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        # return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean(), positive.sum(dim = -1).mean(), negative.sum(dim = -1).mean()

    def loglikeli(self, x_samples, y_samples):
        return 0

    def update(self, x_samples, y_samples, steps=None):
        # no performance improvement, not enabled
        if steps:
            beta = self.beta if steps > 1000 else self.beta * steps / 1000  # beta anealing
        else:
            beta = self.beta

        return self.mi_est_sample(x_samples, y_samples) * self.beta
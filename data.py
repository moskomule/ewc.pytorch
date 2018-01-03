import random
import torch
from torchvision import datasets


class PermutedMNIST(datasets.MNIST):

    def __init__(self, root="~/.torch/data/mnist", train=True, permute_idx=None):
        super(PermutedMNIST, self).__init__(root, train, download=True)
        assert len(permute_idx) == 28 * 28
        if self.train:
            self.train_data = torch.stack([img.float().view(-1)[permute_idx] / 255
                                           for img in self.train_data])
        else:
            self.test_data = torch.stack([img.float().view(-1)[permute_idx] / 255
                                          for img in self.test_data])

    def __getitem__(self, index):

        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        return img, target

    def get_sample(self, sample_size):
        sample_idx = random.sample(range(len(self)), sample_size)
        return [img for img in self.train_data[sample_idx]]

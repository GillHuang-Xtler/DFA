from .dataset import Dataset
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[int(self.idxs[item])]
        return image, label

class FashionMNISTnoniidDataset():
    MAX_NUM_CLASSES_PER_CLIENT = 5
    BATCH_SIZE = 100

    def __init__(self, args):
        super(FashionMNISTnoniidDataset, self).__init__(args)

    def load_train_dataset(self):
        self.get_args().get_logger().debug("Loading Fashion MNIST train data")

        train_data =datasets.FashionMNIST(self.get_args().get_data_path(), train=True, download=True,
                              transform=transforms.Compose([transforms.ToTensor()]))

        idxs = np.arange(len(train_data))
        labels = train_data.train_labels.numpy()
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]

        each_part_size = len(train_data) // self.args.total_part

        partition = []
        for i in range(self.args.total_part):
            first = each_part_size * i
            last = each_part_size * (i + 1)
            part = idxs[first:last]
            partition.append(part)

        train_loader = []
        for part in partition:
            loader = DataLoader(DatasetSplit(train_data, part), batch_size=self.args.batchsize, shuffle=True)
            train_loader.append(loader)


    def get_train_data(self, part_id):
        return self.train_loader[part_id]


    def load_train_dataset(self):
        test_dataset = datasets.FashionMNIST(self.get_args().get_data_path(), train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
        return test_loader
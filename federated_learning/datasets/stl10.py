from .dataset import Dataset
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

class STL10Dataset(Dataset):

    def __init__(self, args):
        super(STL10Dataset, self).__init__(args)

    def load_train_dataset(self):
        self.get_args().get_logger().debug("Loading CIFAR10 train data")

        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        transform = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomCrop(96),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        train_dataset = datasets.STL10(root=self.get_args().get_data_path(), split='train', download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))

        train_data = self.get_tuple_from_data_loader(train_loader)

        self.get_args().get_logger().debug("Finished loading STL10 train data")

        return train_data

    def load_test_dataset(self):
        self.get_args().get_logger().debug("Loading CIFAR10 test data")

        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        test_dataset = datasets.STL10(root=self.get_args().get_data_path(), split='test', download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        test_data = self.get_tuple_from_data_loader(test_loader)

        self.get_args().get_logger().debug("Finished loading STL10 test data")

        return test_data

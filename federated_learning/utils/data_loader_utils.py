import numpy
from .label_replacement import apply_class_label_replacement
import os
import pickle
import random
from ..datasets import Dataset
import torch.utils.data

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np


def generate_data_loaders_from_distributed_dataset(distributed_dataset, batch_size):
    """
    Generate data loaders from a distributed dataset.

    :param distributed_dataset: Distributed dataset
    :type distributed_dataset: list(tuple)
    :param batch_size: batch size for data loader
    :type batch_size: int
    """
    data_loaders = []
    for worker_training_data in distributed_dataset:
        data_loaders.append(Dataset.get_data_loader_from_data(batch_size, worker_training_data[0], worker_training_data[1], shuffle=True))

    return data_loaders

def load_train_data_loader(logger, args):
    """
    Loads the training data DataLoader object from a file if available.

    :param logger: loguru.Logger
    :param args: Arguments
    """
    if os.path.exists(args.get_train_data_loader_pickle_path()):
        return load_data_loader_from_file(logger, args.get_train_data_loader_pickle_path())
    else:
        logger.error("Couldn't find train data loader stored in file")

        raise FileNotFoundError("Couldn't find train data loader stored in file")

def load_benign_data_loader(logger, args):
    """
    Loads the training data DataLoader object from a file if available.

    :param logger: loguru.Logger
    :param args: Arguments
    """
    if os.path.exists(args.get_benign_data_loader_pickle_path()):
        return load_data_loader_from_file(logger, args.get_benign_data_loader_pickle_path())
    else:
        logger.error("Couldn't find train data loader stored in file")

        raise FileNotFoundError("Couldn't find train data loader stored in file")

def load_malicious_data_loader(logger, args):
    """
    Loads the training data DataLoader object from a file if available.

    :param logger: loguru.Logger
    :param args: Arguments
    """
    if os.path.exists(args.get_malicious_data_loader_pickle_path()):
        return load_data_loader_from_file(logger, args.get_malicious_data_loader_pickle_path())
    else:
        logger.error("Couldn't find train data loader stored in file")

        raise FileNotFoundError("Couldn't find train data loader stored in file")


def generate_train_loader(args, dataset):
    train_dataset = dataset.get_train_dataset()
    X, Y = shuffle_data(args, train_dataset)
    return dataset.get_data_loader_from_data(args.get_batch_size(), X, Y)

def generate_train_loader_mal(args, dataset):
    X = []
    Y = []
    for c in dataset:
        X.append(c[0].detach().numpy())
        Y.append(c[1].detach().numpy())
    X = np.array(X)
    Y = np.array(Y)

    batch_size = args.get_batch_size()

    X_torch = torch.from_numpy(X).float()

    Y_torch = torch.from_numpy(Y).long()
    dataset = TensorDataset(X_torch, Y_torch)

    return DataLoader(dataset, batch_size=batch_size)

# def generate_benign_loader(args, dataset):
#     benign_dataset = dataset.get_benign_dataset()
#     X, Y = shuffle_data(args, benign_dataset)
#
#     return dataset.get_data_loader_from_data(args.get_batch_size(), X, Y)
#
# def generate_malicious_loader(args, dataset):
#     malicious_dataset = dataset.get_malicious_dataset()
#     X, Y = shuffle_data(args, malicious_dataset)
#
#     return dataset.get_data_loader_from_data(args.get_batch_size(), X, Y)
#
# def generate_free_loader(args, dataset):
#     malicious_dataset = dataset.get_free_dataset()
#     X, Y = shuffle_data(args, malicious_dataset)
#
#     return dataset.get_data_loader_from_data(args.get_batch_size(), X, Y)

def generate_train_loader_sample(args, dataset):
    train_dataset = dataset.get_train_dataset()
    X, Y = shuffle_data_sample(args, train_dataset)

    return dataset.get_data_loader_from_data(args.get_batch_size(), X, Y)

def load_test_data_loader(logger, args):
    """
    Loads the test data DataLoader object from a file if available.

    :param logger: loguru.Logger
    :param args: Arguments
    """
    if os.path.exists(args.get_test_data_loader_pickle_path()):
        return load_data_loader_from_file(logger, args.get_test_data_loader_pickle_path())
    else:
        logger.error("Couldn't find test data loader stored in file")

        raise FileNotFoundError("Couldn't find train data loader stored in file")

def load_data_loader_from_file(logger, filename):
    """
    Loads DataLoader object from a file if available.

    :param logger: loguru.Logger
    :param filename: string
    """
    logger.info("Loading data loader from file: {}".format(filename))

    with open(filename, "rb") as f:
        return load_saved_data_loader(f)

def generate_test_loader(args, dataset):
    test_dataset = dataset.get_test_dataset()
    X, Y = shuffle_data(args, test_dataset)

    return dataset.get_data_loader_from_data(args.get_test_batch_size(), X, Y)

def shuffle_data_sample(args, dataset):
    data = list(zip(dataset[0], dataset[1]))
    random.shuffle(data)
    X, Y = zip(*data)
    X = numpy.asarray(X)
    Y = numpy.asarray(Y)
    print(type(X), len(Y))

    return X[:6000], Y[:6000]

def shuffle_data(args, dataset):
    data = list(zip(dataset[0], dataset[1]))

    random.shuffle(data)

    X, Y = zip(*data)

    X = numpy.asarray(X)
    Y = numpy.asarray(Y)

    return X, Y

def load_saved_data_loader(file_obj):
    return pickle.load(file_obj)

def save_data_loader_to_file(data_loader, file_obj):
    pickle.dump(data_loader, file_obj)

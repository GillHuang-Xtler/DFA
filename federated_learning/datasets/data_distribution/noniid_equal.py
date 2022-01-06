import torch
import random
import numpy as np
from random import sample


def distribute_batches_1_class(train_data_loader, num_workers, args):
    """
    Gives each worker the same number of batches of training data but one user only have one class.

    :param train_data_loader: Training data loader
    :type train_data_loader: torch.utils.data.DataLoader
    :param num_workers: number of workers
    :type num_workers: int
    """

    args.get_logger().info("Distribute data non-iid in to 1 class")
    distributed_dataset = [[] for i in range(num_workers)]

    class_slot = num_workers / args.get_num_classes()
    if class_slot - int(class_slot) > 0:
        args.get_logger().info("unmatched class and client number")
    else:
        class_slot = int(class_slot)


    for i in range(args.get_num_classes()):
        for batch_idx, (data, target) in enumerate(train_data_loader):
            reduce = (target == i).nonzero()
            target_r = torch.index_select(target, 0, reduce.view(-1))
            data_r = torch.index_select(data, 0, reduce.view(-1))
            rand_idx = random.randint(0, class_slot-1)
            distributed_dataset[i*class_slot + rand_idx].append((data_r, target_r))

    return distributed_dataset


def distribute_batches_2_class(train_data_loader, num_workers, args):
    """
    Gives each worker the same number of batches of training data but one user only have two classes.

    :param train_data_loader: Training data loader
    :type train_data_loader: torch.utils.data.DataLoader
    :param num_workers: number of workers
    :type num_workers: int
    """

    args.get_logger().info("Distribute data non-iid in to 2 classes")
    distributed_dataset = [[] for i in range(num_workers)]

    class_slot = num_workers / args.get_num_classes()
    if class_slot - int(class_slot) > 0:
        args.get_logger().info("unmatched class and client number")
    else:
        class_slot = int(class_slot)

    for i in range(0, args.get_num_classes(), 2):
        for batch_idx, (data, target) in enumerate(train_data_loader):
            if batch_idx < 1000:
                reduce_1 = (target == i).nonzero()
                target_1 = torch.index_select(target, 0, reduce_1.view(-1))
                data_1 = torch.index_select(data, 0, reduce_1.view(-1))
                reduce_2 = (target == i+1).nonzero()
                target_2 = torch.index_select(target, 0, reduce_2.view(-1))
                data_2 = torch.index_select(data, 0, reduce_2.view(-1))
                rand_idx = random.randint(0, class_slot*2-1)
                distributed_dataset[i*class_slot + rand_idx].append((data_1, target_1))
                distributed_dataset[i*class_slot + rand_idx].append((data_2, target_2))
    return distributed_dataset

def distribute_batches_noniid_mal(benign_data_loader, malicious_data_loader, num_workers, args):
    """
    Gives each worker the same number of batches of training data and one user only have two class except the malicious.

    :param train_data_loader: Training data loader
    :type train_data_loader: torch.utils.data.DataLoader
    :param num_workers: number of workers
    :type num_workers: int
    """

    args.get_logger().info("Distribute data non-iid in to 2 classes")
    distributed_dataset = [[] for i in range(num_workers)]

    class_slot = (num_workers - args.get_num_workers() * args.get_mal_prop()) / args.get_num_classes()
    if class_slot - int(class_slot) > 0:
        args.get_logger().info("Unmatched class and client number")
    else:
        class_slot = int(class_slot)


    # for benign users
    for i in range(0, args.get_num_classes(), 2):
        for batch_idx, (data, target) in enumerate(benign_data_loader):
            if batch_idx < 1000:
                reduce_1 = (target == i).nonzero()
                target_1 = torch.index_select(target, 0, reduce_1.view(-1))
                data_1 = torch.index_select(data, 0, reduce_1.view(-1))
                reduce_2 = (target == i+1).nonzero()
                target_2 = torch.index_select(target, 0, reduce_2.view(-1))
                data_2 = torch.index_select(data, 0, reduce_2.view(-1))
                rand_idx = random.randint(0, class_slot*2-1)
                distributed_dataset[i*class_slot + rand_idx].append((data_1, target_1))
                distributed_dataset[i*class_slot + rand_idx].append((data_2, target_2))

    # for malicious users
    for batch_idx, (data, target) in enumerate(malicious_data_loader):
        if args.get_mal_prop() > 0:
            worker_idx = batch_idx % (num_workers * args.get_mal_prop())

            distributed_dataset[int(worker_idx + args.get_num_workers() - args.get_num_workers() * args.get_mal_prop())].append((data, target))

    return distributed_dataset

def distribute_batches_dirichlet(train_data_loader, num_workers, mal_prop, args):
    """
    Gives each worker the same number of batches of training data and one user only have two class except the malicious.

    :param train_data_loader: Training data loader
    :type train_data_loader: torch.utils.data.DataLoader
    :param num_workers: number of workers
    :type num_workers: int
    """

    args.get_logger().info("Distribute data non-iid in Dirichlet distribution")
    distributed_dataset = [[] for i in range(num_workers)]
    beta = args.get_beta()

    num_benign_workers = num_workers*(1-mal_prop)

    min_size = 0
    min_require_size = 10
    num_class = args.get_num_classes()

    # for benign users
    N = 5000
    # len(train_data_loader*args.get_batch_size()
    print("N = " + str(len(train_data_loader)))
    np.random.seed(2022)

    # while min_size < min_require_size:
    idx_batch = [[] for i in range(int(num_benign_workers))]
    for i in range(num_class):
        dataset = []
        for batch_idx, (data, target) in enumerate(train_data_loader):
            reduce = (target == i).nonzero()
            target_r = torch.index_select(target, 0, reduce.view(-1))
            data_r = torch.index_select(data, 0, reduce.view(-1))
            if len(target_r) >0:
                dataset.append((data_r,target_r))
        dataset = dataset[:int(N/args.get_num_classes())]
        proportions = np.random.dirichlet(np.repeat(beta, num_benign_workers))
        proportions = np.array([p * (len(idx_j) < N / num_benign_workers) for p, idx_j in zip(proportions, idx_batch)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(dataset)).astype(int)[:-1]
        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(dataset, proportions))]

    for j in range(len(idx_batch)):
        distributed_dataset[j] = idx_batch[j]

    for j in range(len(idx_batch),num_workers):
        distributed_dataset[j] = idx_batch[0]

    return distributed_dataset

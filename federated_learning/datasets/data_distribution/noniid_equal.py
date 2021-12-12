import torch
import random
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


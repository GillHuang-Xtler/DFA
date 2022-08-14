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
    distributed_dataset = [[] for i in range(int(num_workers * (1-args.get_mal_prop())))]

    class_slot = int(num_workers * (1-args.get_mal_prop())) / args.get_num_classes()
    if class_slot - int(class_slot) > 0:
        args.get_logger().info("unmatched class and client number")
    else:
        class_slot = int(class_slot)


    for i in range(args.get_num_classes()):
        for batch_idx, (data, target) in enumerate(train_data_loader):
            if args.reduce == 0.1:
                if batch_idx < 1000:
                    reduce = (target == i).nonzero()
                    target_r = torch.index_select(target, 0, reduce.view(-1))
                    data_r = torch.index_select(data, 0, reduce.view(-1))
                    rand_idx = random.randint(0, class_slot-1)
                    distributed_dataset[i*class_slot + rand_idx].append((data_r, target_r))
            else:
                reduce = (target == i).nonzero()
                target_r = torch.index_select(target, 0, reduce.view(-1))
                data_r = torch.index_select(data, 0, reduce.view(-1))
                rand_idx = random.randint(0, class_slot - 1)
                distributed_dataset[i * class_slot + rand_idx].append((data_r, target_r))

    for i in range(int(num_workers * (1-args.get_mal_prop())), num_workers):
        distributed_dataset.append(distributed_dataset[0])

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
    distributed_dataset = [[] for i in range(int(num_workers * (1-args.get_mal_prop())))]

    class_slot = int(num_workers * (1-args.get_mal_prop())) / args.get_num_classes()
    if class_slot - int(class_slot) > 0:
        args.get_logger().info("unmatched class and client number")
    else:
        class_slot = int(class_slot)

    for i in range(0, args.get_num_classes(), 2):
        for batch_idx, (data, target) in enumerate(train_data_loader):
            if args.reduce == 0.1:
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
            else:
                reduce_1 = (target == i).nonzero()
                target_1 = torch.index_select(target, 0, reduce_1.view(-1))
                data_1 = torch.index_select(data, 0, reduce_1.view(-1))
                reduce_2 = (target == i + 1).nonzero()
                target_2 = torch.index_select(target, 0, reduce_2.view(-1))
                data_2 = torch.index_select(data, 0, reduce_2.view(-1))
                rand_idx = random.randint(0, class_slot * 2 - 1)
                distributed_dataset[i * class_slot + rand_idx].append((data_1, target_1))
                distributed_dataset[i * class_slot + rand_idx].append((data_2, target_2))

    for i in range(int(num_workers * (1-args.get_mal_prop())), num_workers):
        distributed_dataset.append(distributed_dataset[0])

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


def distribute_batches_dirichlet(train_data_loader, num_workers, mal_prop, args, type=0):
    if type == 0:
        return distribute_batches_dirichlet_type0(train_data_loader, num_workers, mal_prop, args)
    elif type == 1 or type == 2:
        return distribute_batches_dirichlet_new(train_data_loader, num_workers, mal_prop, args, type)
    else:
        print("Type number error.")


def get_client_list(class_idx, num_workers, mal_prop, args, type):
    num_benign_workers = int(num_workers * (1 - mal_prop))
    if type == 1:
        idx_shift = int(num_benign_workers / args.get_num_classes())
        num_client_per_class = idx_shift * 2
        start_idx = class_idx * idx_shift
        end_idx = start_idx + num_client_per_class
        if end_idx > num_benign_workers:
            end_idx = num_benign_workers
        res = [i for i in range(start_idx, end_idx)]
    elif type == 2:
        idx_shift = int(num_benign_workers / args.get_num_classes())
        num_client_per_class = idx_shift * 2
        # res = random.sample(range(num_benign_workers), num_client_per_class)
        tmp = np.random.default_rng(seed=class_idx)
        res = tmp.choice(num_benign_workers, size=num_client_per_class)
        res = res.tolist()
    else:
        res = []
    return res


def distribute_batches_dirichlet_new(train_data_loader, num_workers, mal_prop, args, type):
    args.get_logger().info("Distribute data non-iid in Dirichlet distribution")
    args.get_logger().info("type: #{}", type)

    distributed_dataset = [[] for i in range(num_workers)]
    beta = args.get_beta()

    num_class = args.get_num_classes()

    # for benign users
    N = args.N
    # len(train_data_loader*args.get_batch_size()
    args.get_logger().info("total number for Dirichlet is #{}, with beta as #{}", N, args.get_beta())
    np.random.seed(20220114)
    # random.seed(20220114)

    dataset = None
    for i in range(num_class):
        dataset = []
        for batch_idx, (data, target) in enumerate(train_data_loader):
            reduce = (target == i).nonzero()
            target_r = torch.index_select(target, 0, reduce.view(-1))
            data_r = torch.index_select(data, 0, reduce.view(-1))
            if len(target_r) > 0:
                for j in range(len(target_r)):
                    dataset.append((data_r[j:j + 1], target_r[j:j + 1]))
        dataset = dataset[:int(N / args.get_num_classes())]
        client_list = get_client_list(i, num_workers, mal_prop, args, type)
        proportions = np.random.dirichlet(np.repeat(beta, len(client_list)))
        proportions = (np.cumsum(proportions) * len(dataset)).astype(int)[:-1]
        temp = np.split(dataset, proportions)
        for ii in range(len(temp)):
            distributed_dataset[client_list[ii]] += temp[ii].tolist()

    # sum = 0
    for j in range(len(distributed_dataset)):
        if len(distributed_dataset[j]) == 0:
            copy_number = torch.randint(0,79,(1,)).tolist()[0]
            distributed_dataset[j] = (distributed_dataset[copy_number])
            if len(distributed_dataset[j]) == 0:
                distributed_dataset[j] = distributed_dataset[j-1]
        # sum += len(distributed_dataset[j])
    # print("total num of samples!!!!!!!!:", sum)
    return distributed_dataset


def distribute_batches_dirichlet_type0(train_data_loader, num_workers, mal_prop, args):
    """
    Gives each worker the same number of batches of training data and one user only have two class except the malicious.

    :param train_data_loader: Training data loader
    :type train_data_loader: torch.utils.data.DataLoader
    :param num_workers: number of workers
    :type num_workers: int
    """

    args.get_logger().info("Distribute data non-iid in Dirichlet distribution")
    args.get_logger().info("type: 0")

    distributed_dataset = [[] for i in range(num_workers)]
    beta = args.get_beta()

    num_benign_workers = num_workers*(1-mal_prop)
    num_distribution_limit =  num_workers*(1-mal_prop) / args.get_num_classes()

    num_class = args.get_num_classes()

    # for benign users
    N = args.N
    # len(train_data_loader*args.get_batch_size()
    args.get_logger().info("total number for Dirichlet is #{}, with beta as #{}", N, args.get_beta())
    np.random.seed(2022)

    # while min_size < min_require_size:
    data_batch = []
    idx_batch = [[] for i in range(int(num_distribution_limit))]
    for i in range(num_class):
        dataset = []
        for batch_idx, (data, target) in enumerate(train_data_loader):
            reduce = (target == i).nonzero()
            target_r = torch.index_select(target, 0, reduce.view(-1))
            data_r = torch.index_select(data, 0, reduce.view(-1))
            if len(target_r) > 0:
                # print(len(target_r), type(data_r), data_r.shape, target_r.shape)
                for j in range(len(target_r)):
                    dataset.append((data_r[j:j+1], target_r[j:j+1]))
        dataset = dataset[:int(N/args.get_num_classes())]
        proportions = np.random.dirichlet(np.repeat(beta, num_distribution_limit))
        proportions = np.array([p * (len(idx_j) < N / num_distribution_limit) for p, idx_j in zip(proportions, idx_batch)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(dataset)).astype(int)[:-1]
        data_batch += [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(dataset, proportions))]
        # print(len(data_batch))
        # print(len(data_batch), len(idx_batch))


    for j in range(len(data_batch)):
        distributed_dataset[j] = data_batch[j]

    for j in range(len(data_batch),num_workers):
        distributed_dataset[j] = data_batch[len(data_batch)-1]

    for j in range(len(distributed_dataset)):
        if len(distributed_dataset[j]) == 0:
            distributed_dataset[j] = data_batch[len(data_batch)-1]

    return distributed_dataset

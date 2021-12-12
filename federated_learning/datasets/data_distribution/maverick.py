import torch
import random


def distribute_batches_bias(train_data_loader, num_workers):
    """
    Gives each worker the same number of batches of training data.

    :param train_data_loader: Training data loader
    :type train_data_loader: torch.utils.data.DataLoader
    :param num_workers: number of workers
    :type num_workers: int
    """
    distributed_dataset = [[] for i in range(num_workers+10)]
    bias_distributed_dataset = [[] for i in range(num_workers)]
    _tmp = []

    for batch_idx, (data, target) in enumerate(train_data_loader):
        worker_idx = batch_idx % (num_workers+10)

        distributed_dataset[worker_idx].append((data, target))

    for idx in range(num_workers-1):
        bias_distributed_dataset[idx] = distributed_dataset[idx]
    #
    for sum_idx in range(num_workers, num_workers+10):
        for j in range(len(distributed_dataset[sum_idx])):
            _tmp.append(tuple(distributed_dataset[sum_idx][j]))
    bias_distributed_dataset[num_workers - 1] = _tmp

    return bias_distributed_dataset

def distribute_batches_reduce_1(train_data_loader, num_workers):
    """
    Gives each worker the same number of batches of training data.

    :param train_data_loader: Training data loader
    :type train_data_loader: torch.utils.data.DataLoader
    :param num_workers: number of workers
    :type num_workers: int
    """
    distributed_dataset = [[] for i in range(num_workers)]

    for batch_idx, (data, target) in enumerate(train_data_loader):
        worker_idx = batch_idx % num_workers
        reduce = (target != 1).nonzero()
        try:
            target_r = torch.index_select(target, 0, reduce.view(-1))
            data_r = torch.index_select(data, 0, reduce.view(-1))
            distributed_dataset[worker_idx].append((data_r, target_r))
        except:
            distributed_dataset[worker_idx].append((data, target))


    return distributed_dataset

def distribute_batches_reduce_1_plus(train_data_loader, num_workers):
    """
    Gives each worker the same number of batches of training data.

    :param train_data_loader: Training data loader
    :type train_data_loader: torch.utils.data.DataLoader
    :param num_workers: number of workers
    :type num_workers: int
    """
    distributed_dataset = [[] for i in range(num_workers)]

    for batch_idx, (data, target) in enumerate(train_data_loader):
        worker_idx = batch_idx % num_workers
        reduce = (target != 1).nonzero()
        plus = (target == 1).nonzero()
        target_r = torch.index_select(target, 0, reduce.view(-1))
        data_r = torch.index_select(data, 0, reduce.view(-1))
        distributed_dataset[worker_idx].append((data_r, target_r))
        target_p = torch.index_select(target, 0, plus.view(-1))
        data_p = torch.index_select(data, 0, plus.view(-1))
        distributed_dataset[num_workers-1].append((data_p, target_p))

    return distributed_dataset

def distribute_batches_reduce_2_plusM(train_data_loader, num_workers):
    """
    Gives each worker the same number of batches of training data.

    :param train_data_loader: Training data loader
    :type train_data_loader: torch.utils.data.DataLoader
    :param num_workers: number of workers
    :type num_workers: int
    """
    distributed_dataset = [[] for i in range(num_workers)]

    for batch_idx, (data, target) in enumerate(train_data_loader):
        worker_idx = batch_idx % num_workers
        reduce = (target != 1).nonzero()
        plus = (target == 1).nonzero()
        target_r = torch.index_select(target, 0, reduce.view(-1))
        data_r = torch.index_select(data, 0, reduce.view(-1))
        distributed_dataset[worker_idx].append((data_r, target_r))
        target_p = torch.index_select(target, 0, plus.view(-1))
        data_p = torch.index_select(data, 0, plus.view(-1))
        distributed_dataset[num_workers-2].append((data_p[0:int(len(data_p)/1.1)], target_p[0:int(len(data_p)/1.1)]))
        distributed_dataset[num_workers-1].append((data_p[int(len(data_p)/1.1):], target_p[int(len(data_p)/1.1):]))

    return distributed_dataset

def distribute_batches_reduce_3_plusM(train_data_loader, num_workers):
    """
    Gives each worker the same number of batches of training data.

    :param train_data_loader: Training data loader
    :type train_data_loader: torch.utils.data.DataLoader
    :param num_workers: number of workers
    :type num_workers: int
    """
    distributed_dataset = [[] for i in range(num_workers)]

    for batch_idx, (data, target) in enumerate(train_data_loader):
        worker_idx = batch_idx % num_workers
        reduce = (target != 1).nonzero()
        plus = (target == 1).nonzero()
        target_r = torch.index_select(target, 0, reduce.view(-1))
        data_r = torch.index_select(data, 0, reduce.view(-1))
        distributed_dataset[worker_idx].append((data_r, target_r))
        target_p = torch.index_select(target, 0, plus.view(-1))
        data_p = torch.index_select(data, 0, plus.view(-1))
        rand_idx = random.randint(1, 3)
        distributed_dataset[num_workers - rand_idx].append((data_p, target_p))

    return distributed_dataset

def distribute_batches_reduce_1_only(train_data_loader, num_workers):
    """
    Gives each worker the same number of batches of training data.

    :param train_data_loader: Training data loader
    :type train_data_loader: torch.utils.data.DataLoader
    :param num_workers: number of workers
    :type num_workers: int
    """
    distributed_dataset = [[] for i in range(num_workers)]

    for batch_idx, (data, target) in enumerate(train_data_loader):
        worker_idx = batch_idx % (num_workers-1)
        reduce = (target != 1).nonzero()
        plus = (target == 1).nonzero()
        target_r = torch.index_select(target, 0, reduce.view(-1))
        data_r = torch.index_select(data, 0, reduce.view(-1))
        distributed_dataset[worker_idx].append((data_r, target_r))
        target_p = torch.index_select(target, 0, plus.view(-1))
        data_p = torch.index_select(data, 0, plus.view(-1))
        if len(plus)>0:
            distributed_dataset[num_workers-1].append((data_p, target_p))

    return distributed_dataset

def distribute_batches_reduce_2_plus(train_data_loader, num_workers):
    """
    Gives each worker the same number of batches of training data.

    :param train_data_loader: Training data loader
    :type train_data_loader: torch.utils.data.DataLoader
    :param num_workers: number of workers
    :type num_workers: int
    """
    distributed_dataset = [[] for i in range(num_workers)]

    for batch_idx, (data, target) in enumerate(train_data_loader):
        worker_idx = batch_idx % num_workers
        reduce0 = (target !=0 ).nonzero().cpu().detach().numpy().tolist()
        plus0 = (target == 0).nonzero()
        reduce1 = (target !=1 ).nonzero().cpu().detach().numpy().tolist()
        plus1 = (target == 1).nonzero()
        tmp = [val for val in reduce1 if val in reduce0]
        reduce = torch.LongTensor(tmp)
        target_r = torch.index_select(target, 0, reduce.view(-1))
        data_r = torch.index_select(data, 0, reduce.view(-1))
        distributed_dataset[worker_idx].append((data_r, target_r))
        target_p0 = torch.index_select(target, 0, plus0.view(-1))
        data_p0 = torch.index_select(data, 0, plus0.view(-1))
        target_p1 = torch.index_select(target, 0, plus1.view(-1))
        data_p1 = torch.index_select(data, 0, plus1.view(-1))
        distributed_dataset[num_workers-1].append((data_p0, target_p0))
        distributed_dataset[num_workers-2].append((data_p1, target_p1))

    return distributed_dataset

def distribute_batches_reduce_3_plus(train_data_loader, num_workers):
    """
    Gives each worker the same number of batches of training data.

    :param train_data_loader: Training data loader
    :type train_data_loader: torch.utils.data.DataLoader
    :param num_workers: number of workers
    :type num_workers: int
    """
    distributed_dataset = [[] for i in range(num_workers)]

    for batch_idx, (data, target) in enumerate(train_data_loader):
        worker_idx = batch_idx % num_workers
        reduce0 = (target !=0 ).nonzero().cpu().detach().numpy().tolist()
        plus0 = (target == 0).nonzero()
        plus1 = (target == 1).nonzero()
        reduce1 = (target !=1 ).nonzero().cpu().detach().numpy().tolist()
        reduce2 = (target !=2 ).nonzero().cpu().detach().numpy().tolist()
        plus2 = (target == 2).nonzero()
        _tmp = [val for val in reduce1 if val in reduce0]
        tmp = [val for val in _tmp if val in reduce2]
        reduce = torch.LongTensor(tmp)
        target_r = torch.index_select(target, 0, reduce.view(-1))
        data_r = torch.index_select(data, 0, reduce.view(-1))
        distributed_dataset[worker_idx].append((data_r, target_r))
        target_p0 = torch.index_select(target, 0, plus0.view(-1))
        data_p0 = torch.index_select(data, 0, plus0.view(-1))
        target_p1 = torch.index_select(target, 0, plus1.view(-1))
        data_p1 = torch.index_select(data, 0, plus1.view(-1))
        target_p2 = torch.index_select(target, 0, plus2.view(-1))
        data_p2 = torch.index_select(data, 0, plus2.view(-1))
        distributed_dataset[num_workers-1].append((data_p0, target_p0))
        distributed_dataset[num_workers-2].append((data_p1, target_p1))
        distributed_dataset[num_workers-3].append((data_p2, target_p2))

    return distributed_dataset
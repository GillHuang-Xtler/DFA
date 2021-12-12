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
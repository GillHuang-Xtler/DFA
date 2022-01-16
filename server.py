from loguru import logger
import torch
import time
from federated_learning.arguments import Arguments
from federated_learning.utils import generate_data_loaders_from_distributed_dataset
from federated_learning.datasets import FashionMNISTDataset
from federated_learning.datasets.data_distribution import distribute_batches_equally,distribute_batches_bias, distribute_batches_1_class, distribute_batches_2_class, distribute_batches_dirichlet
from federated_learning.datasets.data_distribution import distribute_batches_noniid_mal
from federated_learning.utils import average_nn_parameters, fed_average_nn_parameters
from federated_learning.utils.aggregation import krum_nn_parameters, multi_krum_nn_parameters, bulyan_nn_parameters, trmean_nn_parameters, median_nn_parameters, fgold_nn_parameters
from federated_learning.utils.attack import reverse_nn_parameters, ndss_nn_parameters, reverse_last_parameters, lie_nn_parameters, free_nn_parameters,free_last_nn_parameters, free_rand_nn_parameters, fang_nn_parameters
from federated_learning.utils import convert_distributed_data_into_numpy
from federated_learning.utils import poison_data
from federated_learning.utils import identify_random_elements, identify_random_elements_inc_49
from federated_learning.utils import save_results
from federated_learning.utils import load_train_data_loader, load_benign_data_loader, load_malicious_data_loader
from federated_learning.utils import load_test_data_loader
from federated_learning.utils import generate_experiment_ids
from federated_learning.utils import convert_results_to_csv
from client import Client
from federated_learning.nets import NetGenMnist, NetGenCifar, FashionMNISTCNNMAL, Cifar10CNNMAL
import math
import copy
import plot
import random
import numpy as np
from federated_learning.worker_selection.random import RandomSelectionStrategy



def train_subset_of_clients(epoch, args, clients, poisoned_workers):
    """
    Train a subset of clients per round.

    :param epoch: epoch
    :type epoch: int
    :param args: arguments
    :type args: Arguments
    :param clients: clients
    :type clients: list(Client)
    :param poisoned_workers: indices of poisoned workers
    :type poisoned_workers: list(int)
    """
    kwargs = args.get_round_worker_selection_strategy_kwargs()
    kwargs["current_epoch_number"] = epoch

    random_workers = args.get_round_worker_selection_strategy().select_round_workers(args,
        list(range(args.get_num_workers())),
        poisoned_workers,
        kwargs)

    previous_weight = []
    for client_idx in random_workers:
        args.get_logger().info("Training epoch #{} on client #{}", str(epoch),
                               str(clients[client_idx].get_client_index()))
        previous_weight = clients[0].get_nn_parameters()
        clients[client_idx].train(epoch)

    args.get_logger().info("Averaging client parameters")
    parameters = [clients[client_idx].get_nn_parameters() for client_idx in random_workers]
    dict_parameters = {client_idx: clients[client_idx].get_nn_parameters() for client_idx in random_workers}

    # attacks

    if args.get_attack_strategy() == "None":
        parameters = parameters
    elif args.get_attack_strategy() == "reverse":
        parameters = reverse_nn_parameters(parameters, previous_weight, args)
    elif args.get_attack_strategy() == "reverse_1":
        parameters = reverse_last_parameters(parameters, previous_weight, args)
    elif args.get_attack_strategy() == "ndss":
        dict_parameters = ndss_nn_parameters(dict_parameters, args)
    elif args.get_attack_strategy() == "lie":
        dict_parameters = lie_nn_parameters(dict_parameters, args)
    elif args.get_attack_strategy() == "fang":
        dict_parameters = fang_nn_parameters(dict_parameters, args)
    elif args.get_attack_strategy() == "freerider":
        dict_parameters = free_rand_nn_parameters(parameters, previous_weight, args)
        # dict_parameters = free_last_nn_parameters(parameters, previous_weight, args)

    # defenses

    new_nn_params = {}
    if args.get_aggregation_method() == "fedavg":
        parameters = {client_idx: clients[client_idx].get_nn_parameters() for client_idx in random_workers}
        sizes = {client_idx: clients[client_idx].get_client_datasize() for client_idx in random_workers}
        new_nn_params, selected_idx = fed_average_nn_parameters(parameters,sizes)
    elif args.get_aggregation_method() == "fedsgd":
        new_nn_params, selected_idx = average_nn_parameters(list(dict_parameters.values()))
    elif args.get_aggregation_method() == "krum":
        new_nn_params, selected_idx = krum_nn_parameters(dict_parameters, args)
    elif args.get_aggregation_method() == "mkrum":
        new_nn_params, selected_idx = multi_krum_nn_parameters(dict_parameters, previous_weight, args)
    elif args.get_aggregation_method() == "bulyan":
        new_nn_params, selected_idx = bulyan_nn_parameters(dict_parameters, args)
    elif args.get_aggregation_method() == "trmean":
        new_nn_params = trmean_nn_parameters(list(dict_parameters.values()), args)
    elif args.get_aggregation_method() == "median":
        new_nn_params = median_nn_parameters(list(dict_parameters.values()), args)
    elif args.get_aggregation_method() == "fgold":
        new_nn_params = fgold_nn_parameters(dict_parameters, args)


    for client in clients:
        args.get_logger().info("Updating parameters on client #{}", str(client.get_client_index()))
        client.update_nn_parameters(new_nn_params)

    all = 0
    select = 0

    if args.get_aggregation_method() in ["mkrum", "krum", "bulyan"]:

        for i in random_workers:
            if i > 80:
                all += 1
            if i > 80 and i in selected_idx:
                select += 1

    return clients[0].test(), random_workers, all, select


def create_clients(args, train_data_loaders, test_data_loader, distributed_train_dataset):
    """
    Create a set of clients.
    """
    clients = []
    if args.get_attack_strategy() == "cua" and (args.get_dataset() == "mnist" or args.get_dataset() == "fashion_mnist"):
        for idx in range(int(args.get_num_workers()*(1-args.get_mal_prop()))):
            clients.append(Client(args = args, client_idx = idx, is_mal= 'False', train_data_loader = train_data_loaders[idx], test_data_loader = test_data_loader, distributed_train_dataset = distributed_train_dataset[idx], gen_net = NetGenMnist(z_dim=args.n_dim)))
        for idx in range(int(args.get_num_workers()*(1-args.get_mal_prop())),int(args.get_num_workers())):
            if args.get_cua_syn_data_version() == "generator":
                gen_net = NetGenMnist(z_dim=args.n_dim)
            else:
                gen_net = FashionMNISTCNNMAL()
            clients.append(Client(args = args, client_idx = idx, is_mal= 'CUA', train_data_loader = train_data_loaders[idx], test_data_loader = test_data_loader, distributed_train_dataset = distributed_train_dataset[idx], gen_net = gen_net))
    if args.get_attack_strategy() == "cua" and (args.get_dataset() == "cifar_10" or args.get_dataset() == "cifar_100"):
        for idx in range(int(args.get_num_workers()*(1-args.get_mal_prop()))):
            clients.append(Client(args = args, client_idx = idx, is_mal= 'False', train_data_loader = train_data_loaders[idx], test_data_loader = test_data_loader, distributed_train_dataset = distributed_train_dataset[idx], gen_net = NetGenCifar(z_dim=args.n_dim)))
        for idx in range(int(args.get_num_workers()*(1-args.get_mal_prop())),int(args.get_num_workers())):
            if args.get_cua_syn_data_version() == "generator":
                gen_net = NetGenCifar(z_dim=args.n_dim)
            else:
                gen_net = Cifar10CNNMAL()
            clients.append(Client(args = args, client_idx = idx, is_mal= 'CUA', train_data_loader = train_data_loaders[idx], test_data_loader = test_data_loader, distributed_train_dataset = distributed_train_dataset[idx], gen_net = gen_net))
    else:
        for idx in range(int(args.get_num_workers())):
            clients.append(Client(args = args, client_idx = idx, is_mal= 'False', train_data_loader = train_data_loaders[idx], test_data_loader = test_data_loader, distributed_train_dataset = distributed_train_dataset[idx], gen_net = NetGenMnist(z_dim=args.n_dim)))
    return clients

def run_machine_learning(clients, args, poisoned_workers):
    """
    Complete machine learning over a series of clients.
    """
    epoch_test_set_results = []
    worker_selection = []
    all_worker_nums = []
    select_attacker_nums = []


    for epoch in range(1, args.get_num_epochs() + 1):
        results, workers_selected, all_worker_num, select_attacker_num = train_subset_of_clients(epoch, args, clients, poisoned_workers)
        epoch_test_set_results.append(results)
        worker_selection.append(workers_selected)
        all_worker_nums.append(all_worker_num)
        select_attacker_nums.append(select_attacker_num)

    return convert_results_to_csv(epoch_test_set_results), worker_selection, all_worker_nums, select_attacker_nums


def run_exp(replacement_method, num_poisoned_workers, KWARGS, client_selection_strategy, idx):
    log_files, results_files, models_folders, worker_selections_files = generate_experiment_ids(idx, 1)

    # Initialize logger
    handler = logger.add(log_files[0], enqueue=True)

    args = Arguments(logger)
    args.set_model_save_path(models_folders[0])
    args.set_num_poisoned_workers(num_poisoned_workers)
    args.set_round_worker_selection_strategy_kwargs(KWARGS)
    args.set_client_selection_strategy(client_selection_strategy)
    args.log()

    train_data_loader = load_train_data_loader(logger, args)
    test_data_loader = load_test_data_loader(logger, args)
    # train_data = FashionMNISTDataset(args).load_train_dataset()

    # Distribute batches

    # if args.get_distribution_method() == "bias":
    #     distributed_train_dataset = distribute_batches_bias(train_data_loader, args.get_num_workers())
    # elif args.get_distribution_method() == "iid":
    #     distributed_train_dataset = distribute_batches_equally(train_data_loader, args.get_num_workers())
    if args.get_distribution_method() == "noniid_1":
        distributed_train_dataset = distribute_batches_1_class(train_data_loader, args.get_num_workers(), args = args)
    elif args.get_distribution_method() == "noniid_2":
        distributed_train_dataset = distribute_batches_2_class(train_data_loader, args.get_num_workers(), args = args)
    elif args.get_distribution_method() == "noniid_dir_0":
        distributed_train_dataset = distribute_batches_dirichlet(train_data_loader, args.get_num_workers(), args.get_mal_prop(), args = args, type=0)
    elif args.get_distribution_method() == "noniid_dir_1":
        distributed_train_dataset = distribute_batches_dirichlet(train_data_loader, args.get_num_workers(), args.get_mal_prop(), args = args, type=1)
    elif args.get_distribution_method() == "noniid_dir_2":
        distributed_train_dataset = distribute_batches_dirichlet(train_data_loader, args.get_num_workers(), args.get_mal_prop(), args = args, type=2)

    # elif args.get_distribution_method() == "noniid_mal":
    #     distributed_train_dataset = distribute_batches_noniid_mal(benign_data_loader, malicious_data_loader, args.get_num_workers(), args = args)
    else:
        distributed_train_dataset = distribute_batches_equally(train_data_loader, args.get_num_workers())


    distributed_train_dataset = convert_distributed_data_into_numpy(distributed_train_dataset)

    poisoned_workers = identify_random_elements(args.get_num_workers(), args.get_num_poisoned_workers())
    distributed_train_dataset = poison_data(logger, distributed_train_dataset, args.get_num_workers(), poisoned_workers,
                                            replacement_method, args.get_poison_effort)

    train_data_loaders = generate_data_loaders_from_distributed_dataset(distributed_train_dataset,
                                                                        args.get_batch_size())

    clients = create_clients(args, train_data_loaders, test_data_loader, distributed_train_dataset)

    results, worker_selection, all_worker_nums, select_attacker_nums = run_machine_learning(clients, args, poisoned_workers)
    max = 0
    for i in results:
        if i[0]>max:
            max = i[0]
    print(max)
    print(sum(select_attacker_nums))
    print(sum(all_worker_nums))
    args.get_logger().info("random all attacker num is #{}, selected attacker num is #{}, best acc is #{} ", str(sum(all_worker_nums)), str(sum(select_attacker_nums)), str(max))

    save_results(results, args.get_dataset() + "_" + args.get_aggregation_method() + "_" +args.get_attack_strategy() + "_" +str(args.get_mal_prop()) + "_" + args.get_distribution_method() + "_" + str(args.get_beta())  + "_" + results_files[0] )
    save_results(worker_selection, args.get_dataset() + "_" + args.get_aggregation_method() + "_" +args.get_attack_strategy() + "_" +str(args.get_mal_prop()) + "_" + args.get_distribution_method() + "_" + str(args.get_beta()) + "_" + worker_selections_files[0])

    logger.remove(handler)

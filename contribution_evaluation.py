from federated_learning.utils import average_nn_parameters
import numpy as np
import math
import itertools
import os
import torch
from client import Client


def calculate_influence(args, clients, random_workers, epoch):
    """
    get influence of each client.
    :param clients: clients
    :type clients: list(Clients)
    """
    workers = []
    for client in clients:
        if client.get_client_index() in random_workers:
            workers.append(client)

    result_deletion = []

    args.get_logger().info("test result on epoch #{}", str(epoch))

    for client in workers:
        other_clients_idx = [worker_id for worker_id in random_workers if worker_id != client.get_client_index()]
        args.get_logger().info("Removing parameters on client #{}", str(client.get_client_index()))
        other_parameters = [clients[client_idx].get_nn_parameters() for client_idx in other_clients_idx]
        new_other_params = average_nn_parameters(other_parameters)
        client.update_nn_parameters(new_other_params)
        args.get_logger().info("Finished calculating Influence on client #{}", str(client.get_client_index()))
        result_deletion_accuracy, result_deletion_loss, result_deletion_class_precision, result_deletion_class_recall = client.test()
        result_deletion.append([result_deletion_accuracy, result_deletion_loss])

    return result_deletion


def make_all_subsets(n_client, random_workers):
    """
    get all combination of subset from a basic set to make reminder set for shapley
    :param n_client:
    :param random_workers:
    :return:
    """
    client_list = list(np.arange(n_client))
    set_of_all_subsets = set([])
    for i in range(len(client_list), -1, -1):
        for element in itertools.combinations(random_workers, i):
            set_of_all_subsets.add(frozenset(element))
    return sorted(set_of_all_subsets)


def calculate_shapley_values(args, clients, random_workers, epoch):
    """
    calculate shapley value
    :param args: args
    :param clients: clients in class Client
    :param random_workers: list of selected workers
    :param epoch: working round
    :return: list of shapley value of each client
    """
    result_deletion = []
    args.get_logger().info("Selected workers #{}", str(random_workers))
    args.get_logger().info("Start calculating Shapley result on epoch #{}", str(epoch))
    client_list = list(np.arange(len(random_workers)))
    shapley_acc = []
    shapley_loss = []
    client_shapley_acc, client_shapley_loss = 0, 0
    total = 0
    factorialTotal = math.factorial(len(random_workers))
    set_of_all_subsets = make_all_subsets(n_client=len(random_workers), random_workers=random_workers)
    for client_idx in random_workers:
        for subset in set_of_all_subsets:
            if client_idx in subset:
                remainderSet = subset.difference(set([client_idx]))
                b = len(remainderSet)
                fact_value = (len(client_list) - b - 1)
                other_parameters = [clients[client].get_nn_parameters() for client in subset]
                new_other_params = average_nn_parameters(other_parameters)
                clients[client_idx].update_nn_parameters(new_other_params)
                result_deletion_accuracy, result_deletion_loss, result_deletion_class_precision, result_deletion_class_recall = \
                    clients[client_idx].test()
                result_deletion.append([result_deletion_accuracy, result_deletion_loss])
                if len(remainderSet) > 0:
                    remainder_parameters = [clients[client].get_nn_parameters() for client in remainderSet]
                    new_remainder_params = average_nn_parameters(remainder_parameters)
                    clients[client_idx].update_nn_parameters(new_remainder_params)
                    remainder_accuracy, remainder_loss, remainder_precision, remainder_class_recall = clients[
                        client_idx].test()
                else:
                    remainder_accuracy, remainder_loss = 0, 0
                difference_acc = result_deletion_accuracy - remainder_accuracy
                difference_loss = result_deletion_loss - remainder_loss
                divisor = (math.factorial(fact_value) * math.factorial(b) * 1.0) / (factorialTotal * 1.0)
                weight_value_acc = divisor * difference_acc
                weight_value_loss = divisor * difference_loss
                client_shapley_acc += weight_value_acc
                client_shapley_loss += weight_value_loss
        shapley_acc.append(client_shapley_acc/100)
        shapley_loss.append(client_shapley_loss/100)
        # total = total + clientShapley
        args.get_logger().info("Finished calculating Shapley Value by acc: #{}, by loss: #{} on client #{}", str(client_shapley_acc/100), str(client_shapley_loss/100), str(client_idx))
        client_shapley_acc = 0
        client_shapley_loss = 0

    return shapley_acc, shapley_loss


def get_subset_index(subset):
    """
    get index from a subset
    :param subset: subset
    :type set(int)
    :return: index of subset
    :type str joined by '_'
    """
    subset_idx = '_'.join(sorted(set(str(i) for i in subset)))
    return subset_idx


def save_temporary_model(args, temp_save_dir, epoch, subset_idx, client):
    """
    Saves the model if necessary.
    """
    args.get_logger().debug("Saving model to flat file storage. Save #{}", str(subset_idx))

    # if not os.path.exists(args.get_save_model_folder_path()):
    #     os.mkdir(args.get_save_model_folder_path())

    full_save_path = os.path.join(temp_save_dir,
                                  "model_" + str(subset_idx) + "_" + str(epoch) + ".model")
    torch.save(client.get_nn_parameters(), full_save_path)


def load_model_from_file(args, client, model_file_path):
    """
    Load a model from a file.

    :param model_file_path: string
    """
    model_class = args.get_net()
    model = model_class()

    if os.path.exists(model_file_path):
        try:
            model.load_state_dict(torch.load(model_file_path))
            args.get_logger().info("Loading model: #{}", str(model_file_path))
        except:
            print("Couldn't load model. Attempting to map CUDA tensors to CPU to solve error.")

            model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
    else:
        print("Could not find model: {}".format(model_file_path))

    client.set_net(model)
    result_deletion_accuracy, result_deletion_loss, result_deletion_class_precision, result_deletion_class_recall = client.test()
    return result_deletion_accuracy, result_deletion_loss


def calculate_shapley_values_save_temp(args, clients, random_workers, epoch):
    """
    calculate shapley value and save temp models
    :param args: args
    :param clients: client in Client
    :param random_workers: list of selected workers
    :param epoch:
    :return:
    """
    result_deletion = []
    args.get_logger().info("Selected workers #{}", str(random_workers))
    args.get_logger().info("Start calculating Shapley result on epoch #{}", str(epoch))
    client_list = list(np.arange(len(random_workers)))
    shapley = []
    client_shapley = 0
    total = 0
    factorialTotal = math.factorial(len(random_workers))
    set_of_all_subsets = make_all_subsets(n_client=len(random_workers), random_workers=random_workers)
    temp_save_dir = './temp_' + str(epoch) + '_models'
    if not os.path.exists(temp_save_dir):
        os.mkdir(temp_save_dir)
    for client_idx in random_workers:
        for subset in set_of_all_subsets:
            if client_idx in subset:
                remainderSet = subset.difference(set([client_idx]))
                b = len(remainderSet)
                factValue = (len(client_list) - b - 1)
                temp_save_path = os.path.join(temp_save_dir, get_subset_index(subset=subset))
                if not os.path.exists(temp_save_path):
                    other_parameters = [clients[client].get_nn_parameters() for client in subset]
                    new_other_params = average_nn_parameters(other_parameters)
                    clients[client_idx].update_nn_parameters(new_other_params)
                    result_deletion_accuracy, result_deletion_loss, result_deletion_class_precision, result_deletion_class_recall = \
                        clients[client_idx].test()
                    result_deletion.append([result_deletion_accuracy, result_deletion_loss])
                    save_temporary_model(args=args, temp_save_dir=temp_save_dir, epoch=epoch,
                                         subset_idx=get_subset_index(subset), client=clients[client_idx])
                else:
                    result_deletion_accuracy, result_deletion_loss = load_model_from_file(args,
                                                                                          client=clients[client_idx],
                                                                                          model_file_path=temp_save_path)
                    result_deletion.append([result_deletion_accuracy, result_deletion_loss])

                remainder_load_path = os.path.join(temp_save_dir,
                                                   "model_" + str(get_subset_index(remainderSet)) + "_" + str(
                                                       epoch) + ".model")
                if len(remainderSet) > 0 and os.path.exists(remainder_load_path):
                    remainder_parameters = [clients[client].get_nn_parameters() for client in remainderSet]
                    new_remainder_params = average_nn_parameters(remainder_parameters)
                    clients[client_idx].update_nn_parameters(new_remainder_params)
                    remainder_accuracy, remainder_loss, remainder_precision, remainder_class_recall = clients[
                        client_idx].test()
                    remainder_accuracy, remainder_loss = load_model_from_file(args, client=clients[client_idx],
                                                                              model_file_path=remainder_load_path)

                elif len(remainderSet) > 0 and not os.path.exists(remainder_load_path):
                    remainder_parameters = [clients[client].get_nn_parameters() for client in remainderSet]
                    new_remainder_params = average_nn_parameters(remainder_parameters)
                    clients[client_idx].update_nn_parameters(new_remainder_params)
                    remainder_accuracy, remainder_loss, remainder_precision, remainder_class_recall = clients[
                        client_idx].test()

                else:
                    remainder_accuracy, remainder_loss = 0, 0
                difference = result_deletion_accuracy - remainder_accuracy
                divisor = (math.factorial(factValue) * math.factorial(b) * 1.0) / (factorialTotal * 1.0)
                weightValue = divisor * difference
                client_shapley += weightValue / 100
        shapley.append(client_shapley)
        # total = total + clientShapley
        args.get_logger().info("Finished calculating Shapley Value #{} on client #{}", str(clientShapley), str(client))
        client_shapley = 0

    return shapley

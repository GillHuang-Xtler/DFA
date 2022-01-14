import torch
import numpy as np
from federated_learning.arguments import Arguments

def multi_krum_nn_parameters(dict_parameters, previous_weight, args):
    """
    multi krum passed parameters.

    :param dict_parameters: nn model named parameters with client index
    :type parameters: list
    """

    args.get_logger().info("Averaging parameters on multi krum")
    multi_krum = 5
    candidate_num = 7
    distances = {}
    tmp_parameters = {}
    pre_distance = []
    for idx, parameter in dict_parameters.items():
        distance = []
        for _idx, _parameter in dict_parameters.items():

            dis = 0
            for key in parameter.keys():
                dis = dis + (torch.norm(_parameter[key].float() - parameter[key].float()) ** 2)
            distance.append(dis)

            # pre_distance.append(sum(pre_dis))
            tmp_parameters[idx] = parameter
        # pre_dis = [torch.norm((_parameter[name].data - previous_weight[name].data).float()) for name in parameter.keys()]
        # pre_distance.append(sum(pre_dis))
        distance.sort()
        args.get_logger().info("Distance #{}", str(distance))
        distances[idx] = sum(distance[:candidate_num+1])

    args.get_logger().info("Distances #{} on client #{}", str(distances.values()),
                           str(distances.keys()))
    sorted_distance = dict(sorted(distances.items(), key=lambda item: item[1]))
    candidate_parameters = [tmp_parameters[idx] for idx in sorted_distance.keys()][:multi_krum]
    args.get_logger().info("Averaging parameters on client #{}", str(list(sorted_distance.keys())[:multi_krum]))
    new_params = {}
    for name in candidate_parameters[0].keys():
        new_params[name] = sum([param[name].data for param in candidate_parameters]) / len(candidate_parameters)

    return new_params, list(sorted_distance.keys())[:multi_krum]


def krum_nn_parameters(dict_parameters, args):
    """
    krum passed parameters.

    :param dict_parameters: nn model named parameters with client index
    :type parameters: list
    """

    args.get_logger().info("Averaging parameters on krum")

    candidate_num = 6
    distances = {}
    tmp_parameters = {}
    for idx, parameter in dict_parameters.items():
        distance = []
        for _idx, _parameter in dict_parameters.items():
            dis = [torch.norm((parameter[name].data - _parameter[name].data).float())**2 for name in parameter.keys()]
            distance.append(sum(dis))
            tmp_parameters[idx] = parameter
        # distance = sum(torch.Tensor(distance).float())
        distance.sort()
        args.get_logger().info("Distances #{}", str(distance))
        # print("benign distance: " + str(distance))
        distances[idx] = sum(distance[:candidate_num])
    args.get_logger().info("Distances #{} on client #{}", str(distances.values()),
                           str(distances.keys()))
    sorted_distance = dict(sorted(distances.items(), key=lambda item: item[1]))
    candidate_parameters = [tmp_parameters[idx] for idx in sorted_distance.keys()][:1]
    args.get_logger().info("Averaging parameters on client #{}", str(list(sorted_distance.keys())[:1]))

    new_params = {}
    for name in candidate_parameters[0].keys():
        new_params[name] = sum([param[name].data for param in candidate_parameters]) / len(candidate_parameters)

    return new_params, list(sorted_distance.keys())[:1]

def bulyan_nn_parameters(dict_parameters, args):
    """
    bulyan passed parameters.

    :param dict_parameters: nn model named parameters with client index
    :type parameters: list
    """
    args.get_logger().info("Averaging parameters on bulyan")
    multi_krum = 5
    candidate_num = 7
    distances = {}
    tmp_parameters = {}
    for idx, parameter in dict_parameters.items():
        distance = []
        for _idx, _parameter in dict_parameters.items():
            dis = [torch.norm((parameter[name].data - _parameter[name].data).float())**2 for name in parameter.keys()]
            distance.append(sum(dis))
            tmp_parameters[idx] = parameter
        # distance = sum(torch.Tensor(distance).float())
        distance.sort()
        distances[idx] = sum(distance[:candidate_num])
    args.get_logger().info("Distances #{} on client #{}", str(distances.values()),
                           str(distances.keys()))
    sorted_distance = dict(sorted(distances.items(), key=lambda item: item[1]))
    candidate_parameters = [tmp_parameters[idx] for idx in sorted_distance.keys()][1:multi_krum-1]
    args.get_logger().info("Averaging parameters on client #{}", str(list(sorted_distance.keys())[1:multi_krum-1]))
    new_params = {}
    for name in candidate_parameters[0].keys():
        new_params[name] = sum([param[name].data for param in candidate_parameters]) / len(candidate_parameters)

    return new_params, list(sorted_distance.keys())[1:multi_krum-1]

def trmean_nn_parameters(parameters, args):
    """
    Trimmed mean of passed parameters.

    :param parameters: nn model named parameters
    :type parameters: list
    """
    args.get_logger().info("Averaging parameters on trimmed mean")

    new_params = {}
    for name in parameters[0].keys():
        tmp = []
        for param in parameters:
            tmp.append(param[name].data.long())
        # print(name)
        # print(tmp)
        max_data = torch.max(torch.stack(tmp), 0)[0]
        min_data = torch.min(torch.stack(tmp), 0)[0]
        # print(type(min_data))
        new_params[name] = sum([param[name].data for param in parameters]).float()
        new_params[name] -= ((max_data+min_data).float())
        new_params[name] /= (len(parameters)-2)

    return new_params


def median_nn_parameters(parameters, args):
    """
    median of passed parameters.

    :param parameters: nn model named parameters
    :type parameters: list
    """
    args.get_logger().info("Averaging parameters on median")

    new_params = {}
    for name in parameters[0].keys():
        tmp = []
        for param in parameters:
            # print(param[name].data.float().shape)
            # print(name)
            tmp.append(param[name].data.float())
            # print(param[name].data.float().shape)
        median_data = torch.median(torch.stack(tmp), 0)[0]
        new_params[name] = median_data

    return new_params

def fgold_nn_parameters(dict_parameters, args):
    """
    median of passed parameters.

    :param parameters: nn model named parameters
    :type parameters: list
    """
    args.get_logger().info("Averaging parameters on fools gold")
    distances = {}
    tmp_parameters = {}
    candidate_num = args.get_num_workers()/10 - args.get_num_poisoned_workers() +1

    for idx, parameter in dict_parameters.items():
        distance = []
        for _idx, _parameter in dict_parameters.items():
            dis = [torch.norm((parameter[name].data - _parameter[name].data).float())**2 for name in parameter.keys()]
            # if sum(dis) < args.get_similarity_epsilon() and sum(dis) != 0 :
            #     distance.append(10000)
            #     args.get_logger().info("small distance as #{}", str(sum(dis)))
            distance.append(sum(dis))
            tmp_parameters[idx] = parameter
        # distance = sum(torch.Tensor(distance).float())
        distance.sort()
        distances[idx] = sum(distance)
    args.get_logger().info("Distances #{} on client #{}", str(distances.values()),
                           str(distances.keys()))
    sorted_distance = dict(sorted(distances.items(), key=lambda item: item[1]))
    # sorted_distance = dict((k, v) for k, v in sorted_distance.items() if v >= 10000)
    candidate_parameters = [tmp_parameters[idx] for idx in sorted_distance.keys()][1:]
    args.get_logger().info("Averaging parameters on client #{}", str(list(sorted_distance.keys())[1:]))

    new_params = {}
    for name in candidate_parameters[0].keys():
        tmp = []
        for param in candidate_parameters:
            tmp.append(param[name].data)
        median_data = torch.median(torch.stack(tmp), 0)[0]
        new_params[name] = median_data

    return new_params


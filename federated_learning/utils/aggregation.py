import torch
import numpy as np
from federated_learning.arguments import Arguments
import copy

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


def dfad_check(args, net, dataloader):

    if torch.cuda.is_available() and args.get_cuda():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    args.get_logger().info("dfad defense check")
    net.eval()
    correct = 0
    total = 0
    targets_ = []
    pred_ = []
    beta = args.get_defense_beta()

    conf_value = 0.0

    with torch.no_grad():
        for (images, labels) in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            confidence, predicted = torch.max(outputs.data, 1)

            conf_value += confidence

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            targets_.extend(labels.cpu().view_as(predicted).numpy())
            pred_.extend(predicted.cpu().numpy())

    # accuracy = 100 * correct / total

    conf_value = torch.mean(conf_value / total)

    class_bal_list = np.array([pred_.count(c) for c in range(0,10)])
    bal_value = 1.0/np.std(class_bal_list)

    # args.get_logger().debug('Test local: Accuracy: {}/{} ({:.0f}%)'.format(correct, total, accuracy))

    args.get_logger().info("balance value:\n" + str(bal_value)) # + "balance list:\n" + str(class_bal_list) + ))
    args.get_logger().info("confidence value:\n" + str(conf_value))

    d_score = (1 + beta**2) * (bal_value * conf_value) / ( beta**2*bal_value + conf_value)
    return d_score


def dfad_nn_parameters(dict_parameters, check_net, check_data_loader, args):
    check_list = []
    for idx, parameter in dict_parameters.items():
        check_net.load_state_dict(copy.deepcopy(parameter), strict=True)
        d_score = dfad_check(args, check_net, check_data_loader)
        check_list.append([idx, d_score])
    check_list.sort(key=lambda t: t[1])  # sort according to the second element d_score, small -> big

    args.get_logger().debug("check list:\n" + str(check_list))

    new_params = {}
    selected_idx = []
    remove_num = 2  # the number of clients to remove
    for c in range(remove_num, len(check_list)):
        idx = check_list[c][0]
        selected_idx.append(idx)

    new_params = {}
    net_param_name_list = dict_parameters[selected_idx[0]].keys()
    for name in net_param_name_list:
        new_params[name] = sum([dict_parameters[idx][name].data for idx in selected_idx]) / len(selected_idx)

    args.get_logger().debug("selected idx:\n" + str(selected_idx))

    return new_params, selected_idx
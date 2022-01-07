import torch
import numpy as np

def reverse_nn_parameters(parameters, previous_weight, args):
    """
    generate reverse all layers parameters.

    :param parameters: nn model named parameters
    :type parameters: list
    """

    args.get_logger().info("Reverse all layers of gradients from attackers")
    new_parameters = []
    for params in parameters[:len(parameters)-args.get_num_attackers()]:
        new_parameters.append(params)

    for params in parameters[len(parameters)-args.get_num_attackers():]:
        for name in parameters[0].keys():
            params[name] = (2*previous_weight[name].data - params[name].data)
            # params[name] = - params[name].data
        new_parameters.append(params)

    return new_parameters

def reverse_last_parameters(parameters, previous_weight, args):
    """
    generate reverse last layers parameters.

    :param parameters: nn model named parameters
    :type parameters: list
    """
    args.get_logger().info("Reverse last layers of gradients from attackers")
    layers = list(parameters[0].keys())
    new_parameters = []
    for params in parameters[:len(parameters)-args.get_num_attackers()]:
        new_parameters.append(params)

    for params in parameters[len(parameters)-args.get_num_attackers():]:
        for name in parameters[0].keys():
            if name in layers[-(args.get_num_reverse_layers()):]:
                params[name] = (2*previous_weight[name].data - params[name].data)
            else:
                params[name] = params[name]
        new_parameters.append(params)

    return new_parameters

def lie_nn_parameters(dict_parameters, args):
    """
    generate lie parameters.

    :param parameters: nn model named parameters
    :type parameters: list
    """

    z_value = args.get_lie_z_value()
    mean_params = {}
    std_params = {}
    for name in dict_parameters[list(dict_parameters.keys())[0]].keys():
        mean_params[name] = sum([param[name].data for param in dict_parameters.values()])/len(dict_parameters.values())
        _std_params = []
        for param in dict_parameters.values():
            _std_params.append(param[name].data)
        val = torch.stack(_std_params)
        std_params[name] = torch.std(val.float())

    args.get_logger().info("Averaging parameters on model lie attackers")
    new_parameters = {}
    for client_idx in dict_parameters.keys():
        if client_idx < args.get_num_workers()*(1-args.get_mal_prop()):
            new_parameters[client_idx] = dict_parameters[client_idx]
        else:
            mal_param = {}
            for name in dict_parameters[list(dict_parameters.keys())[0]].keys():
                mal_param[name] = mean_params[name].data + z_value * std_params[name].data
            new_parameters[client_idx] = mal_param

    # mal_param = {}
    # for name in dict_parameters[list(dict_parameters.keys())[0]].keys():
    #     mal_param[name] = mean_params[name] + z_value * std_params[name]
    #
    # # [new_parameters[i+int(args.get_num_workers()*(1-args.get_mal_prop()))]] = mal_param for i in range(int(args.get_num_workers()*args.get_mal_prop()))]
    # for i in range(int(args.get_num_workers()*args.get_mal_prop())):
    #     new_parameters[ i+ int(args.get_num_workers()*0.1 * (1 - args.get_mal_prop()))] = mal_param


    return new_parameters

### Multi-krum and Bulyan
def fang_nn_parameters(dict_parameters, args):
    """
    generate fang parameters.

    :param parameters: nn model named parameters
    :type parameters: list
    """

    args.get_logger().info("Averaging parameters on model fang attackers")
    new_parameters = dict_parameters


    return new_parameters


def model_distance(m1_params, m2_params):
    """
    for ndss_nn_parameters
    """
    distance = 0
    for key in m1_params.keys():
        # print(m1_params[key].type())
        distance = distance + (torch.norm(m1_params[key].float()-m2_params[key].float())**2)
    return distance


def get_deviation_and_model_avg(dict_parameters, normal_idx_list):
    """
    for ndss_nn_parameters
    """
    model_avg = {}
    for normal_idx in normal_idx_list:
        client_param = dict_parameters[normal_idx]  # normal client
        for key in client_param.keys():
            if key in model_avg:
                model_avg[key] = model_avg[key].data + client_param[key].data
            else:
                model_avg[key] = client_param[key]
    deviation = {}

    for key in model_avg.keys():
        model_avg[key] = model_avg[key].data / len(normal_idx_list)
        deviation[key] = torch.sign(model_avg[key])
    return model_avg, deviation


def get_malicious_model(model_avg, lamda, deviation):
    """
    for ndss_nn_parameters
    """
    mal_model = {}
    for key in model_avg.keys():
        mal_model[key] = model_avg[key].data - (lamda * deviation[key].data)
    return mal_model


def oracle_check(mal_model, dict_parameters, normal_idx_list, upper_bound):
    """
    for ndss_nn_parameters
    """
    for c in normal_idx_list:
        mal_dis = model_distance(mal_model, dict_parameters[c])
        if mal_dis > upper_bound:
            return False
        # else:
        #     print("mal_dis", mal_dis, "upper_bound", upper_bound)
    return True


def ndss_nn_parameters(dict_parameters, args):
    """
    The implementation of "Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses
    for Federated Learning", (AGR-agnostic attacks, Min-Max), according the authors' open-sourced code:
    https://github.com/vrt1shjwlkr/NDSS21-Model-Poisoning/blob/main/cifar10/release-fedsgd-alexnet-mkrum-unknown-benign-gradients.ipynb

    :param parameters: nn model named parameters
    :type parameters: list
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.get_logger().info("Doing NDSS attackers (using in server.py, writing in attack.py)")

    count_attacker = 0
    normal_idx_list = []
    for client_idx in dict_parameters.keys():
        if client_idx >= args.get_num_workers() * (1 - args.get_mal_prop()):  # attacker
            count_attacker += 1
        else:
            normal_idx_list.append(client_idx)
    if count_attacker == 0:
        return dict_parameters

    print("normal clients: ", len(normal_idx_list), ", attackers: ", count_attacker)

    # calculate the upper bound of gradient distance
    upper_bound = 0
    for i in range(len(normal_idx_list)-1):
        for j in range(i+1, len(normal_idx_list)):
            dis = model_distance(dict_parameters[normal_idx_list[i]], dict_parameters[normal_idx_list[j]])
            if dis > upper_bound:
                upper_bound = dis
    print("upper bound is:", upper_bound)

    model_avg, deviation = get_deviation_and_model_avg(dict_parameters, normal_idx_list)

    for client_idx in dict_parameters.keys():
        if client_idx >= args.get_num_workers() * (1 - args.get_mal_prop()):
            # building malicious model parameters for the attacker
            lamda = torch.Tensor([10.0]).float().to(device)
            threshold_diff = 1e-5
            lamda_fail = lamda
            lamda_succ = 0

            while torch.abs(lamda_succ - lamda) > threshold_diff:
                mal_model = get_malicious_model(model_avg, lamda, deviation)
                # print('lamda is ', lamda)
                if oracle_check(mal_model, dict_parameters, normal_idx_list, upper_bound):
                    # print('successful lamda is ', lamda)

                    lamda_succ = lamda
                    lamda = lamda + lamda_fail / 2
                else:
                    lamda = lamda - lamda_fail / 2

                lamda_fail = lamda_fail / 2

            mal_model = get_malicious_model(model_avg, lamda_succ, deviation)
            dict_parameters[client_idx] = mal_model

    return dict_parameters

def free_nn_parameters(parameters, previous_weight, args):
    """
    generate reverse all layers parameters.

    :param parameters: nn model named parameters
    :type parameters: list
    """

    args.get_logger().info("Data Free Untargeted Attack")
    new_parameters = []
    for params in parameters[:len(parameters)-args.get_num_attackers()]:
        new_parameters.append(params)

    tmp = {}
    for name in previous_weight.keys():
        tmp[name] = previous_weight[name].data

    for i in range(args.get_num_attackers()):
        new_parameters.append(tmp)
    args.get_logger().info("the last 2 client do not have any data for training")
    dict_parameters = {client_idx: new_parameters[client_idx] for client_idx in range(len(parameters))}
    return dict_parameters

def free_rand_nn_parameters(parameters, previous_weight, args):
    """
    generate reverse all layers parameters.

    :param parameters: nn model named parameters
    :type parameters: list
    """

    args.get_logger().info("Data Free Untargeted Attack")
    new_parameters = []
    for params in parameters[:len(parameters)-args.get_num_attackers()]:
        new_parameters.append(params)

    tmp = {}
    for name in previous_weight.keys():
        max_value = torch.max(previous_weight[name].data)
        min_value = torch.min(previous_weight[name].data)
        # print(previous_weight[name].data.size())
        tmp[name] = (torch.rand(previous_weight[name].data.size()))-0.5
        # tmp[name] = (torch.rand(previous_weight[name].data.size()))*(max_value-min_value) + min_value

    for i in range(args.get_num_attackers()):
        new_parameters.append(tmp)
    args.get_logger().info("the last 2 client do not have any data for training and add random factor")
    dict_parameters = {client_idx: new_parameters[client_idx] for client_idx in range(len(parameters))}
    return dict_parameters

def free_last_nn_parameters(parameters, previous_weight, args):
    """
    generate reverse all layers parameters.

    :param parameters: nn model named parameters
    :type parameters: list
    """

    args.get_logger().info("Data Free last layer Untargeted Attack")
    layers = list(parameters[0].keys())
    new_parameters = []
    for params in parameters[:len(parameters)-args.get_num_attackers()]:
        new_parameters.append(params)

    for params in parameters[len(parameters)-args.get_num_attackers():]:
        for name in parameters[0].keys():
            if name in layers[-(args.get_num_reverse_layers()):]:
                params[name] = (torch.rand(previous_weight[name].data.size()))-0.5
            else:
                params[name] = params[name]
        new_parameters.append(params)

    dict_parameters = {client_idx: new_parameters[client_idx] for client_idx in range(len(parameters))}
    return dict_parameters

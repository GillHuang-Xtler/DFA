def reduce_1(targets, target_set):
    """
    :param targets: Target class IDs
    :type targets: list
    :param target_set: Set of class IDs possible
    :type target_set: list
    :return: new class IDs
    """
    for idx in range(len(targets)):
        if targets[idx] == 1:
            targets[idx] = 9

    return targets
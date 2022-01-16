from .selection_strategy import SelectionStrategy
import random
import math
from scipy.stats import wasserstein_distance
import numpy as np
import heapq
import torch
from kmeans_pytorch import kmeans
from federated_learning.utils.tensor_converter import convert_distributed_data_into_numpy

class RandomSelectionStrategy(SelectionStrategy):
    """
    Randomly selects workers out of the list of all workers
    """

    def select_round_workers(self, args, workers, poisoned_workers, kwargs):
        if args.get_attack_strategy() == "none":
            return random.sample(workers[:int(len(workers)* (1-args.get_mal_prop()))], kwargs["NUM_WORKERS_PER_ROUND"])
        else:
            return random.sample(workers, kwargs["NUM_WORKERS_PER_ROUND"])

    def select_round_workers_minus_1(self, workers, poisoned_workers, kwargs):
        return random.sample(workers[:90], kwargs["NUM_WORKERS_PER_ROUND"]-1)

    def select_round_workers_minus_2(self, workers, poisoned_workers, kwargs):
        return random.sample(workers[:80], kwargs["NUM_WORKERS_PER_ROUND"]-2)

    def select_round_workers_except_49(self, workers, poisoned_workers, kwargs):
        workers.remove(49)
        return random.sample(workers, kwargs["NUM_WORKERS_PER_ROUND"]-1)

    def compute_wasserstein_distance(self, distribution1, distribution2):
        return wasserstein_distance(distribution1,distribution2)

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def S(self, weight):
        R = random.random()
        return math.pow(R, 1 / weight)

    def norm(self, dis):
        a = dis[9] / 100
        return [i / (100 * a) for i in dis]

    def a_Reservoir(self, samples, m):
        """
        :samples: [(item, weight), ...]
        :k: number of selected items
        :returns: [(item, weight), ...]
        """

        heap = []
        for sample in samples:
            wi = sample
            # ui = random.uniform(0, 1)
            ui = np.random.rand(1)
            ki = ui ** (1 / wi)

            if len(heap) < m:
                heapq.heappush(heap, (ki, sample))
            elif ki > heap[0][0]:
                heapq.heappush(heap, (ki, sample))

                if len(heap) > m:
                    heapq.heappop(heap)

        return [samples.index(item[1]) for item in heap]


    def compute_probability1(self, global_distribution, current_distribution, client_distribution):
        alpha = 0.1
        EMDG = []
        for i in client_distribution:
            EMDG.append(self.compute_wasserstein_distance(global_distribution, i))
        EMDG = [(i - min(EMDG)) / (max(EMDG) - min(EMDG)) for i in EMDG]

        #
        EMDC = []
        for i in client_distribution:
            EMDC.append(self.compute_wasserstein_distance([m for m in current_distribution], [j for j in i]))
        EMDC = [(i - min(EMDC)) / (max(EMDC) - min(EMDC)) for i in EMDC]
        EMDC = [i / 1 for i in EMDC]

        _emd = []

        for i in range(len(client_distribution)):
            _emd.append((-alpha * EMDG[i] + EMDC[i]))

        return self.softmax(_emd)

    def compute_probability(self, global_distribution, current_distribution, client_distribution, epoch):
        alpha = 1
        EMDG = []
        for i in client_distribution:
            EMDG.append(self.compute_wasserstein_distance(global_distribution, i))
        # EMDG = [(i - min(EMDG)) / (max(EMDG) - min(EMDG)) for i in EMDG]

        #
        EMDC = []
        for i in client_distribution:
            EMDC.append(self.compute_wasserstein_distance([m for m in current_distribution], [j for j in i]))
        # EMDC = [(i - min(EMDC)) / (max(EMDC) - min(EMDC)) for i in EMDC]
        # EMDC = [i / 1 for i in EMDC]

        # print(EMDG)
        _emd = []
        # print(EMDC)
        for i in range(len(client_distribution)):
            _emd.append((0.11 * EMDG[i] - 0.001 * epoch * EMDC[i]))
            # _emd.append(EMDC[i]/10)

        return self.softmax(_emd)

    def select_round_workers_distribution(self, workers, poisoned_workers,clients, current_distribution, kwargs,epoch):
        client_distribution = []
        global_distribution=np.ones(10)
        for client_idx in range(len(clients)):
            _client_distribution = clients[client_idx].get_client_distribution()
            client_distribution.append(_client_distribution)
        client_distribution = [self.norm(i) for i in client_distribution]

        probability = self.compute_probability(global_distribution, current_distribution, client_distribution, epoch)
        print("probability"+ str(probability))

        num_round_workers  = kwargs["NUM_WORKERS_PER_ROUND"]
        choosed_workers = self.a_Reservoir(probability.tolist(), num_round_workers)
        return choosed_workers

    def select_round_workers_sv(self, workers, poisoned_workers,clients, current_probability, kwargs):

        num_round_workers  = kwargs["NUM_WORKERS_PER_ROUND"]
        choosed_workers = self.a_Reservoir(current_probability.tolist(), num_round_workers)
        return choosed_workers

    def select_round_workers_actvSAMP(self, workers, poisoned_workers,clients, kwargs):
        clients_distribution = []
        for client_idx in range(len(clients)):
            _client_distribution = clients[client_idx].get_client_distribution()
            clients_distribution.append(_client_distribution)
        data_size, dims, num_clusters = len(clients), len(clients_distribution[0]), kwargs["NUM_WORKERS_PER_ROUND"]
        clients_distribution = torch.from_numpy(np.array(clients_distribution))
        cluster_ids_x, cluster_centers = kmeans(
        X = clients_distribution, num_clusters = num_clusters, distance = 'euclidean')

        clusters = []
        for cluster_name in range(num_clusters):
            _cluster = []
            _cluster.extend([i for i in range(len(cluster_ids_x)) if cluster_ids_x[i] == cluster_name])
            clusters.append(_cluster)

        choosed_workers = []
        for cluster in clusters:
            choosed_workers.append((random.sample(cluster,1))[0])
        return choosed_workers

    def select_round_workers_TiFL(self, workers, poisoned_workers,clients, accs,  kwargs):

        probability = [-i for i in accs]
        print (probability)

        num_round_workers  = kwargs["NUM_WORKERS_PER_ROUND"]
        choosed_workers = self.a_Reservoir(probability, num_round_workers)
        print(choosed_workers)
        return choosed_workers







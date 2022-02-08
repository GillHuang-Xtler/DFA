from .nets import Cifar10CNN
from .nets import FashionMNISTCNN
from .nets import Cifar100ResNet
from .nets import FashionMNISTResNet
from .nets import Cifar10ResNet
from .nets import Cifar100VGG
from .nets import MNISTCNN
from .worker_selection import BeforeBreakpoint
from .worker_selection import AfterBreakpoint
from .worker_selection import PoisonerProbability
import torch.nn.functional as F
import torch
import json

# Setting the seed for Torch
SEED = 1
torch.manual_seed(SEED)

class Arguments:

    def __init__(self, logger):
        self.logger = logger

        self.dataset = "fashion_mnist"  # "cifar_10" "fashion_mnist"
        self.batch_size = 10
        self.test_batch_size = 1000
        self.epochs = 100
        self.cuda = False
        self.shuffle = False
        self.log_interval = 10
        self.kwargs = {}

        self.round_worker_selection_strategy = None
        self.round_worker_selection_strategy_kwargs = None

        self.save_model = False
        self.save_temp_model = False
        self.save_epoch_interval = 1
        self.save_model_path = "models"
        self.epoch_save_start_suffix = "start"
        self.epoch_save_end_suffix = "end"
        self.get_poison_effort = 'full'
        self.num_workers = 100
        self.aggregation = "mkrum"  # trmean, bulyan, mkrum, fedsgd, median
        self.attack = "cua"  # cua, ndss, lie, fang, none
        self.ndss_deviation_type = "sign"  # std, sign

        self.cua_version = "target_class"  # target_class, infer_class
        self.cua_syn_data_version = "generator"  # generator, layer
        self.use_real_data = True
        self.static = False

        self.dev_type = 'sign'
        self.mal_prop = 0.2
        self.num_reverse_layers = 3
        # self.num_poisoned_workers = 10
        self.lie_z_value = 0.2
        self.n_dim = 128
        # self.lie_z_value = {1:0.68947, 2:0.68947, 3:0.69847, 5:0.7054, 8:0.71904,10:0.72575, 12:0.73891}

        self.beta = 0.5
        self.distribution_method = "noniid_dir_2"

        self.num_classes = 10


        if self.dataset == "cifar_10":
            self.net = Cifar10CNN
            # self.net = Cifar10ResNet

            self.lr = 0.01
            self.momentum = 0.5
            self.scheduler_step_size = 50
            self.scheduler_gamma = 0.5
            self.min_lr = 1e-10
            self.N = 50000
            self.generator_image_num = 50
            self.generator_local_epoch = 10
            self.layer_image_num = 50
            self.layer_image_epoch = 10
            self.reduce = 1

            self.train_data_loader_pickle_path = "data_loaders/cifar10/train_data_loader.pickle"
            self.test_data_loader_pickle_path = "data_loaders/cifar10/test_data_loader.pickle"


        elif self.dataset == "fashion_mnist":
            self.net = FashionMNISTCNN
            # self.net = FashionMNISTResNet

            self.lr = 0.001
            self.momentum = 0.9
            self.scheduler_step_size = 10
            self.scheduler_gamma = 0.1
            self.min_lr = 1e-10
            self.N = 5000
            self.generator_image_num = 50
            self.generator_local_epoch = 5
            self.layer_image_num = 50
            self.layer_image_epoch = 5
            self.reduce = 0.1

            self.train_data_loader_pickle_path = "data_loaders/fashion-mnist/train_data_loader.pickle"
            self.test_data_loader_pickle_path = "data_loaders/fashion-mnist/test_data_loader.pickle"
            self.benign_data_loader_pickle_path = "data_loaders/fashion-mnist/benign_data_loader.pickle"
            self.malicious_data_loader_pickle_path = "data_loaders/fashion-mnist/malicious_data_loader.pickle"


        else:
            print("Incorrect dataset information, please check.")

        self.loss_function = torch.nn.CrossEntropyLoss

        self.default_model_folder_path = "default_models"

        self.data_path = "data"

    def get_dataset(self):
        return self.dataset

    def get_ndss_deviation_type(self):
        return self.ndss_deviation_type

    def get_cua_version(self):
        return self.cua_version

    def get_beta(self):
        return self.beta

    def get_cua_syn_data_version(self):
        return self.cua_syn_data_version

    def get_round_worker_selection_strategy(self):
        return self.round_worker_selection_strategy

    def get_round_worker_selection_strategy_kwargs(self):
        return self.round_worker_selection_strategy_kwargs

    def set_round_worker_selection_strategy_kwargs(self, kwargs):
        self.round_worker_selection_strategy_kwargs = kwargs

    def set_client_selection_strategy(self, strategy):
        self.round_worker_selection_strategy = strategy

    def get_data_path(self):
        return self.data_path

    def get_aggregation_method(self):
        return self.aggregation

    def get_attack_strategy(self):
        return self.attack

    def get_lie_z_value(self):
        return self.lie_z_value

    def get_mal_prop(self):
        return self.mal_prop

    def get_distribution_method(self):
        return self.distribution_method

    def get_epoch_save_start_suffix(self):
        return self.epoch_save_start_suffix

    def get_epoch_save_end_suffix(self):
        return self.epoch_save_end_suffix

    def set_train_data_loader_pickle_path(self, path):
        self.train_data_loader_pickle_path = path

    def get_train_data_loader_pickle_path(self):
        return self.train_data_loader_pickle_path

    def get_benign_data_loader_pickle_path(self):
        return self.benign_data_loader_pickle_path

    def get_malicious_data_loader_pickle_path(self):
        return self.malicious_data_loader_pickle_path

    def get_similarity_epsilon(self):
        return self.similarity_epsilon

    def set_test_data_loader_pickle_path(self, path):
        self.test_data_loader_pickle_path = path

    def get_test_data_loader_pickle_path(self):
        return self.test_data_loader_pickle_path

    def get_cuda(self):
        return self.cuda

    def get_dev_type(self):
        return self.dev_type

    def get_scheduler_step_size(self):
        return self.scheduler_step_size

    def get_scheduler_gamma(self):
        return self.scheduler_gamma

    def get_min_lr(self):
        return self.min_lr

    def get_default_model_folder_path(self):
        return self.default_model_folder_path

    def get_num_epochs(self):
        return self.epochs

    def get_num_classes(self):
        return self.num_classes

    def set_num_poisoned_workers(self, num_poisoned_workers):
        self.num_poisoned_workers = num_poisoned_workers

    def set_num_workers(self, num_workers):
        self.num_workers = num_workers

    def set_model_save_path(self, save_model_path):
        self.save_model_path = save_model_path

    def get_logger(self):
        return self.logger

    def get_loss_function(self):
        return self.loss_function

    def get_net(self):
        return self.net

    def get_num_workers(self):
        return self.num_workers

    def get_num_poisoned_workers(self):
        return self.num_poisoned_workers

    def get_poison_effort(self):
        return self.get_poison_effort

    def get_num_reverse_layers(self):
        return self.num_reverse_layers

    def get_learning_rate(self):
        return self.lr

    def get_momentum(self):
        return self.momentum

    def get_shuffle(self):
        return self.shuffle

    def get_batch_size(self):
        return self.batch_size

    def get_test_batch_size(self):
        return self.test_batch_size

    def get_log_interval(self):
        return self.log_interval

    def get_save_model_folder_path(self):
        return self.save_model_path

    def get_learning_rate_from_epoch(self, epoch_idx):
        lr = self.lr * (self.scheduler_gamma ** int(epoch_idx / self.scheduler_step_size))

        if lr < self.min_lr:
            self.logger.warning("Updating LR would place it below min LR. Skipping LR update.")

            return self.min_lr

        self.logger.debug("LR: {}".format(lr))

        return lr

    def get_contribution_measurement_round(self):
        return  self.contribution_measurement_round

    def get_contribution_measurement_metric(self):
        return self.contribution_measurement_metric

    def should_save_model(self, epoch_idx):
        """
        Returns true/false models should be saved.

        :param epoch_idx: current training epoch index
        :type epoch_idx: int
        """
        if not self.save_model:
            return False

        if epoch_idx == 1 or epoch_idx % self.save_epoch_interval == 0:
            return True

    def log(self):
        """
        Log this arguments object to the logger.
        """
        self.logger.debug("Arguments: {}", str(self))

    def __str__(self):
        return "\nBatch Size: {}\n".format(self.batch_size) + \
               "Test Batch Size: {}\n".format(self.test_batch_size) + \
               "Epochs: {}\n".format(self.epochs) + \
               "Learning Rate: {}\n".format(self.lr) + \
               "Momentum: {}\n".format(self.momentum) + \
               "CUDA Enabled: {}\n".format(self.cuda) + \
               "Shuffle Enabled: {}\n".format(self.shuffle) + \
               "Log Interval: {}\n".format(self.log_interval) + \
               "Scheduler Step Size: {}\n".format(self.scheduler_step_size) + \
               "Scheduler Gamma: {}\n".format(self.scheduler_gamma) + \
               "Scheduler Minimum Learning Rate: {}\n".format(self.min_lr) + \
               "Client Selection Strategy: {}\n".format(self.round_worker_selection_strategy) + \
               "Client Selection Strategy Arguments: {}\n".format(json.dumps(self.round_worker_selection_strategy_kwargs, indent=4, sort_keys=True)) + \
               "Model Saving Enabled: {}\n".format(self.save_model) + \
               "Model Saving Interval: {}\n".format(self.save_epoch_interval) + \
               "Model Saving Path (Relative): {}\n".format(self.save_model_path) + \
               "Epoch Save Start Prefix: {}\n".format(self.epoch_save_start_suffix) + \
               "Epoch Save End Suffix: {}\n".format(self.epoch_save_end_suffix) + \
               "Number of Clients: {}\n".format(self.num_workers) + \
               "Number of Poisoned Clients: {}\n".format(self.num_poisoned_workers) + \
               "NN: {}\n".format(self.net) + \
               "Train Data Loader Path: {}\n".format(self.train_data_loader_pickle_path) + \
               "Test Data Loader Path: {}\n".format(self.test_data_loader_pickle_path) + \
               "Loss Function: {}\n".format(self.loss_function) + \
               "Default Model Folder Path: {}\n".format(self.default_model_folder_path) + \
               "Data Path: {}\n".format(self.data_path)

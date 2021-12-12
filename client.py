import torch
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from federated_learning.schedulers import MinCapableStepLR
import os
import numpy
import copy
from federated_learning.utils import generate_train_loader, generate_train_loader_mal
import torch.nn.functional as F

class Client:

    def __init__(self, args, client_idx, is_mal, train_data_loader, test_data_loader, distributed_train_dataset, gen_net):
        """
        :param args: experiment arguments
        :type args: Arguments
        :param client_idx: Client index
        :type client_idx: int
        :param train_data_loader: Training data loader
        :type train_data_loader: torch.utils.data.DataLoader
        :param test_data_loader: Test data loader
        :type test_data_loader: torch.utils.data.DataLoader
        """
        self.args = args
        self.client_idx = client_idx
        self.is_mal = is_mal
        self.gen_net = gen_net

        self.device = self.initialize_device()
        self.set_net(self.load_default_model())

        self.loss_function = self.args.get_loss_function()()
        self.optimizer = optim.SGD(self.net.parameters(),
            lr=self.args.get_learning_rate(),
            momentum=self.args.get_momentum())
        self.scheduler = MinCapableStepLR(self.args.get_logger(), self.optimizer,
            self.args.get_scheduler_step_size(),
            self.args.get_scheduler_gamma(),
            self.args.get_min_lr())

        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader

        self.distributed_train_dataset = distributed_train_dataset

        self.cua_gen_optimizer = optim.SGD(self.gen_net.parameters(), lr=self.args.get_learning_rate(), momentum=self.args.get_momentum())

    def initialize_device(self):
        """
        Creates appropriate torch device for client operation.
        """
        if torch.cuda.is_available() and self.args.get_cuda():
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")

    def set_net(self, net):
        """
        Set the client's NN.

        :param net: torch.nn
        """
        self.net = net
        self.net.to(self.device)

    def get_is_mal(self):
        """
        Return if this client is attacker

        :param net: torch.nn
        """

        return self.is_mal

    def load_default_model(self):
        """
        Load a model from default model file.

        This is used to ensure consistent default model behavior.
        """
        model_class = self.args.get_net()
        default_model_path = os.path.join(self.args.get_default_model_folder_path(), model_class.__name__ + ".model")

        return self.load_model_from_file(default_model_path)

    def load_model_from_file(self, model_file_path):
        """
        Load a model from a file.

        :param model_file_path: string
        """
        model_class = self.args.get_net()
        model = model_class()

        if os.path.exists(model_file_path):
            try:
                model.load_state_dict(torch.load(model_file_path))
            except:
                self.args.get_logger().warning("Couldn't load model. Attempting to map CUDA tensors to CPU to solve error.")

                model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
        else:
            self.args.get_logger().warning("Could not find model: {}".format(model_file_path))

        return model

    def get_client_index(self):
        """
        Returns the client index.
        """
        return self.client_idx

    def get_nn_parameters(self):
        """
        Return the NN's parameters.
        """
        return self.net.state_dict()

    def get_client_dataloader(self):
        """
        Return the NN's parameters.
        """
        return self.train_data_loader

    def update_nn_parameters(self, new_params):
        """
        Update the NN's parameters.

        :param new_params: New weights for the neural network
        :type new_params: dict
        """
        self.net.load_state_dict(copy.deepcopy(new_params), strict=True)

    def get_client_distribution(self):
        label_class_set = {0,1,2,3,4,5,6,7,8,9}
        client_class_nums = {class_val: 0 for class_val in label_class_set}
        for target in self.distributed_train_dataset[1]:
            try:
                client_class_nums[target] += 1
            except:
                continue
        return list(client_class_nums.values())

    def get_client_datasize(self):
        return len(self.distributed_train_dataset[1])

    def get_init_balanced_syn_images(self, gen_net, num_class=10, factor=1):
        num_total = 5

        noise = torch.randn(num_total, self.args.n_dim).cpu()
        x_query = gen_net(noise)
        return x_query

        # num_each = int(num_total / num_class)
        # count_each = torch.zeros(10).long()
        # res_noise = None
        # r = 0
        # while True:
        #     noise = torch.randn(1000, self.args.n_dim).cpu()
        #     x_query = gen_net(noise)
        #     with torch.no_grad():
        #         v_output = self.net(x_query)
        #         v_output_p = F.softmax(v_output, dim=1)
        #         v_confidence, v_predicted = torch.max(v_output_p, 1)
        #     idx_c, cc = torch.unique(v_predicted, return_counts=True)
        #
        #     for i, e in enumerate(count_each, 0):
        #         if e < num_each:
        #             idx_i = (v_predicted == i).nonzero()
        #             idx_i = idx_i.view(idx_i.shape[0])
        #             e = e.int()
        #
        #         if res_noise is None:
        #             res_noise = noise[idx_i][:num_each - e]
        #         else:
        #             res_noise = torch.cat((res_noise, noise[idx_i][:num_each - e]), 0)
        #     count_each[idx_c] += cc
        #
        #     if r % 20 == 0:
        #         print("generated samples:", count_each)
        #         print("progress:", res_noise.shape[0], '/', num_total)
        #     r += 1
        #     if res_noise.shape[0] >= num_total * factor:
        #         break
        # num_remainder = num_total - res_noise.shape[0]
        # if num_remainder > 0:
        #     if self.args.cuda:
        #         noise = torch.randn(num_remainder, self.z_dim).cuda()
        #     else:
        #         noise = torch.randn(num_remainder, self.z_dim).cpu()
        #     res_noise = torch.cat((res_noise, noise), 0)
        # print("generation rounds:", r, ", init query number:", r * 1000 + num_remainder)
        # # comm.save_results("ini_num_query = " + str(r * 1000 + num_remainder), self.args.res_filename)
        #
        # index = torch.randperm(res_noise.shape[0])
        # res_noise = res_noise[index]
        # res = gen_net(res_noise)
        # return res

    def inference_global_model(self, noise_images):
        mal_dataset = []
        for input in noise_images:
            if self.args.get_cua_version() == 1:
                input_temp = input.unsqueeze(0)
                output = self.net(input_temp)
                _, predicted = torch.max(output.data, 1)
            elif self.args.get_cua_version() == 2:
                predicted = torch.tensor([0])
            else:
                raise Exception("wrong cua version")

            # print("########################hahahhhoooooohhhoooo", predicted.shape)
            # print(predicted.data)
            # print(input_temp.data)

            mal_dataset.append([input, predicted.data.squeeze(0)])
        noise_images_labels = generate_train_loader_mal(self.args, mal_dataset)
        return noise_images_labels

    def generate_noise_images(self, args, gen_net):
        noise = torch.randn(1000, args.n_dim)
        noise_images = gen_net(noise)

        return noise_images

    def train(self, epoch):
        """
        :param epoch: Current epoch #
        :type epoch: int
        """
        if self.is_mal == 'False':
            self.net.train()

            # save model
            if self.args.should_save_model(epoch):
                self.save_model(epoch, self.args.get_epoch_save_start_suffix())

            running_loss = 0.0
            for i, (inputs, labels) in enumerate(self.train_data_loader, 0):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % self.args.get_log_interval() == 0:
                    self.args.get_logger().info('[%d, %5d] loss: %.3f' % (epoch, i, running_loss / self.args.get_log_interval()))

                    running_loss = 0.0

            self.scheduler.step()

        elif self.is_mal == 'CUA':
            self.net.train()
            print("mal train")
            # save model
            if self.args.should_save_model(epoch):
                self.save_model(epoch, self.args.get_epoch_save_start_suffix())


            noise_images = self.get_init_balanced_syn_images(self.gen_net, num_class=10, factor=1)

            mal_data_loader = self.inference_global_model(noise_images = noise_images)

            running_loss = 0.0
            for i, (inputs, labels) in enumerate(mal_data_loader, 0):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.loss_function(outputs, labels)
                loss.backward(retain_graph=True)
                self.optimizer.step()

                # training the generator
                loss_gen = - self.loss_function(outputs, labels)
                loss_gen.backward()
                self.cua_gen_optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % self.args.get_log_interval() == 0:
                    self.args.get_logger().info(
                        '[%d, %5d] loss: %.3f' % (epoch, i, running_loss / self.args.get_log_interval()))

                    running_loss = 0.0

        # save model
        if self.args.should_save_model(epoch):
            self.save_model(epoch, self.args.get_epoch_save_end_suffix())

        return running_loss

    def save_model(self, epoch, suffix):
        """
        Saves the model if necessary.
        """
        self.args.get_logger().debug("Saving model to flat file storage. Save #{}", epoch)

        if not os.path.exists(self.args.get_save_model_folder_path()):
            os.mkdir(self.args.get_save_model_folder_path())

        full_save_path = os.path.join(self.args.get_save_model_folder_path(), "model_" + str(self.client_idx) + "_" + str(epoch) + "_" + suffix + ".model")
        torch.save(self.get_nn_parameters(), full_save_path)

    def calculate_class_precision(self, confusion_mat):
        """
        Calculates the precision for each class from a confusion matrix.
        """
        return numpy.diagonal(confusion_mat) / numpy.sum(confusion_mat, axis=0)

    def calculate_class_recall(self, confusion_mat):
        """
        Calculates the recall for each class from a confusion matrix.
        """
        return numpy.diagonal(confusion_mat) / numpy.sum(confusion_mat, axis=1)

    def test(self):
        self.net.eval()

        correct = 0
        total = 0
        targets_ = []
        pred_ = []
        loss = 0.0
        with torch.no_grad():
            for (images, labels) in self.test_data_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                targets_.extend(labels.cpu().view_as(predicted).numpy())
                pred_.extend(predicted.cpu().numpy())

                loss += self.loss_function(outputs, labels).item()

        accuracy = 100 * correct / total
        confusion_mat = confusion_matrix(targets_, pred_)

        class_precision = self.calculate_class_precision(confusion_mat)
        class_recall = self.calculate_class_recall(confusion_mat)

        self.args.get_logger().debug('Test set: Accuracy: {}/{} ({:.0f}%)'.format(correct, total, accuracy))
        self.args.get_logger().debug('Test set: Loss: {}'.format(loss))
        self.args.get_logger().debug("Classification Report:\n" + classification_report(targets_, pred_))
        self.args.get_logger().debug("Confusion Matrix:\n" + str(confusion_mat))
        self.args.get_logger().debug("Class precision: {}".format(str(class_precision)))
        self.args.get_logger().debug("Class recall: {}".format(str(class_recall)))

        return accuracy, loss, class_precision, class_recall

    def local_test(self):
        self.net.eval()
        correct = 0
        total = 0
        targets_ = []
        pred_ = []
        loss = 0.0
        with torch.no_grad():
            for (images, labels) in self.test_data_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                targets_.extend(labels.cpu().view_as(predicted).numpy())
                pred_.extend(predicted.cpu().numpy())

                loss += self.loss_function(outputs, labels).item()

        accuracy = 100 * correct / total

        self.args.get_logger().debug('Test local: Accuracy: {}/{} ({:.0f}%)'.format(correct, total, accuracy))

        return accuracy
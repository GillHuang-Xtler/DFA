import pickle
import torch
import logging
import os
import numpy as np
import pathlib
from loguru import logger
from federated_learning.schedulers  import MinCapableStepLR
import torch.optim as optim
from federated_learning.nets import FashionMNISTCNN
import logging
logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                    level=logging.DEBUG)

def get_data_loader_from_file(TRAIN_DATA_LOADER_FILE_PATH, TEST_DATA_LOADER_FILE_PATH ):

    train_file = open(TRAIN_DATA_LOADER_FILE_PATH, "rb")
    train_data = pickle.load(train_file)
    test_file = open(TEST_DATA_LOADER_FILE_PATH, "rb")
    test_data = pickle.load(test_file)
    logging.debug("finished getting data")

    return train_data, test_data


def get_net_from_default_models():
    # select net
    net = FashionMNISTCNN()
    logging.debug("finished loading models")
    return net

def dis_loss(outputs, labels, net_param):
    loss_function = torch.nn.CrossEntropyLoss()

    loss = loss_function(outputs, labels)
    print('loss'+str(loss))
    # loss = torch.zeros(1, requires_grad=True)

    model_all = torch.load("tmp_models/weights_all+1.model")
    # if new_param is None:
    #     return loss

    mu = 0.001
    reg = torch.tensor(0.)

    for key in net_param.keys():
        diff = net_param[key] - model_all[key]
        reg += (torch.sum(torch.norm(diff.float())).float())
        print('reg' + str(reg))
    loss += (reg)

    return loss


def train(device, epoches , train_data, net, use_disloss = True):
    logging.debug("start to train")
    startepoch = 0
    loss_function = torch.nn.CrossEntropyLoss()
    save_path = 'tmp_models/weights_all_mal.model'
    if os.path.exists(save_path) is not True:
        pathlib.Path("tmp_models").mkdir(parents=True, exist_ok=True)

    optimizer = optim.Adam(net.to(device).parameters(),
                               lr=0.001,)
                               # momentum=0.9)
    scheduler = MinCapableStepLR(logger = logger,
                                 optimizer = optimizer,
                                 step_size = 10,
                                 gamma = 0.1,
                                 min_lr = 1e-5)
    for epoch in range(startepoch, epoches):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_data, 0):
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            if use_disloss == False:
                loss = loss_function(outputs, labels)
            else:
                loss = dis_loss(outputs=outputs, labels =labels, net_param = net.state_dict())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 0:
                logging.info(
                    '[%d, %5d] loss: %.3f' % (epoch, i, running_loss / 10))

                running_loss = 0.0
        scheduler.step()

        # if epoch == epoches/2:
        #     torch.save(net.state_dict(), 'tmp_models/weights_half.model')
        # elif epoch == epoches-2:
        #     torch.save(net.state_dict(), 'tmp_models/weights_last.model')

    logging.debug("Finished Training")

    torch.save(net.state_dict(), save_path)

    return running_loss


def local_test(test_data_loader, device, net):
    correct = 0
    total = 0
    targets_ = []
    pred_ = []
    loss = 0.0
    loss_function = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for (images, labels) in test_data_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            targets_.extend(labels.cpu().view_as(predicted).numpy())
            pred_.extend(predicted.cpu().numpy())

            loss += loss_function(outputs, labels).item()

    accuracy = 100 * correct / total

    logging.info('Test local: Accuracy: {}/{} ({:.0f}%)'.format(correct, total, accuracy))
    return accuracy

if __name__ =="__main__":
    # prepare data
    TRAIN_DATA_LOADER_FILE_PATH = "data_loaders/fashion-mnist/free_data_loader.pickle"
    TEST_DATA_LOADER_FILE_PATH = "data_loaders/fashion-mnist/test_data_loader.pickle"
    train_data, test_data = get_data_loader_from_file(TRAIN_DATA_LOADER_FILE_PATH, TEST_DATA_LOADER_FILE_PATH )
    model_all = FashionMNISTCNN()
    model_all.load_state_dict(torch.load("tmp_models/weights_all.model"))

    # train
    running_loss = train(torch.device('cpu'), epoches=5, train_data=train_data, net=model_all)
    # print(running_loss)

    # test
    # model_zero = model_half = model_last =  model_all = FashionMNISTCNN()
    # model_reverse = FashionMNISTCNN()
    # model_zero = torch.load("default_models/FashionMNISTCNN.model")
    model_all = torch.load("tmp_models/weights_all+1.model")
    # model_last = torch.load("tmp_models/weights_last.model")
    # model_half = torch.load("tmp_models/weights_half.model")
    model_reverse = torch.load("tmp_models/weights_all_mal.model")
    # model_reverse.load_state_dict(torch.load("tmp_models/weights_reverse_loss.model"))
    # accuracy = local_test(test_data_loader = test_data, device= torch.device('cpu'), net = model_reverse)
    # dis_last = [torch.norm((model_all[name].data - model_last[name].data).float()) for name in model_all.keys()]
    # dis_half = [torch.norm((model_all[name].data - model_half[name].data).float()) for name in model_all.keys()]
    # dis_zero = [torch.norm((model_all[name].data - model_zero[name].data).float())  for name in model_all.keys()]
    dis_zero = [torch.norm((model_all[name].data - model_reverse[name].data).float())  for name in model_all.keys()]
    #
    print(sum(dis_zero))


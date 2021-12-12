from federated_learning.arguments import Arguments
from federated_learning.nets import Cifar10CNN
from federated_learning.nets import Cifar100ResNet
from federated_learning.nets import FashionMNISTCNN
from federated_learning.nets import FashionMNISTResNet
from federated_learning.nets import Cifar10ResNet
from federated_learning.nets import Cifar100VGG, STL10VGG, MNISTCNN
# from federated_learning.nets import TRECCNN
import os
import torch
from loguru import logger

if __name__ == '__main__':
    args = Arguments(logger)
    if not os.path.exists(args.get_default_model_folder_path()):
        os.mkdir(args.get_default_model_folder_path())

    # ---------------------------------
    # ----------- Cifar10CNN ----------
    # ---------------------------------
    full_save_path = os.path.join(args.get_default_model_folder_path(), "Cifar10CNN.model")
    torch.save(Cifar10CNN().state_dict(), full_save_path)
    # ---------------------------------
    # --------- Cifar10ResNet ---------
    # ---------------------------------
    full_save_path = os.path.join(args.get_default_model_folder_path(), "Cifar10ResNet.model")
    torch.save(Cifar10ResNet().state_dict(), full_save_path)

    # ---------------------------------
    # -------- FashionMNISTCNN --------
    # ---------------------------------
    full_save_path = os.path.join(args.get_default_model_folder_path(), "FashionMNISTCNN.model")
    torch.save(FashionMNISTCNN().state_dict(), full_save_path)

    # ---------------------------------
    # ------ FashionMNISTResNet -------
    # ---------------------------------
    full_save_path = os.path.join(args.get_default_model_folder_path(), "FashionMNISTResNet.model")
    torch.save(FashionMNISTResNet().state_dict(), full_save_path)

    # ---------------------------------
    # ----------- Cifar100CNN ---------
    # ---------------------------------
    full_save_path = os.path.join(args.get_default_model_folder_path(), "Cifar100ResNet.model")
    torch.save(Cifar100ResNet().state_dict(), full_save_path)

    # ---------------------------------
    # ----------- Cifar100VGG ---------
    # ---------------------------------
    full_save_path = os.path.join(args.get_default_model_folder_path(), "Cifar100VGG.model")
    torch.save(Cifar100VGG().state_dict(), full_save_path)

    # # ---------------------------------
    # # ----------- TRECCNN ---------
    # # ---------------------------------
    # full_save_path = os.path.join(args.get_default_model_folder_path(), "TRECCNN.model")
    # torch.save(TRECCNN().state_dict(), full_save_path)

    # ---------------------------------
    # ----------- MNISTCNN ------------
    # ---------------------------------
    full_save_path = os.path.join(args.get_default_model_folder_path(), "MNISTCNN.model")
    torch.save(MNISTCNN().state_dict(), full_save_path)

    # ---------------------------------
    # ----------- STL10VGG ------------
    # ---------------------------------
    full_save_path = os.path.join(args.get_default_model_folder_path(), "STL10VGG.model")
    torch.save(STL10VGG().state_dict(), full_save_path)
# import torch
# import torchtext
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from .dataset import Dataset
# from torchtext import data
# from torchtext import datasets
# # from torch.utils.data import DataLoader
#
# import numpy as np
# import random
# import math
#
# class TextDataLoader:
#     def __init__(self,data_iter):
#         self.data_iter = data_iter
#         self.length = len(data_iter)
#
#     def __len__(self):
#         return self.length
#
#     def __iter__(self):
#         # 注意：此处调整features为 batch first，并调整label的shape和dtype
#         for batch in self.data_iter:
#             yield(torch.transpose(batch.text,0,1),
#                   torch.unsqueeze(batch.label.float(),dim = 1))
#
# class TRECDataset(Dataset):
#
#     def __init__(self, args):
#         super(TRECDataset, self).__init__(args)
#
#
#
#     def load_train_dataset(self):
#         self.get_args().get_logger().debug("Loading TREC train data")
#
#         # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#         # transform = transforms.Compose([
#         #     transforms.Pad(4),
#         #     transforms.RandomCrop(96),
#         #     transforms.RandomHorizontalFlip(),
#         #     transforms.ToTensor(),
#         #     normalize
#         # ])
#         TEXT = data.Field(tokenize="spacy", tokenizer_language="en_core_web_sm", batch_first=True)
#         LABEL = data.LabelField()
#         train_dataset, test_dataset = datasets.TREC.splits(TEXT, LABEL,root= 'data', fine_grained=False)
#         TEXT.build_vocab(train_dataset)
#         train_iter, test_iter = torchtext.data.Iterator.splits(
#             (train_dataset, test_dataset), sort_within_batch=True, sort_key=lambda x: len(x.text),
#             batch_sizes=(len(train_dataset), len(test_dataset)))
#         MAX_VOCAB_SIZE = 25_000
#         TEXT.build_vocab(train_dataset, max_size=MAX_VOCAB_SIZE,
#                          vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
#         LABEL.build_vocab(train_dataset)
#
#         train_loader = TextDataLoader(train_iter)
#
#         train_data = self.get_tuple_from_data_loader(train_loader)
#
#         self.get_args().get_logger().debug("Finished loading TREC train data")
#
#         return train_data
#
#     def load_test_dataset(self):
#         self.get_args().get_logger().debug("Loading TREC test data")
#         #
#         # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#         # transform = transforms.Compose([
#         #     transforms.ToTensor(),
#         #     normalize
#         # ])
#         TEXT = data.Field(tokenize="spacy", tokenizer_language="en_core_web_sm", batch_first=True)
#         LABEL = data.LabelField()
#         train_dataset, test_dataset = datasets.TREC.splits(TEXT, LABEL,root= 'data', fine_grained=False)
#         TEXT.build_vocab(train_dataset)
#         train_iter, test_iter = torchtext.data.Iterator.splits(
#             (train_dataset, test_dataset), sort_within_batch=True, sort_key=lambda x: len(x.text),
#             batch_sizes=(len(train_dataset), len(test_dataset)))
#         MAX_VOCAB_SIZE = 25_000
#         TEXT.build_vocab(train_dataset, max_size=MAX_VOCAB_SIZE,
#                          vectors="glove.6B.100d", unk_init=torch.Tensor.normal_)
#         LABEL.build_vocab(train_dataset)
#
#         test_loader = TextDataLoader(test_iter)
#
#         train_data = self.get_tuple_from_data_loader(test_loader)
#
#         # test_data = self.get_tuple_from_data_loader(test_loader)
#
#         self.get_args().get_logger().debug("Finished loading TREC test data")
#
#         return train_data

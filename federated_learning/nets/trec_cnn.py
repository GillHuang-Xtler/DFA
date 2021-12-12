from typing import Tuple
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchtext import data
from torchtext import data
from torchtext import datasets
import torchkeras

# TEXT = data.Field(tokenize="spacy", tokenizer_language="en_core_web_sm", batch_first=True)
# LABEL = data.LabelField()
# train_dataset, test_dataset = datasets.TREC.splits(TEXT, LABEL, root='data', fine_grained=False)
#
# MAX_VOCAB_SIZE = 25_000
# TEXT.build_vocab(train_dataset,max_size=MAX_VOCAB_SIZE,
#                  vectors="glove.6B.100d",unk_init=torch.Tensor.normal_)
# LABEL.build_vocab(train_dataset)
# INPUT_DIM = len(TEXT.vocab)
# EMBEDDING_DIM = 100
# N_FILTERS = 100
# FILTER_SIZES = [2,3,4]
# OUTPUT_DIM = len(LABEL.vocab)
# DROPOUT = 0.5
# PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
#
# class TRECCNN(nn.Module):
#     def __init__(self, vocab_size = INPUT_DIM, embedding_dim=EMBEDDING_DIM, n_filters=N_FILTERS, filter_sizes =FILTER_SIZES, output_dim=OUTPUT_DIM,
#                  dropout=DROPOUT, pad_idx=PAD_IDX):
#         super(TRECCNN, self).__init__()
#
#         self.embed = nn.Embedding(vocab_size, embedding_dim, pad_idx)
#
#         self.convs = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim, out_channels=n_filters
#                                               , kernel_size=fs) for fs in filter_sizes])
#
#         self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
#
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, text):
#         # text:(batch,seq)
#         embeded = self.embed(text)
#         # embeded:(batch,seq,embedding)
#
#         embeded = embeded.permute(0, 2, 1)
#         # embeded:(batch,embedding,seq)
#
#         conveds = [torch.relu(conv(embeded)) for conv in self.convs]
#         # conved:(batch,n_filters,seq-kernel_size+1)
#
#         pooled = [F.max_pool1d(conved, conved.shape[-1]).squeeze(-1) for conved in conveds]
#         # pooled:(batch,n_filters)
#
#         cat = self.dropout(torch.cat(pooled, dim=1))
#         # cat:(batch,n_filters*len(filter_size))
#
#         return self.fc(cat)
#         # （batch,output_dim）
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torchvision import models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from miscc.config import cfg
from attention import SpatialAttention as SPATIAL_NET
from attention import ChannelAttention as CHANNEL_NET
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

from miscc.utils import mkdir_p
from miscc.utils import build_super_images
from miscc.losses import sent_loss, words_loss
from miscc.config import cfg, cfg_from_file

from datasets import TextDataset
from datasets import prepare_data

from model import RNN_ENCODER, CNN_ENCODER

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms


class RNN_ENCODER(nn.Module):
    def __init__(self, ntoken, ninput=300, drop_prob=0.5,
                 nhidden=128, nlayers=1, bidirectional=True):
        super(RNN_ENCODER, self).__init__()
        self.n_steps = cfg.TEXT.WORDS_NUM
        self.ntoken = ntoken  # size of the dictionary
        self.ninput = ninput  # size of each embedding vector
        self.drop_prob = drop_prob  # probability of an element to be zeroed
        self.nlayers = nlayers  # number of recurrent layers
        self.bidirectional = bidirectional
        self.rnn_type = cfg.RNN_TYPE
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        # number of features in the hidden state
        self.nhidden = nhidden // self.num_directions

        self.define_module()
        self.init_weights()

    def define_module(self):
        self.encoder = nn.Embedding(self.ntoken, self.ninput)
        self.drop = nn.Dropout(self.drop_prob)
        if self.rnn_type == 'LSTM':
            # dropout: if non-zero, introduces a dropout layer on
            # the outputs of each RNN layer except the last layer
            self.rnn = nn.LSTM(self.ninput, self.nhidden,
                               self.nlayers, batch_first=True,
                               dropout=self.drop_prob,
                               bidirectional=self.bidirectional)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(self.ninput, self.nhidden,
                              self.nlayers, batch_first=True,
                              dropout=self.drop_prob,
                              bidirectional=self.bidirectional)
        else:
            raise NotImplementedError

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()),
                    Variable(weight.new(self.nlayers * self.num_directions,
                                        bsz, self.nhidden).zero_()))
        else:
            return Variable(weight.new(self.nlayers * self.num_directions,
                                       bsz, self.nhidden).zero_())

    def forward(self, captions, cap_lens, hidden, mask=None):
        # input: torch.LongTensor of size batch x n_steps
        # --> emb: batch x n_steps x ninput
        # drop: random zero
        emb = self.drop(self.encoder(captions))

        # Returns: a PackedSequence object
        cap_lens = cap_lens.data.tolist()

        emb = pack_padded_sequence(emb, cap_lens, batch_first=True)
        # hidden and memory (num_layers * num_directions, batch, hidden_size):
        # tensor containing the initial hidden state for each element in batch.
        # output (batch, seq_len, hidden_size * num_directions)
        # or a PackedSequence object:
        # tensor containing output features (h_t) from the last layer of RNN
        output, hidden = self.rnn(emb, hidden)
        # PackedSequence object
        # --> (batch, seq_len, hidden_size * num_directions)
        output = pad_packed_sequence(output, batch_first=True)[0]
        # output = self.drop(output)
        # --> batch x hidden_size*num_directions x seq_len
        words_emb = output.transpose(1, 2)
        # --> batch x num_directions*hidden_size
        if self.rnn_type == 'LSTM':
            sent_emb = hidden[0].transpose(0, 1).contiguous()
        else:
            sent_emb = hidden.transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)
        return words_emb, sent_emb

def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/DAMSM/bird.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # args = parse_args()
    # if args.cfg_file is not None:
    #     cfg_from_file(args.cfg_file)
    #
    # if args.gpu_id == -1:
    #     cfg.CUDA = False
    # else:
    #     cfg.GPU_ID = args.gpu_id
    #
    # if args.data_dir != '':
    #     cfg.DATA_DIR = args.data_dir
    # print('Using config:')
    # pprint.pprint(cfg)
    #
    # if not cfg.TRAIN.FLAG:
    #     args.manualSeed = 100
    # elif args.manualSeed is None:
    #     args.manualSeed = random.randint(1, 10000)
    # random.seed(args.manualSeed)
    # np.random.seed(args.manualSeed)
    # torch.manual_seed(args.manualSeed)
    # if cfg.CUDA:
    #     torch.cuda.manual_seed_all(args.manualSeed)
    #
    # ##########################################################################
    # now = datetime.datetime.now(dateutil.tz.tzlocal())
    # timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    # output_dir = '../output/%s_%s_%s' % \
    #     (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)
    #
    # model_dir = os.path.join(output_dir, 'Model')
    # image_dir = os.path.join(output_dir, 'Image')
    # mkdir_p(model_dir)
    # mkdir_p(image_dir)
    #
    # torch.cuda.set_device(cfg.GPU_ID)
    # cudnn.benchmark = True
    #
    # # Get data loader ##################################################
    # imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
    # batch_size = cfg.TRAIN.BATCH_SIZE
    # image_transform = transforms.Compose([
    #     transforms.Scale(int(imsize * 76 / 64)),
    #     transforms.RandomCrop(imsize),
    #     transforms.RandomHorizontalFlip()])
    # dataset = TextDataset(cfg.DATA_DIR, 'train',
    #                       base_size=cfg.TREE.BASE_SIZE,
    #                       transform=image_transform)
    #
    # print(dataset.n_words, dataset.embeddings_num)
    # assert dataset
    # dataloader = torch.utils.data.DataLoader(
    #     dataset, batch_size=batch_size, drop_last=True,
    #     shuffle=True, num_workers=int(cfg.WORKERS))
    #
    # # # validation data #
    # dataset_val = TextDataset(cfg.DATA_DIR, 'test',
    #                           base_size=cfg.TREE.BASE_SIZE,
    #                           transform=image_transform)
    # dataloader_val = torch.utils.data.DataLoader(
    #     dataset_val, batch_size=batch_size, drop_last=True,
    #     shuffle=True, num_workers=int(cfg.WORKERS))
    #
    # text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    #
    # print(text_encoder)

    import torch


    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenized input
    text1 = "Who was Jim Henson ? Jim Henson was a puppeteer"
    text2 = "I am The Anh"
    tokenized_text = tokenizer.tokenize(text1)

    # Mask a token that we will try to predict back with `BertForMaskedLM`
    masked_index = 6
    tokenized_text[masked_index] = '[MASK]'
    assert tokenized_text == ['who', 'was', 'jim', 'henson', '?', 'jim', '[MASK]', 'was', 'a', 'puppet', '##eer']

    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    segments_ids = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    model = BertModel.from_pretrained('bert-base-uncased')
    print(model)
    model.eval()

    # Predict hidden states features for each layer
    encoded_layers, _ = model(tokens_tensor)
    output = encoded_layers[1]
    # We have a hidden states for each of the 12 layers in model bert-base-uncased
    assert len(encoded_layers) == 12
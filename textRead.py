from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()

MAX_LENGTH = 20
# make dict

SOS_token = 0
EOS_token = 1
UNKNOWN_token = 2


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.index2word = {}
        self.word2count = {0: "SOS", 1: "EOS", 2: "UNKNOWN"}
        self.n_words = 3  # count SOS and EOS and UNKWON

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def readData(path1, path2):
    print("Reading lines...")

    inputs = open(path1, encoding='utf-8').read().strip().split('\n')
    outputs = open(path2, encoding='utf-8').read().strip().split('\n')

    inp = Lang('inputs')
    outp = Lang('outputs')

    pair = []
    for i in range(len(inputs)):
        pair.append([inputs[i], outputs[i]])

    return inputs, outputs, inp, outp, pair

def prepareData(input_lang, output_lang, pairs):
    print("Read %s sentence pairs" % len(pairs))

    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs

def fileWrite(pairs, path3):
    pair = []
    f = open(path3, 'w')
    for p in range(len(pairs)):
        pair = pairs[p]
        f.write(pair[0])
        f.write('\t')
        f.write(pair[1])
        f.write('\n')
    f.close()

train_source, train_target, train_inp, train_outp, train_pair = readData('/workspace/workspace/chatbot/data/train.src', '/workspace/workspace/chatbot/data/train.tgt')
train_input, train_output, train_pairs = prepareData(train_inp, train_outp, train_pair)
fileWrite(train_pairs, '/workspace/workspace/chatbot/train_text.txt')
print("train file saved")

valid_source, valid_target, valid_inp, valid_outp, valid_pair = readData('/workspace/workspace/chatbot/data/valid.src', '/workspace/workspace/chatbot/data/valid.tgt')
valid_input, valid_output, valid_pairs = prepareData(valid_inp, valid_outp, valid_pair)
fileWrite(valid_pairs, '/workspace/workspace/chatbot/valid_text.txt')
print("valid file saved")

test_source, test_target, test_inp, test_outp, test_pair = readData('/workspace/workspace/chatbot/data/test.src', '/workspace/workspace/chatbot/data/test.tgt')
test_input, test_output, test_pairs = prepareData(test_inp, test_outp, test_pair)
fileWrite(test_pairs, '/workspace/workspace/chatbot/test_text.txt')
print("test file saved")


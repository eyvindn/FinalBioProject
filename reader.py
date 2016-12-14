# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

import numpy as np
import tensorflow as tf


def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data]


def ptb_raw_data():
  """Load PTB raw data from data directory "data_path".

  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.

  The PTB dataset comes from Tomas Mikolov's webpage:

  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz

  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """
  train_path = os.path.join(data_path, "ptb.train.txt")
  valid_path = os.path.join(data_path, "ptb.valid.txt")
  test_path = os.path.join(data_path, "ptb.test.txt")

  word_to_id = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, test_data, vocabulary

def load_promoter_data(data_path=None):

    with open('promoters_small.fa', 'r') as handle:
        content = handle.readlines()

    X = []
    Y = []
    smallx = []
    mapping = {
        "A": [1,0,0,0],
        "T": [0,1,0,0],
        "G": [0,0,1,0],
        "C": [0,0,0,1]
    }
    currLine = None
    skip = False

    for line in content:
        #print(line)
        if not skip or line[0:1] == ">":
            if line[0:1] == ">":
                skip = False
                # Start reading in
                if currLine != None:
                    X.append(currLine[:-1])
                    label = np.zeros((2000))
                    label[1000] = 1 #1 little fucker is the TSS
                    Y.append(label)
                currLine = []
            else:
                for character in line.strip():
                    if character not in mapping:
                        skip = True
                        currLine = None
                        break
                    else:
                        currLine.append(mapping[character])
    if not skip:
        X.append(currLine[:-1])
        label = np.zeros((2000))
        label[1000] = 1  # 1 little fucker is the TSS
        Y.append(label)

    return (np.array(X), np.array(Y))

def promoter_iterator(raw_data, batch_size, num_steps):
    #raw_data should be tuple ([no_of_tss:s, 2000, 4], [no_of_tss:s, 2000, 2]), let's try and make batch
    X = raw_data[0]
    Y = raw_data[1]

    #Calculate how many batche_sizes are in this
    total_length = X.shape[0]
    trimmed_length = (total_length // batch_size) * batch_size

    #Trim these so we can reshape them
    X = X[0:trimmed_length][:][:]
    Y = Y[0:trimmed_length][:]

    #Reshape!
    X = np.reshape(X, (batch_size, -1 ,4))
    Y = np.reshape(Y, (batch_size, -1))

    #Verify that num_steps is a factor of 2001
    if 2000 % num_steps != 0:
        print("ERROR PLEASE MAKE NUM STEPS A MULTIPLE!")

    length = X.shape[1]
    epoch_size = length // 2000

    for i in range(epoch_size):
        for j in range(1000 - num_steps + 1, 1001):
            x = X[:, i * 2000 + j: i * 2000 + j + num_steps][:]
            y = Y[:, i * 2000 + j: i * 2000 + j + num_steps]
            yield (x, y)

    # length = X.shape[1]
    # epoch_size = length // num_steps
    #
    # for i in range(epoch_size):
    #     x = X[:, i * num_steps: (i + 1) * num_steps]
    #     y = Y[:, i * num_steps: (i + 1) * num_steps]
    #     yield (x, y)



def ptb_iterator(raw_data, batch_size, num_steps):
  """Iterate on the raw PTB data.

  This generates batch_size pointers into the raw PTB data, and allows
  minibatch iteration along these pointers.

  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.

  Yields:
    Pairs of the batched data, each a matrix of shape [batch_size, num_steps].
    The second element of the tuple is the same data time-shifted to the
    right by one.

  Raises:
    ValueError: if batch_size or num_steps are too high.
  """
  raw_data = np.array(raw_data, dtype=np.int32)
  data_len = len(raw_data)
  batch_len = data_len // batch_size
  data = np.zeros([batch_size, batch_len], dtype=np.int32)
  for i in range(batch_size):
    data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

  epoch_size = (batch_len - 1) // num_steps

  if epoch_size == 0:
    raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

  for i in range(epoch_size):
    x = data[:, i*num_steps:(i+1)*num_steps]
    y = data[:, i*num_steps+1:(i+1)*num_steps+1]
    yield (x, y)

# derp = load_promoter_data()
# print("doing reshape")
# herp = promoter_iterator(derp, 40, 100)
# test = herp.next()
# print(test[0].shape)
# print(test[1].shape)
# print(test[0])
# print(test[1])
#
# # print(derp[0].shape)
# # print(derp[1].shape)
# # print(derp[0][1])
# # print(derp[1][1])
#
# print("TEST")

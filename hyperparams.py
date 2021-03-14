# -*- coding: utf-8 -*-
#/usr/bin/python2

import os
class Hyperparams:
    '''Hyperparameters'''
    # data
    base_dir = 'your_data_path'
    base_dir = os.getcwd() + '/'
    source_train = base_dir + 'corpora/train_query.txt'
    target_train = base_dir + 'corpora/train_answer.txt'
    source_test = base_dir + 'corpora/test_query.txt'
    target_test = base_dir + 'corpora/test_answer.txt'
    source_dev = base_dir + 'corpora/dev_query.txt'
    target_dev = base_dir + 'corpora/dev_answer.txt' 
    # training
    batch_size = 64 # alias = N
    lr = 0.0001 # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir2020_test' # log directory
    
    # model
    maxlen = 50 # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    min_cnt = 1 # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512 # alias = C
    num_blocks = 6 # number of encoder/decoder blocks
    num_epochs = 50000
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False # If True, use sinusoid. If false, positional embedding.
    
    num_layers=1
    max_turn=15
    
    
    

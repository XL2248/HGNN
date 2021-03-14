# -*- coding: utf-8 -*-
#/usr/bin/python2
from __future__ import print_function
#from hyperparams import Hyperparams as hp
import tensorflow as tf
import pickle as pkl
import numpy as np
import codecs, code, os
#import regex
emotion2idx = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}
idx2emotion = {0: 'neutral', 1: 'surprise', 2: 'fear', 3: 'sadness', 4: 'joy', 5: 'disgust', 6: 'anger'}

print(os.getcwd()) 
base_dir = os.getcwd() + '/'
def load_de_vocab(hp):
    vocab = [line.split()[0] for line in codecs.open(base_dir+'preprocessed/de.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def load_en_vocab(hp):
    vocab = [line.split()[0] for line in codecs.open(base_dir+'preprocessed/en.vocab.tsv', 'r', 'utf-8').read().splitlines() if int(line.split()[1])>=hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def load_speaker_vocab(hp):
    vocab = [line.split('\n')[0] for line in codecs.open(base_dir+'preprocessed/speakers.txt', 'r', 'utf-8').read().splitlines()]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word

def create_data(hp, source_sents, target_sents, image_fea, A): 
    de2idx, idx2de = load_de_vocab(hp)
    en2idx, idx2en = load_en_vocab(hp)
    speaker2idx, idx2speaker = load_speaker_vocab(hp)
    # Index

    x_A, x_list, x_image_list, y_image_list, y_list, Sources, Targets, Src_emotion, Tgt_emotion, Speaker = [], [], [], [], [], [], [], [], [], []
    max_turn = 0
    max_length = 0
    for index, (source_sent, target_sent, image_f, a) in enumerate(zip(source_sents, target_sents, image_fea, A)):
        source_sent_split = source_sent.split(u"</d>")
        source_sent_split.pop()
        image_feature = image_f.split("\t\t")
        image_feature.pop()
        image = image_feature[:-1]
        y_imag = image_feature[-1]

        x=[]
        x_image=[]
        y_image=[]
        src_emotion=[]
        turn_num = 0
        for sss, imag in zip(source_sent_split, image):
            if len(sss.split())==0 or len(sss.split("\t\t")) == 1:
                print('is 0', index, sss)
                continue
            #print(sss)
            x_speaker, text, emotion = sss.split("\t\t")[0], sss.split("\t\t")[1], sss.split("\t\t")[2]
            if len((text + u" </S>").split()) > max_length:
                max_length = len((text + u" </S>").split())
       
            x.append( [de2idx.get(word, 1) for word in (text + u" </S>").split()]) # 1: OOV, </S>: End of Text
            x_image.append([float(item) for item in imag.split()])

            src_emotion.append([emotion2idx[emotion.split()[0]]])
            turn_num += 1

        target_sent_split = target_sent.split(u"</d>")
        if len(x) > max_turn:
            max_turn = len(x)

        speaker = []
        tgt_emotion = []
        name = ' '.join(target_sent_split[0].split())
        if name not in speaker2idx:
            speaker.append(speaker2idx[u"newer"])
        else:
            speaker.append(speaker2idx[name])
        tgt_emotion.append(emotion2idx[target_sent_split[2].split()[0]])
        src_emotion.append(emotion2idx[target_sent_split[2].split()[0]])
        y = [en2idx.get(word, 1) for word in (target_sent_split[1] + u" </S>").split()] 
        y_image.append([float(item) for item in y_imag.split()])

        
        if max(len(x), len(y)) <=hp.maxlen:
            x_list.append(np.array(x))
            x_image_list.append(np.array(x_image))
            y_image_list.append(np.array(y_image))
            y_list.append(np.array(y))
            Src_emotion.append(np.array(src_emotion))
            Tgt_emotion.append(np.array(tgt_emotion))
            Speaker.append(np.array(speaker))
            Sources.append(source_sent)
            Targets.append(target_sent)
            x_A.append(a)

    #code.interact(local=locals())
    print('max_turn=', max_turn)
    # Pad      
    print('max_length=', max_length)
    X = np.zeros([len(x_list), hp.max_turn, hp.maxlen], np.int32)
    X_image = np.zeros([len(x_list), hp.max_turn, 17], np.float32)
    Y_image = np.zeros([len(x_list), 17], np.float32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    X_length = np.zeros([len(x_list), hp.max_turn], np.int32)
    X_turn_number = np.zeros([len(x_list)], np.int32)
    SRC_emotion = np.zeros([len(x_list), hp.max_turn], np.int32)
    TGT_emotion = np.zeros([len(y_list)], np.int32)
    Speakers = np.zeros([len(y_list)], np.int32)
    X_A = np.zeros([len(x_list), 7, 90, 90], np.float32)
    for i, (x, y, z) in enumerate(zip(x_list, y_list, x_image_list)):# i-th dialogue
        j = 0
        for j in range(len(x)): # j-th turn
            if j >= hp.max_turn :
                break
            if len(x[j])<hp.maxlen:
                X[i][j] = np.lib.pad(x[j], [0, hp.maxlen-len(x[j])], 'constant', constant_values=(0, 0))#i-th dialogue j-th turn
            else:
                X[i][j]=x[j][:hp.maxlen]#
            X_image[i][j] = z[j]
            X_length[i][j] = len(x[j])# seq length mask
            SRC_emotion[i][j] = Src_emotion[i][j][0]
            #code.interact(local=locals())
        X_turn_number[i] = len(x) + 1# turn number`
        Y_image[i] = y_image_list[i]
        X_image[i][j+1] = y_image_list[i]
        #code.interact(local=locals())

        Y[i] = np.lib.pad(y, [0, hp.maxlen-len(y)], 'constant', constant_values=(0, 0))
        TGT_emotion[i] = Tgt_emotion[i][0]
        Speakers[i] = Speaker[i][0]
        for k in range(len(x_A[i])):
            X_A[i][k] = x_A[i][k].toarray()

    return X, X_image, Y_image, X_length, Y, Sources, Targets, X_turn_number, SRC_emotion, TGT_emotion, Speakers, X_A

def load_image_data(hp, data_type):
    image_features = [line for line in codecs.open(hp.corpora_path + data_type+'_image.txt', 'r', 'utf-8').read().split("\n") if line]

    return image_features

def load_train_data(hp):
    de_sents = [line for line in codecs.open(hp.source_train, 'r', 'utf-8').read().split("\n") if line]
    en_sents = [line for line in codecs.open(hp.target_train, 'r', 'utf-8').read().split("\n") if line]
    image_fea = load_image_data(hp, data_type='train')    
    A = pkl.load(open(hp.corpora_path + 'train.pkl', 'rb'))
    X, X_image, Y_image, X_length, Y, Sources, Targets, X_turn_number, SRC_emotion, TGT_emotion, Speakers, A = create_data(hp, de_sents, en_sents, image_fea, A)
    return X, X_image, Y_image, X_length,Y,Sources,Targets, X_turn_number, SRC_emotion, TGT_emotion, Speakers, A
    
def load_test_data(hp):
    def _refine(line):

        return line.strip()
    
    de_sents = [_refine(line) for line in codecs.open(hp.source_test, 'r', 'utf-8').read().split("\n") if line]
    en_sents = [_refine(line) for line in codecs.open(hp.target_test, 'r', 'utf-8').read().split("\n") if line]
    image_fea = load_image_data(hp, data_type='test')
    A = pkl.load(open(hp.corpora_path + 'test.pkl', 'rb'))
    X, X_image, Y_image, X_length, Y, Sources, Targets, X_turn_number, SRC_emotion, TGT_emotion, Speakers, A = create_data(hp, de_sents, en_sents, image_fea, A)
    return X, X_image, Y_image, X_length, Y, Sources, Targets, X_turn_number, SRC_emotion, TGT_emotion, Speakers, A # (1064, 150)

def load_dev_data(hp):
    def _refine(line):

        return line.strip()
    
    de_sents = [_refine(line) for line in codecs.open(hp.source_dev, 'r', 'utf-8').read().split("\n") if line]
    en_sents = [_refine(line) for line in codecs.open(hp.target_dev, 'r', 'utf-8').read().split("\n") if line]
    image_fea = load_image_data(hp, data_type='dev')
    A = pkl.load(open(hp.corpora_path + 'dev.pkl', 'rb'))
    X, X_image, Y_image, X_length, Y, Sources, Targets, X_turn_number, SRC_emotion, TGT_emotion, Speakers, A = create_data(hp, de_sents, en_sents, image_fea, A)
    return X, X_image, Y_image, X_length,Y, Sources, Targets, X_turn_number, SRC_emotion, TGT_emotion, Speakers, A # (1064, 150)

def get_batch_data(hp):
    # Load data
    X, X_image, Y_image, X_length, Y, sources,targets, X_turn_number, SRC_emotion, TGT_emotion, Speakers, A = load_train_data(hp)
    
    # calc total batch count
    num_batch = len(X) // hp.batch_size
    
    # Convert to tensor
    X = tf.convert_to_tensor(X, tf.int32)
    X_ima = tf.convert_to_tensor(X_image, tf.float32)
    Y_ima = tf.convert_to_tensor(Y_image, tf.float32)
    Y = tf.convert_to_tensor(Y, tf.int32)
    X_length = tf.convert_to_tensor(X_length,tf.int32)
    X_turn_number = tf.convert_to_tensor(X_turn_number,tf.int32)
    SRC_emotion = tf.convert_to_tensor(SRC_emotion, tf.int32)
    TGT_emotion = tf.convert_to_tensor(TGT_emotion, tf.int32)
    Speakers = tf.convert_to_tensor(Speakers, tf.int32)
    A = tf.convert_to_tensor(A, tf.float32)

    # Create Queues
    input_queues = tf.train.slice_input_producer([X, X_image, Y_image, X_length, Y, sources, targets, X_turn_number, SRC_emotion, TGT_emotion, Speakers, A])
            
    # create batch queues
    x, x_image, y_image, x_length, y, sources, targets, X_turn_number, SRC_emotion, TGT_emotion, Speakers, A = tf.train.shuffle_batch(input_queues,
                                num_threads=8,
                                batch_size=hp.batch_size, 
                                capacity=hp.batch_size*64,   
                                min_after_dequeue=hp.batch_size*32, 
                                allow_smaller_final_batch=False)

    return x, x_image, y_image, x_length, y, num_batch ,sources,targets, X_turn_number, SRC_emotion, TGT_emotion, Speakers, A # (N, T), (N, T), ()


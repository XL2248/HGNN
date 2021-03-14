#coding=utf-8
import sys
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import pickle, code, re, collections
# import nltk
from sacremoses import MosesTokenizer
# from nltk.book import *
# 正则表达式过滤特殊符号用空格符占位，双引号、单引号、句点、逗号
pat_letter = re.compile(r'[^a-zA-Z0-9 \/\'\,\.\!\$\?\-\'\"\(\)\+\~\=\%]+')
# 还原常见缩写单词
pat_is = re.compile("(it|he|she|that|this|there|here)(\'s)", re.I)
pat_s = re.compile("(?<=[a-zA-Z])\'s") 
pat_s2 = re.compile("(?<=s)\'s?")
pat_not = re.compile("(?<=[a-zA-Z])n\'t") 
pat_would = re.compile("(?<=[a-zA-Z])\'d") 
pat_will = re.compile("(?<=[a-zA-Z])\'ll") 
pat_am = re.compile("(?<=[I|i])\'m") # 
pat_are = re.compile("(?<=[a-zA-Z])\'re") #
pat_ve = re.compile("(?<=[a-zA-Z])\'ve") #
pat_y = re.compile("(?<=[a-zA-Z])y\'") # 
pat_Y = re.compile("(?<=[a-zA-Z])Y\'") #
# filepath = 'data_emotion.p'
# data, W, vocab, word_idx_map, max_sentence_length, label_index = pickle.load(open(filepath, 'rb'))
# code.interact(local=locals())

import csv
data_type=sys.argv[1]
def replace_abbreviations(text):
    new_text = text
    new_text = pat_letter.sub(' ', text).strip().lower()
    # new_text = pat_is.sub(r"\1 's", new_text)
    # new_text = pat_s.sub("", new_text)
    # new_text = pat_s2.sub("", new_text)
    # new_text = pat_not.sub(" not", new_text)
    # new_text = pat_would.sub(" would", new_text)
    # new_text = pat_will.sub(" will", new_text)
    # new_text = pat_am.sub(" am", new_text)
    # new_text = pat_are.sub(" are", new_text)
    # new_text = pat_ve.sub(" have", new_text)
    # new_text = pat_y.sub("you ", new_text)
    # new_text = pat_Y.sub("you ", new_text)
    # new_text = new_text.replace('\'', ' \'')
    return new_text

def symmetric_normalization(A):
        d = np.array(A.sum(1)).flatten()
        d_inv = 1. / d
        d_inv[np.isinf(d_inv)] = 0.
        D_inv = sp.diags(d_inv)
        return D_inv.dot(A)

def readFileRows(filepath, dimension_size=17, normalization=True):
    word2embed = {}
    k = 0
    with open('speakers1.txt', 'r') as fopen:
        for line in fopen:
            w = line.split('\n')[0]
            word2embed[w] = k
            k += 1
    #code.interact(local=locals())
    emotion2idx = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}
    
    with open(filepath, 'r') as f:
        # file = csv.reader(f)
        # for line in file:
        #     print(line)
        reader = csv.DictReader(f)
        Utterance = []
        Speaker = []
        Emotion = []
        Sentiment = []
        Dialogue_ID = []
        Utterance_ID = []
        for row in reader:
            Utterance.append(row['Utterance'])
            Speaker.append(row['Speaker'])
            Emotion.append(row['Emotion'])
            Sentiment.append(row['Sentiment'])
            Dialogue_ID.append(row['Dialogue_ID'])
            Utterance_ID.append(row['Utterance_ID'])
        dialogue = []
        speaker = []
        emotion = []
        index = -1
        tk = MosesTokenizer()
        count = 0
        adj_matrix = []
        #code.interact(local=locals())
        print('complete dialogue number, utterance number',len(set(Dialogue_ID)), len(Dialogue_ID))
        for idx in range(int(Dialogue_ID[-1])+1):
            for D_id in Dialogue_ID:
                #code.interact(local=locals())
                if D_id == str(idx): # idx-th dialogue.
                    index += 1
                    dialogue.append(' '.join(tk.tokenize(replace_abbreviations(Utterance[index]))))
                    if replace_abbreviations(Speaker[index]) not in word2embed:
                        speaker.append('newer')
                    else:
                        speaker.append(replace_abbreviations(Speaker[index]))
                    #speaker.append(Speaker[index])
                    emotion.append(Emotion[index])
                else:
                    if len(dialogue) < 2:
                        dialogue, speaker, emotion = [], [], []
                        continue;

                    for k in range(len(dialogue)-1):

                        adj_matrices = []
                        adj_matrix_speaker_text = sp.lil_matrix((90, 90), dtype='int8')
                        adj_matrix_speaker_ima = sp.lil_matrix((90, 90), dtype='int8')
                        adj_matrix_text_ima = sp.lil_matrix((90, 90), dtype='int8')
                        adj_matrix_utt_utt = sp.lil_matrix((90, 90), dtype='int8')
                        adj_matrix_speaker_speaker = sp.lil_matrix((90, 90), dtype='int8')
                        adj_matrix_emotion_text = sp.lil_matrix((90, 90), dtype='int8')
                        adj_matrix_emotion_ima = sp.lil_matrix((90, 90), dtype='int8')
                        for j in range(k+1):# iter dialogue[0:k]
                            adj_matrix_speaker_text[j, 70+word2embed[speaker[j]]] = 1
                            adj_matrix_speaker_text[70+word2embed[speaker[j]], j] = 1
                            adj_matrix_speaker_ima[j+35, 70+word2embed[speaker[j]]] = 1
                            adj_matrix_speaker_ima[70+word2embed[speaker[j]], j+35] = 1
                            adj_matrix_text_ima[j, j+35] = 1
                            adj_matrix_text_ima[j+35, j] = 1
                            adj_matrix_emotion_text[j, 83+emotion2idx[emotion[j]]] = 1
                            adj_matrix_emotion_text[83+emotion2idx[emotion[j]], j] = 1
                            adj_matrix_emotion_ima[j, 83+emotion2idx[emotion[j]]] = 1
                            adj_matrix_emotion_ima[83+emotion2idx[emotion[j]], j] = 1
                            for i in range(j, k+1):
                                adj_matrix_speaker_speaker[70+word2embed[speaker[j]], 70+word2embed[speaker[i]]] = 1
                                adj_matrix_speaker_speaker[70+word2embed[speaker[i]], 70+word2embed[speaker[j]]] = 1
                                if speaker[i] == speaker[j]:
                                    adj_matrix_utt_utt[j, i] = 1
                                    #print(i+35)
                                    adj_matrix_utt_utt[j, i+35] = 1
                                    adj_matrix_utt_utt[j+35, i] = 1
                                    adj_matrix_utt_utt[j+35, i+35] = 1
                                elif i - j == 1:
                                    adj_matrix_utt_utt[j, i] = 1
                                    adj_matrix_utt_utt[j, i+35] = 1
                                    adj_matrix_utt_utt[j+35, i] =1
                                    adj_matrix_utt_utt[j+35, i+35] = 1
                                elif emotion[i] == emotion[j]:
                                    adj_matrix_utt_utt[j, i] = 1
                                    adj_matrix_utt_utt[j, i+35] = 1
                                    adj_matrix_utt_utt[j+35, i] =1
                                    adj_matrix_utt_utt[j+35, i+35] = 1
                                
                            adj_matrix_speaker_text = adj_matrix_speaker_text.tocsr()
                            adj_matrix_speaker_ima = adj_matrix_speaker_ima.tocsr()
                            adj_matrix_text_ima = adj_matrix_text_ima.tocsr()
                            adj_matrix_utt_utt = adj_matrix_utt_utt.tocsr()
                            adj_matrix_speaker_speaker = adj_matrix_speaker_speaker.tocsr()
                            adj_matrix_emotion_ima = adj_matrix_emotion_ima.tocsr()
                            adj_matrix_emotion_text = adj_matrix_emotion_text.tocsr()
                        #if normalization:
                        #    adj_matrix_speaker = symmetric_normalization(adj_matrix_speaker)
                        #    adj_matrix_adj = symmetric_normalization(adj_matrix_adj)
                        #    adj_matrix_emotion = symmetric_normalization(adj_matrix_emotion)
                        adj_matrices.append(adj_matrix_speaker_text)
                        adj_matrices.append(adj_matrix_speaker_ima)
                        adj_matrices.append(adj_matrix_text_ima)
                        adj_matrices.append(adj_matrix_utt_utt)
                        adj_matrices.append(adj_matrix_speaker_speaker)
                        adj_matrices.append(adj_matrix_emotion_text)
                        adj_matrices.append(adj_matrix_emotion_ima)
                        #code.interact(local=locals())
                        adj_matrix.append(adj_matrices)

                    dialogue = []
                    speaker = []
                    emotion = []
                    #code.interact(local=locals())
        print('count=',count)            # break;
        print('index=',index)
        pkl.dump(adj_matrix, open(data_type+'.pkl', 'wb'), protocol=-1)
        print(len(adj_matrix))

filepath = data_type+'_sent_emo.csv'
readFileRows(filepath)


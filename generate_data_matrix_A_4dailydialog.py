#coding=utf-8
import sys
import pickle as pkl
import scipy.sparse as sp
import numpy as np
import pickle, code, re, collections,json
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

def readFileRows(filepath, dimension_size=17):
    '''
    word2embed = {}
    with open(data_type+'.txt', 'r') as fopen:
        for line in fopen:
            w = line.strip().split()
            word2embed[' '.join(w[:-dimension_size])] = w[-dimension_size:]
            #code.interact(local=locals())
    '''
    emotion2idx = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger': 6}
    emotion2count = {'no_emotion': 0, 'surprise': 0, 'fear': 0, 'sadness': 0, 'happiness': 0, 'disgust': 0, 'anger': 0}
    tk = MosesTokenizer()
    max_turn = 0
    dialogue_num = 0
    adj_matrix = []
    length_dict = []
    with open(filepath, 'r') as f:
        # file = csv.reader(f)
        for line in f.readlines():
            dialogue_num += 1
            temp = json.loads(line)
            dialogue, speaker, emotion = [], [], []
            for index in range(len(temp['dialogue'])):
                dialogue.append(' '.join(tk.tokenize(replace_abbreviations(temp['dialogue'][index]['text']))))
                if index % 2 == 0:
                    speaker.append('A')
                else:
                    speaker.append('B')
                if temp['dialogue'][index]['emotion'] == 'no_emotion':
                    emotion.append('neutral')
                elif temp['dialogue'][index]['emotion'] == 'happiness':
                    #print(temp['dialogue'][index]['emotion'])
                    emotion.append('joy')
                else:
                    emotion.append(temp['dialogue'][index]['emotion'])
                emotion2count[temp['dialogue'][index]['emotion']] += 1
            if len(temp['dialogue']) > max_turn:
                max_turn = len(temp['dialogue'])
#            code.interact(local=locals())
            with open(filepath_w_query, 'a') as f_q, open(filepath_w_answer, 'a') as f_a:
                for k in range(len(dialogue)-1):
                    adj_matrices = []
                    adj_matrix_utt_utt = sp.lil_matrix((42, 42), dtype='int8')
                    adj_matrix_emotion_text = sp.lil_matrix((42, 42), dtype='int8')
                    adj_matrix_adj = sp.lil_matrix((42, 42), dtype='int8')
                    length_dict.append(k+1)
                    for j in range(k+1):
                        adj_matrix_emotion_text[j, 35+emotion2idx[emotion[j]]] = 1
                        adj_matrix_emotion_text[35+emotion2idx[emotion[j]], j] = 1
                        for i in range(j, k+1):
                            if i - j == 1:
                                adj_matrix_utt_utt[j, i] = 1
                                adj_matrix_utt_utt[i, j] = 1

                            elif emotion[i] == emotion[j]:
                                adj_matrix_utt_utt[j, i] = 1
                                adj_matrix_utt_utt[i, j] = 1

                        adj_matrix_utt_utt = adj_matrix_utt_utt.tocsr()
                        adj_matrix_emotion_text = adj_matrix_emotion_text.tocsr()
                        f_q.write(speaker[j] + '\t\t' +dialogue[j] + '\t\t' + emotion[j] + ' </d> ')
                    f_q.write('\n')
                    f_a.write(speaker[k+1] + '\t' + dialogue[k+1] + '\t' + str(emotion2idx[emotion[k+1]]))
                    f_a.write('\n')
                    adj_matrices.append(adj_matrix_utt_utt)
                    adj_matrices.append(adj_matrix_emotion_text)
                    adj_matrix.append(adj_matrices)
            dialogue = []
            speaker = []
            emotion = []

    print('max_turn=',max_turn)
    print('dialogue_num=',dialogue_num)
    print('emotion=', emotion2count)
    pkl.dump(adj_matrix, open(data_type+'.pkl', 'wb'), protocol=-1)
    print(len(adj_matrix))
    print('len(length_dict)', len(length_dict))
    max_length = 60#length_dict.sort(reverse=True)[0]
    len_dict = {}
    for i in range(max_length):
        len_dict[i] = 0
    for item in length_dict:#.sort(reverse=True):
        for j in range(item):
           len_dict[j] += 1
    print(len_dict)


filepath = data_type+'.json'
filepath_w_query = data_type+'_query.txt'
filepath_w_answer = data_type+'_answer.txt'
filepath_w_image = data_type+'_image.txt'
readFileRows(filepath)


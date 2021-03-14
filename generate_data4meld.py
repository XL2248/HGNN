#coding=utf-8
import sys
import numpy as np
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

def readFileRows(filepath, dimension_size=17):
    word2embed = {}
    with open(data_type+'.txt', 'r') as fopen:
        for line in fopen:
            w = line.strip().split()
            word2embed[' '.join(w[:-dimension_size])] = w[-dimension_size:]
            #code.interact(local=locals())
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
        print('complete dialogue number, utterance number',len(set(Dialogue_ID)), len(Dialogue_ID))
        for idx in range(int(Dialogue_ID[-1])+1):
            for D_id in Dialogue_ID:
                #code.interact(local=locals())
                if D_id == str(idx): # idx-th dialogue.
                    index += 1
                    dialogue.append(' '.join(tk.tokenize(replace_abbreviations(Utterance[index]))))
                    speaker.append(replace_abbreviations(Speaker[index]))
                    emotion.append(Emotion[index])

                else:
                    if len(dialogue) < 2:
                        dialogue, speaker, emotion = [], [], []
                        continue;

                    with open(filepath_w_query, 'a') as f_q, open(filepath_w_answer, 'a') as f_a, open(filepath_w_image, 'a') as f_i:
                        for k in range(len(dialogue)-1):

                            for j in range(k+1):
                                f_q.write(speaker[j] + '\t\t' +dialogue[j] + '\t\t' + emotion[j] + ' </d> ')
                                dia_utt = 'dia'+str(idx)+'_utt'+str(j)
                                if dia_utt not in word2embed.keys():
                                    count += 1
                                    temp = np.random.uniform(-0.25, 0.25, 17)
                                    word2embed[dia_utt] = [str(item) for item in temp.tolist()]
                                f_i.write(' '.join(word2embed[dia_utt])+'\t\t')

                            dia_utt = 'dia'+str(idx)+'_utt'+str(k+1)
                            if dia_utt not in word2embed.keys():
                                temp = np.random.uniform(-0.25, 0.25, 17)
                                word2embed[dia_utt] = [str(item) for item in temp.tolist()]
                            f_i.write(' '.join(word2embed[dia_utt])+'\t\t')
                            f_q.write('\n')
                            f_i.write('\n')
                            f_a.write(speaker[k+1] + ' </d> ' + dialogue[k+1] + ' </d> ' + emotion[k+1] + ' </d> ')
                            f_a.write('\n')
                    dialogue = []
                    speaker = []
                    emotion = []
                    #code.interact(local=locals())
        print('count=',count)            # break;
        print('index=',index)


filepath = data_type+'_sent_emo.csv'
filepath_w_query = data_type+'_query.txt'
filepath_w_answer = data_type+'_answer.txt'
filepath_w_image = data_type+'_image.txt'
readFileRows(filepath)


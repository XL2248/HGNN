 #coding=utf-8
import numpy as np
import pickle, code, re, collections, codecs
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
from collections import Counter
import csv

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

def readFileRows(filepath, speakers, Speaker):
    word2embed = {}
    with open(filepath, 'r') as f:
        # file = csv.reader(f)
        # for line in file:
        #     print(line)
        reader = csv.DictReader(f)
        Utterance = []
        #Speaker = []
        Emotion = []
        Sentiment = []
        Dialogue_ID = []
        Utterance_ID = []
        for row in reader:
            Utterance.append(row['Utterance'])
            Speaker.append(replace_abbreviations(row['Speaker']))
            ##speakers.add(replace_abbreviations(row['Speaker']))
            Emotion.append(row['Emotion'])
            Sentiment.append(row['Sentiment'])
            Dialogue_ID.append(row['Dialogue_ID'])
            Utterance_ID.append(row['Utterance_ID'])
        #print('count=',count)            # break;

def allNum(word):
    allNum=True
    for ww in word:
        if ww>'9' or ww<'0':
            allNum=False
            break
    return allNum
        # code.interact(local=locals())
speakers = set()
Speaker = []
filepath = 'train_sent_emo.csv'
readFileRows(filepath, speakers, Speaker)
filepath = 'test_sent_emo.csv'
readFileRows(filepath, speakers, Speaker)
filepath = 'dev_sent_emo.csv'
readFileRows(filepath, speakers, Speaker)
speaker2cnt = Counter(Speaker)
with codecs.open('speakers.txt', 'w', 'utf-8') as fout:
    fout.write('newer\n')
    #for item in speakers:
    for word, cnt in speaker2cnt.most_common(len(speaker2cnt)):
        #fout.write(item+'\n')
        if allNum(word)==False and cnt > 21:#30-20 21-30
            #fout.write(u"{}\t{}\n".format(word, cnt))
            fout.write(u"{}\n".format(word))
print('count=',len(Speaker))

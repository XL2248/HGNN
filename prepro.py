# -*- coding: utf-8 -*-
#/usr/bin/python2
from __future__ import print_function
from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import os, code
#import regex
from collections import Counter

def allNum(word):
    allNum=True
    for ww in word:
        if ww>'9' or ww<'0':
            allNum=False
            break
    return allNum
def make_vocab_src(fpath, fname):
    '''Constructs vocabulary.
    
    Args:
      fpath: A string. Input file path.
      fname: A string. Output file name.
    
    Writes vocabulary line by line to `preprocessed/fname`
    '''  
    file1 = codecs.open(fpath, 'r', 'utf-8')
    words = []
    for line in file1.readlines():
        for content_emotion in line.split("</d>"):
            #code.interact(local=locals())
            if len(content_emotion.split("\t\t")) > 1:
                content = content_emotion.split("\t\t")[1]
                words.extend(content.split())
    #words = text.split()
    word2cnt = Counter(words)
    if not os.path.exists('preprocessed'): os.mkdir('preprocessed')
    with codecs.open('preprocessed/{}'.format(fname), 'w', 'utf-8') as fout:
        fout.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>", "</S>"))
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            if allNum(word)==False:
                fout.write(u"{}\t{}\n".format(word, cnt))

def make_vocab_tgt(fpath, fname, fname2):
    '''Constructs vocabulary.
    
    Args:
      fpath: A string. Input file path.
      fname: A string. Output file name.
    
    Writes vocabulary line by line to `preprocessed/fname`
    '''
    words = []
    speakers = []
    file1 = codecs.open(fpath, 'r', 'utf-8')
    for line in file1.readlines():
        #code.interact(local=locals())
        words.extend(line.split(u"</d>")[1].split())
        speakers.append(' '.join(line.split(u"</d>")[0].split()))

    word2cnt = Counter(words)
    speaker2cnt = Counter(speakers)
    if not os.path.exists('preprocessed'): os.mkdir('preprocessed')
    with codecs.open('preprocessed/{}'.format(fname), 'w', 'utf-8') as fout:
        fout.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>", "</S>"))
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            if allNum(word)==False:
                fout.write(u"{}\t{}\n".format(word, cnt))

    with codecs.open('preprocessed/{}'.format(fname2), 'w', 'utf-8') as fout:
        fout.write("{}\t1000000000\n".format("<newer>"))
        for word, cnt in speaker2cnt.most_common(len(speaker2cnt)):
            if allNum(word)==False:
                fout.write(u"{}\t{}\n".format(word, cnt))

if __name__ == '__main__':
    make_vocab_src(hp.source_train, "de.vocab.tsv")
    make_vocab_tgt(hp.target_train, "en.vocab.tsv", "speaker.name.tsv")
    print("Done")

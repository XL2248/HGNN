# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import print_function
import codecs
import os
import argparse
import tensorflow as tf
import numpy as np
from nltk.collocations import BigramCollocationFinder
from nltk.probability import FreqDist
import math
#from hyperparams import Hyperparams as hp
from data_load import load_test_data, load_de_vocab, load_en_vocab
from train import Graph
from nltk.translate.bleu_score import corpus_bleu
def cal_Distinct(corpus):
    """
    Calculates unigram and bigram diversity
    Args:
        corpus: tokenized list of sentences sampled
    Returns:
        uni_diversity: distinct-1 score
        bi_diversity: distinct-2 score
    """
    bigram_finder = BigramCollocationFinder.from_words(corpus)
    bi_diversity = len(bigram_finder.ngram_fd) / bigram_finder.N

    dist = FreqDist(corpus)
    uni_diversity = len(dist) / len(corpus)

    return uni_diversity, bi_diversity


def cal_BERTScore(refer, candidate):
    _, _, bert_scores = score(candidate, refer, 
                              bert="bert-base-uncased", no_idf=True)
    bert_scores = bert_scores.tolist()
    bert_scores = [0.5 if math.isnan(score) else score for score in bert_scores]
    return np.mean(bert_scores)


def cal_acc_f1(tp, fn, fp, tn):
    # return (macro-f1, micro-f1, Acc)
    acc = (tp + tn) / (tp + fn + fp + tn)
    precision_p, precision_n = tp / (tp + fp), tn / (tn + fn)
    recall_p, recall_n = tp / (tp + fn), tn / (tn + fp)
    avg_pre, avg_recall = (precision_n + precision_p) / 2, (recall_p + recall_n) / 2
    macro_f1 = 2 * avg_pre * avg_recall / (avg_pre + avg_recall)
    mi_pre = (tp + tn) / (tp + fp + tn + fn)
    mi_rec = (tp + tn) / (tp + fn + tn + fp)
    micro_f1 = 2 * mi_pre * mi_rec / (mi_pre + mi_rec)
    return macro_f1, micro_f1, acc


def cal_acc_P_R_F1(tp, fn, fp, tn):
    # cal the F1 metric from the stat data of the postive label
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    acc = (tp + tn) / (tp + fn + fp + tn)
    return round(precision, 4), round(recall, 4), round(f1, 4), round(acc, 4)

def eval(hp): 
    # Load graph
    g = Graph(hp=hp, is_training=False)
    print("Graph loaded")
    
    # Load data

    X, X_image, X_length, Y, Sources, Targets, X_turn_number, SRC_emotion, TGT_emotion, Speakers, A = load_test_data(hp)
    #print(X)
    de2idx, idx2de = load_de_vocab(hp)
    en2idx, idx2en = load_en_vocab(hp)
     
     
    # Start session         
    with g.graph.as_default():    
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ## Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")
              
            ## Get model name
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1] # model name
            #fftmp=open("tmp.txt","w")
            ## Inference
            if not os.path.exists('results'): os.mkdir('results')
            with codecs.open("results/" + mname, "w", "utf-8") as fout:
                list_of_refs, hypotheses, test_loss = [], [], []
                for i in range(len(X) // hp.batch_size):
                     
                    ### Get mini-batches
                    x = X[i*hp.batch_size: (i+1)*hp.batch_size]
                    x_length=X_length[i*hp.batch_size: (i+1)*hp.batch_size]
                    y = Y[i*hp.batch_size: (i+1)*hp.batch_size]
                    x_emotion = SRC_emotion[i*hp.batch_size: (i+1)*hp.batch_size]
                    speaker = Speakers[i*hp.batch_size: (i+1)*hp.batch_size]
                    x_image = X_image[i*hp.batch_size: (i+1)*hp.batch_size]
                    a = A[i*hp.batch_size: (i+1)*hp.batch_size]
                    sources = Sources[i*hp.batch_size: (i+1)*hp.batch_size]
                    targets = Targets[i*hp.batch_size: (i+1)*hp.batch_size]
                    eval_bath = sess.run(g.mean_loss, {g.x: x, g.x_image: x_image, g.x_length:x_length,g.y: y, g.x_emotion: x_emotion, g.speaker: speaker, g.A: a, g.x_turn_number: x_turn_number})
                    test_loss.append( eval_bath)

                    ### Autoregressive inference
                    preds = np.zeros((hp.batch_size, hp.maxlen), np.int32)
                    for j in range(hp.maxlen):
                        _preds = sess.run(g.preds, {g.x: x,g.x_length:x_length, g.y: preds})
                        preds[:, j] = _preds[:, j]
                     
                    ### Write to file
                    for source, target, pred in zip(sources, targets, preds): # sentence-wise
                        got = " ".join(idx2en[idx] for idx in pred).split("</S>")[0].strip()
                        fout.write("- source: " + source +"\n")
                        fout.write("- expected: " + target + "\n")
                        fout.write("- got: " + got + "\n\n")
                        fout.flush()
                          
                        # bleu score
                        #ref = target.split()
                        ref = target.split(u"</d>")[1].split()
                        hypothesis = got.split()
                        if len(ref) > 3 and len(hypothesis) > 3:
                            list_of_refs.append([ref])
                            hypotheses.append(hypothesis)
              
                ## Calculate bleu score
                score = corpus_bleu(list_of_refs, hypotheses)
                fout.write("Test Bleu Score = " + str(100*score))
                print("Test Bleu Score = " + str(100*score))
                print("eval PPL = %.5lf"%(round(math.exp(np.mean(test_loss)), 4)))
                print("eval loss = %.5lf"%(np.mean(test_loss)))
                # Distinct-1, Distinct-2
                candidates = []
                for line in hypotheses:
                   candidates.extend(line)
                distinct_1, distinct_2 = cal_Distinct(candidates)
                print('Distinct-1:' + str(round(distinct_1, 4)) + 'Distinct-2:' + str(round(distinct_2, 4)))
                                          
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translate script')
    base_dir = 'your_data_path'
    parser.add_argument('--source_train', type=str, default=base_dir + 'corpora/train_query.txt', help='src train file')
    parser.add_argument('--target_train', type=str, default=base_dir + 'corpora/train_answer.txt', help='src train file')
    parser.add_argument('--source_test', type=str, default=base_dir + 'corpora/test_query.txt', help='src test file')
    parser.add_argument('--target_test', type=str, default=base_dir + 'corpora/test_answer.txt', help='tgt test file')
    parser.add_argument('--source_dev', type=str, default=base_dir + 'corpora/dev_query.txt', help='src dev file')
    parser.add_argument('--target_dev', type=str, default=base_dir + 'corpora/dev_answer.txt', help='tgt dev file')
    parser.add_argument('--logdir', type=str, default='logdir2020_test', help='logdir')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout ratio')
    parser.add_argument('--hidden_units', type=int, default=512,
                        help='context encoder hidden size')
    parser.add_argument('--num_blocks', type=int, default=6, help='num_blocks')
    parser.add_argument('--num_heads', type=int, default=8, help='num_heads')
    parser.add_argument('--maxlen', type=int, default=50, help='maxlen')
    parser.add_argument('--min_cnt', type=int, default=1, help='min_cnt')
    parser.add_argument('--num_epochs', type=int, default=20000, help='num_epochs')
    parser.add_argument('--num_layers', type=int, default=1, help='num_layers')
    parser.add_argument('--max_turn', type=int, default=33, help='max_turn')
    parser.add_argument('--sinusoid', dest='sinusoid', action='store_true')
    hp = parser.parse_args()
    print('[!] Parameters:')
    print(hp)
    eval(hp)
    print("Done")
    
    

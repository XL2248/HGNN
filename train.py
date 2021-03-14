# -*- coding: utf-8 -*-
#/usr/bin/python2
from __future__ import print_function
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')   
import tensorflow as tf
from keras.layers import Dropout
#from hyperparams import Hyperparams as hyps
from data_load import get_batch_data, load_de_vocab, load_en_vocab, load_dev_data, load_test_data
from modules import *
from graph import HGraph
from nn import linear, maxout, smoothed_softmax_cross_entropy_with_logits
from evaluation import *
import os, codecs, code
#from tqdm import tqdm
from nltk.collocations import BigramCollocationFinder
from nltk.probability import FreqDist
import numpy as np
#import codecs
import nltk,math
from nltk.translate.bleu_score import corpus_bleu
import argparse
from sklearn.metrics import classification_report,precision_recall_fscore_support,confusion_matrix,accuracy_score
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def cal_Distinct(corpus):
    """
    Calculates unigram and bigram diversity
    args:
        corpus: tokenized list of sentences sampled
    Returns:
        uni_diversity: distinct-1 score
        bi_diversity: distinct-2 score
    """
    bigram_finder = BigramCollocationFinder.from_words(corpus)
    if bigram_finder.N == 0:
        bi_diversity = 0
    else:
        bi_diversity = len(bigram_finder.ngram_fd)*0.1 / bigram_finder.N * 10
    if len(corpus) == 0:
        uni_diversity = 0
    else:
        dist = FreqDist(corpus)
        uni_diversity = len(dist)*0.1 / len(corpus) * 10

    return uni_diversity, bi_diversity

def layer_norm(input_tensor, name=None):
  """Run layer normalization on the last dimension of the tensor."""
  return tf.contrib.layers.layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)

class Graph():
    def __init__(self, hp, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.x, self.x_image, self.y_image, self.x_length, self.y, self.num_batch, self.source, self.target, self.x_turn_number, self.x_emotion, self.y_emotion, self.speaker, self.A = get_batch_data(hp) # (N, T)
            else: # inference
                self.x = tf.placeholder(tf.int32, shape=(None, hp.max_turn, hp.maxlen)) # shape=(16, 15, 50)
                self.x_image = tf.placeholder(tf.float32, shape=(None, hp.max_turn, 17))
                self.y_image = tf.placeholder(tf.float32, shape=(None, 17))
                self.x_length = tf.placeholder(tf.int32, shape=(None, hp.max_turn))
                self.y = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
                self.x_emotion = tf.placeholder(tf.int32, shape=(None, hp.max_turn))
                self.y_emotion = tf.placeholder(tf.int32, shape=(None, ))
                self.speaker = tf.placeholder(tf.int32, shape=(None, ))
                self.A = tf.placeholder(tf.float32, shape=(None, 7, 90, 90))
                self.x_turn_number = tf.placeholder(tf.int32, shape=(None, ))

            # define decoder inputs
            self.decoder_inputs = tf.concat((tf.ones_like(self.y[:, :1])*2, self.y[:, :-1]), -1) # 2:<S>

            # Load vocabulary    
            # de2idx, idx2de = load_de_vocab(hp)
            en2idx, idx2en = load_en_vocab(hp)
            speaker_memory = tf.get_variable('speaker_memory',
                                       dtype=tf.float32,
                                       shape=[13, hp.hidden_units],
                                       initializer=tf.contrib.layers.xavier_initializer())
            emotion_memory = tf.get_variable('emotion_memory',
                                       dtype=tf.float32,
                                       shape=[7, hp.hidden_units],
                                       initializer=tf.contrib.layers.xavier_initializer())
            outputs_speaker = tf.nn.embedding_lookup(speaker_memory, self.speaker)
            outputs_speaker_ = tf.tile(tf.expand_dims(outputs_speaker, 1), [1, 50, 1])
            # Encoder
            with tf.variable_scope("encoder"):
                ## Embedding
                embeddingsize = hp.hidden_units/2
                self.enc_embed = embedding(tf.reshape(self.x,[-1,hp.maxlen]), #batch_size*max_turn=240 shape=(240, 50, 256)
                                      vocab_size=len(en2idx), 
                                      num_units=embeddingsize, 
                                      scale=True,
                                      scope="enc_embed")
                single_cell = tf.nn.rnn_cell.GRUCell(hp.hidden_units)
                self.rnn_cell = tf.nn.rnn_cell.MultiRNNCell([single_cell]*hp.num_layers)
                print (self.enc_embed.get_shape())
                self.sequence_length=tf.reshape(self.x_length,[-1])
                print(self.sequence_length.get_shape())
                self.uttn_outputs, self.uttn_states = tf.nn.dynamic_rnn(cell=self.rnn_cell, inputs=self.enc_embed,sequence_length=self.sequence_length, dtype=tf.float32,swap_memory=True)

                print(hp.batch_size, hp.max_turn, hp.hidden_units)

                self.enc = tf.reshape(self.uttn_states,[hp.batch_size,hp.max_turn,hp.hidden_units]) #shape=(16, 15, 512)

                ## Positional Encoding
                if hp.sinusoid:
                    self.enc += positional_encoding(self.x,
                                      num_units=hp.hidden_units, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="enc_pe")
                else:
                    self.enc += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]), 0), [tf.shape(self.x)[0], 1]),
                                      vocab_size=hp.maxlen, 
                                      num_units=hp.hidden_units, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="enc_pe")
                    
                 
                ## Dropout
                self.enc = tf.layers.dropout(self.enc,#shape=(32, 15, 512), 
                                            rate=hp.dropout_rate, 
                                            training=tf.convert_to_tensor(is_training))
                print('self.enc=', self.enc)
                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ### Multihead Attention
                        self.enc,_ = multihead_attention(queries=self.enc, 
                                                        keys=self.enc, 
                                                        num_units=hp.hidden_units, 
                                                        num_heads=hp.num_heads, 
                                                        dropout_rate=hp.dropout_rate,
                                                        is_training=is_training,
                                                        causality=False)
                        
                        ### Feed Forward
                        self.enc = feedforward(self.enc, num_units=[4*hp.hidden_units, hp.hidden_units])#shape=(32, 15, 512),
                        #code.interact(local=locals())
            matrix = tf.get_variable("transform", [self.x_image.shape.as_list()[-1], self.enc.shape.as_list()[-1]], dtype=tf.float32)
            self.x_ima = tf.map_fn(lambda x: tf.matmul(x, matrix), self.x_image, dtype=tf.float32)
            #code.interact(local=locals())
            self.enc = tf.concat((self.enc, self.x_ima), -2)
            s_m = tf.tile(tf.expand_dims(speaker_memory, 0), [hp.batch_size, 1, 1])
            e_m = tf.tile(tf.expand_dims(emotion_memory, 0), [hp.batch_size, 1, 1])
            self.enc = tf.concat((self.enc, e_m), -2)
            self.enc = tf.concat((self.enc, s_m), -2)
            self.H1 = HGraph(256, activation='relu')([self.enc, self.A])
            self.H1 = Dropout(hp.dropout_rate)(self.H1)
            self.H2 = HGraph(256, activation='relu')([self.H1, self.A])
            self.enc = Dropout(hp.dropout_rate)(self.H2)
            self.enc = tf.map_fn(lambda x: x, self.enc, dtype=tf.float32)
            self.enc = feedforward(self.enc, num_units=[4*hp.hidden_units, hp.hidden_units])
            with tf.variable_scope("emotion"):
               text_image = tf.map_fn(lambda x: x[:-1], self.enc, dtype=tf.float32)
               x3 = tf.reduce_max(text_image, axis = 1)
               self.emotion_logits = linear(x3, 7, True, False, scope="softmax")
               outputs_emotion = tf.matmul(self.emotion_logits, emotion_memory)
               outputs_emotion_ = tf.tile(tf.expand_dims(outputs_emotion, 1), [1, 50, 1])#shape=(50, 50, 128)

            # Decoder
            with tf.variable_scope("decoder"):
                ## Embedding
                self.dec = embedding(self.decoder_inputs, 
                                      vocab_size=len(en2idx), 
                                      num_units=hp.hidden_units,
                                      scale=True, 
                                      scope="dec_embed")
                
                ## Positional Encoding
                if hp.sinusoid:
                    self.dec += positional_encoding(self.decoder_inputs,
                                      vocab_size=hp.maxlen, 
                                      num_units=hp.hidden_units, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="dec_pe")
                else:
                    self.dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs)[1]), 0), [tf.shape(self.decoder_inputs)[0], 1]),
                                      vocab_size=hp.maxlen, 
                                      num_units=hp.hidden_units, 
                                      zero_pad=False, 
                                      scale=False,
                                      scope="dec_pe")
                
                ## Dropout
                self.dec = tf.layers.dropout(self.dec, 
                                            rate=hp.dropout_rate, 
                                            training=tf.convert_to_tensor(is_training))
                print('self.dec', self.dec) #shape=(50, 50, 512)
                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ## Multihead Attention ( self-attention)
                        self.dec,_ = multihead_attention(queries=self.dec, 
                                                        keys=self.dec, 
                                                        num_units=hp.hidden_units, 
                                                        num_heads=hp.num_heads, 
                                                        dropout_rate=hp.dropout_rate,
                                                        is_training=is_training,
                                                        causality=True, 
                                                        scope="self_attention")
                        print('self.dec', self.dec)#shape=(50, 50, 512) 
                        ## Multihead Attention ( vanilla attention)
                        self.dec,self.attn = multihead_attention(queries=self.dec, 
                                                        keys=self.enc, 
                                                        num_units=hp.hidden_units, 
                                                        num_heads=hp.num_heads,
                                                        dropout_rate=hp.dropout_rate,
                                                        is_training=is_training, 
                                                        causality=False,
                                                        scope="vanilla_attention")
                        ## Feed Forward
                        print('self.dec', self.dec)#shape=(50, 50, 512)
                        self.dec = feedforward(self.dec, num_units=[4*hp.hidden_units, hp.hidden_units])
                        #code.interact(local=locals())

            self.dec_emo = tf.concat((outputs_emotion_, outputs_speaker_),-1)
            self.dec_emo_spe = tf.concat((self.dec, self.dec_emo),-1)
            g = tf.nn.sigmoid(layer_norm(linear(self.dec_emo_spe, 256, False, False,
                            scope="context_gate"),
                     name="context_gate_ln")) 

            self.dec_emo_spe = self.dec + g * outputs_emotion_ + (1 - g) * outputs_speaker_
            self.dec_emo_spe = tf.layers.dropout(self.dec_emo_spe,#shape=(32, 50, 512), 
                                            rate=hp.dropout_rate,
                                            training=tf.convert_to_tensor(is_training))
            # Final linear projection
            self.logits = tf.layers.dense(self.dec_emo_spe, len(en2idx))#shape=(128, 50, 5124)
            self.preds = tf.to_int32(tf.arg_max(self.logits, dimension=-1))#shape=(128, 50)
            self.istarget = tf.to_float(tf.not_equal(self.y, 0))
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y))*self.istarget)/ (tf.reduce_sum(self.istarget))
            tf.summary.scalar('acc', self.acc)
                
#            if is_training:  
                # Loss
            self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=len(en2idx)))
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)#shape=(256, 50)
            self.mean_loss = tf.reduce_sum(self.loss*self.istarget) / (tf.reduce_sum(self.istarget))

            if is_training:
                tgt_emotion = label_smoothing(tf.one_hot(self.y_emotion, depth=7))
                emotion_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.emotion_logits, labels=tgt_emotion)
                emotion_loss  = tf.reduce_mean(emotion_loss)
                # Training Scheme
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
                self.train_op = self.optimizer.minimize((1 - hp.alpha) * self.mean_loss + hp.alpha * emotion_loss, global_step=self.global_step)
                   
                # Summary 
                tf.summary.scalar('mean_loss', self.mean_loss + emotion_loss)
                self.merged = tf.summary.merge_all()

if __name__ == '__main__':                
    parser = argparse.ArgumentParser(description='Translate script')
    base_dir = os.getcwd() + '/'
    parser.add_argument('--source_train', type=str, default=base_dir + 'corpora/train_query.txt', help='src train file')
    parser.add_argument('--target_train', type=str, default=base_dir + 'corpora/train_answer.txt', help='src train file')
    parser.add_argument('--source_test', type=str, default=base_dir + 'corpora/test_query.txt', help='src test file')
    parser.add_argument('--target_test', type=str, default=base_dir + 'corpora/test_answer.txt', help='tgt test file')
    parser.add_argument('--source_dev', type=str, default=base_dir + 'corpora/dev_query.txt', help='src dev file')
    parser.add_argument('--target_dev', type=str, default=base_dir + 'corpora/dev_answer.txt', help='tgt dev file')
    parser.add_argument('--corpora_path', type=str, default=base_dir + 'corpora/', help='image file')
    parser.add_argument('--logdir', type=str, default='logdir2020_test', help='logdir')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout ratio')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
    parser.add_argument('--hidden_units', type=int, default=512, 
                        help='context encoder hidden size')
    parser.add_argument('--num_blocks', type=int, default=6, help='num_blocks')
    parser.add_argument('--num_heads', type=int, default=8, help='num_heads')
    parser.add_argument('--maxlen', type=int, default=50, help='maxlen')
    parser.add_argument('--min_cnt', type=int, default=1, help='min_cnt')
    parser.add_argument('--num_epochs', type=int, default=20000, help='num_epochs')
    parser.add_argument('--num_layers', type=int, default=1, help='num_layers')
    parser.add_argument('--max_turn', type=int, default=35, help='max_turn')
    parser.add_argument('--sinusoid', dest='sinusoid', action='store_true')
    hp = parser.parse_args()
    print('[!] Parameters:')
    print(hp)
    # Load vocabulary    
    # de2idx, idx2de = load_de_vocab(hp)
    en2idx, idx2en = load_en_vocab(hp)
    
    # Construct graph
    g = Graph(hp, "train"); print("Graph loaded")
    X, X_image, Y_image, X_length, Y, Sources, Targets, X_turn_number, SRC_emotion, TGT_emotion, Speakers, A = load_dev_data(hp)
    # Start session
    sv = tf.train.Supervisor(graph=g.graph, 
                             logdir=hp.logdir,
                             save_model_secs=0)
    #preEpoch= 
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    with sv.managed_session(config = tfconfig) as sess:
        early_break = 0
        old_eval_loss=10000
        for epoch in range(1, hp.num_epochs+1): 
            if sv.should_stop(): break
            loss=[]
            
            print('g.num_batch',g.num_batch)
            for step in range(g.num_batch):
                _,loss_step,attns,sources,targets = sess.run([g.train_op,g.mean_loss,g.attn,g.source,g.target])
                loss.append(loss_step)
                #step += 1
                if step%500==0:
                    gs = sess.run(g.global_step)
                    print("train loss:%.5lf\n"%(np.mean(loss)))
                    sv.saver.save(sess, hp.logdir + '/test_model_epoch_%02d_gs_%d' % (epoch, gs))

                    mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1]
                    print(mname)
                    eval_loss=[]
                    bleu=[]
                    pred_emotion=[]
                    for i in range(len(X) // hp.batch_size):
                       ### Get mini-batches
                       x = X[i*hp.batch_size: (i+1)*hp.batch_size]
                       x_length=X_length[i*hp.batch_size: (i+1)*hp.batch_size]
                       y = Y[i*hp.batch_size: (i+1)*hp.batch_size]
                       x_emotion = SRC_emotion[i*hp.batch_size: (i+1)*hp.batch_size]
                       speaker = Speakers[i*hp.batch_size: (i+1)*hp.batch_size]
                       sources = Sources[i*hp.batch_size: (i+1)*hp.batch_size]
                       targets = Targets[i*hp.batch_size: (i+1)*hp.batch_size]
                       x_image = X_image[i*hp.batch_size: (i+1)*hp.batch_size]
                       x_turn_number = X_turn_number[i*hp.batch_size: (i+1)*hp.batch_size]
                       a = A[i*hp.batch_size: (i+1)*hp.batch_size]
                       eval_bath = sess.run(g.mean_loss, {g.x: x, g.x_image: x_image, g.x_length:x_length,g.y: y, g.x_emotion: x_emotion, g.speaker: speaker, g.A: a, g.x_turn_number: x_turn_number})
                       eval_loss.append( eval_bath)
                       viterbi_sequence = sess.run(g.emotion_logits, {g.x: x, g.x_image: x_image, g.x_length:x_length, g.y: y, g.x_emotion: x_emotion, g.speaker: speaker, g.A: a, g.x_turn_number: x_turn_number})
                       pred_emotion.extend(viterbi_sequence.tolist())# = tf.map_fn(lambda x: x[0][x[1]], (viterbi_sequence, X_turn_number), dtype = tf.int32)

                       preds = np.zeros((hp.batch_size, hp.maxlen), np.int32)
                       for j in range(hp.maxlen):
                           _preds = sess.run(g.preds, {g.x: x, g.x_image: x_image, g.x_length:x_length, g.y: preds, g.x_emotion: x_emotion, g.speaker: speaker, g.A: a, g.x_turn_number: x_turn_number})
                           preds[:, j] = _preds[:, j]
                    
                       ### Write to file
                       list_of_refs, hypotheses, candidates = [], [], []
                       for source, target, pred in zip(sources, targets, preds): # sentence-wise
                           got = " ".join(idx2en[idx] for idx in pred).split("</S>")[0].strip()
                           # bleu score
                           ref = got.split()
                           #code.interact(local=locals())
                           #hypothesis = target.split()
                           hypothesis = target.split(u"</d>")[1].split()
                           score = nltk.translate.bleu_score.sentence_bleu([hypothesis],ref,(0.25, 0.25, 0.25, 0.25),nltk.translate.bleu_score.SmoothingFunction().method1)
                           bleu.append(score)
                           candidates.extend(ref)
                    #code.interact(local=locals())
                    true_label, predicted_label = [], []
                    correct_labels = 0
                    for index in range(len(pred_emotion)):
                        predicted_label.append(np.argmax(pred_emotion[index]))
                        true_label.append(TGT_emotion[index])

                    print(classification_report(true_label, predicted_label, digits=4))
                    print('Weighted FScore: \n ', precision_recall_fscore_support(true_label, predicted_label, average='weighted'))
                    print("eval PPL = %.5lf"%(round(math.exp(np.mean(eval_loss)), 4)))
                    print("eval loss = %.5lf"%(np.mean(eval_loss)))
                    print("Bleu Score = %.5lf"%(100*np.mean(bleu)))
                    #'''
                    # Distinct-1, Distinct-2
                    distinct_1, distinct_2 = cal_Distinct(candidates)
                    #print(f'Distinct-1: {round(distinct_1, 4)}; Distinct-2: {round(distinct_2, 4)}')
                    print('Distinct-1 = %.5lf'%(round(distinct_1, 4)))
                    print('Distinct-2 = %.5lf'%(round(distinct_2, 4)))
                    #'''
                    if np.mean(eval_loss) > old_eval_loss:
                        early_break +=1
                    else:
                        early_break = 0
                    old_eval_loss=np.mean(eval_loss)
                    #if early_break>=5:
                    #    break

            if epoch > 4:# test?
                evaluation(hp)
            #test
print("Done")    


from __future__ import print_function

from keras import activations, initializers
from keras import regularizers
from keras.engine import Layer
from keras.layers import Dropout
import code
import tensorflow as tf
import keras.backend as K


class HGraph(Layer):
    def __init__(self, output_dim,
                 init='glorot_uniform', activation='linear',
                 weights=None, W_regularizer=None,
                 b_regularizer=None, bias=True, 
                 self_links=True, consecutive_links=False, 
                 backward_links=False, edge_weighting=False, **kwargs):
        self.supports_masking = True
        self.init = initializers.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim  # number of features per node

        self.self_links = self_links
        self.consecutive_links = consecutive_links
        self.backward_links = backward_links
        self.edge_weighting = edge_weighting

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.bias = bias
        self.initial_weights = weights

        self.input_dim = None
        self.W = None
        self.b = None
        self.num_nodes = None
        self.num_features = None
        self.num_relations = None
        self.num_adjacency_matrices = None

        super(HGraph, self).__init__(**kwargs)

    def compute_output_shape(self, input_shapes):
        features_shape = input_shapes[0]
        output_shape = (None, features_shape[1], self.output_dim)
        return output_shape

    def build(self, input_shapes):
        features_shape = input_shapes[0]
        assert len(features_shape) == 3
        self.input_dim = features_shape[1]
        self.num_nodes = features_shape[1]
        self.num_features = features_shape[2]
        self.num_relations = input_shapes[1][1]

        self.num_adjacency_matrices = 4
        self.node_types = 4
        print('self.num_adjacency_matrices',self.num_adjacency_matrices) 
        if self.consecutive_links:
            self.num_adjacency_matrices += 1

        if self.backward_links:
            self.num_adjacency_matrices *= 2

        if self.self_links:
            self.num_adjacency_matrices += 1

        self.W = []
        self.B = []
        self.W_edges = []
        for i in range(self.node_types):
        #    for j in range(self.num_i
            self.W.append(self.add_weight((self.num_features, self.output_dim), # shape: (num_features, output_dim)
                                                    initializer=self.init,
                                                    name='{}_W_rel_{}'.format(self.name, i),
                                                    regularizer=self.W_regularizer))

            if self.edge_weighting:
                self.W_edges.append(self.add_weight((self.input_dim, self.num_features), # shape: (num_features, output_dim)
                                                        initializer='ones',
                                                        name='{}_W_edge_{}'.format(self.name, i),
                                                        regularizer=self.W_regularizer))

            self.B.append(self.add_weight((self.input_dim, self.output_dim),
                                            initializer='random_uniform',
                                            name='{}_b_{}'.format(self.name, i),
                                            regularizer=self.b_regularizer))

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        super(HGraph, self).build(input_shapes)
    def compute_mask(self, input_tensor, mask=None):
        # print("Graph:",mask)
        return mask
    def call (self, inputs, mask=None):
        features = inputs[0] # Shape: (None, num_nodes, num_features)
        batch_size = features.shape.as_list()[0]
        node_num = features.shape.as_list()[1]
        dim_size = features.shape.as_list()[2]
        
        A = inputs[1]  # Shapes: (None, num_nodes, num_nodes)
        #code.interact(local=locals())
        A = K.permute_dimensions(A, [1, 0, 2, 3])
        A_ = []
        for i in range(self.num_relations):
            A_.append(A[i])
        eye = A_[0] * K.zeros(self.num_nodes, dtype=tf.float32) + K.eye(self.num_nodes, dtype=tf.float32)
        tmp_u1 = K.ones((35, self.num_nodes), dtype=tf.float32)
        tmp_u0 = K.zeros((35, self.num_nodes), dtype=tf.float32)
        tmp_e1 = K.ones((7, self.num_nodes), dtype=tf.float32)
        tmp_e0 = K.zeros((7, self.num_nodes), dtype=tf.float32)
        tmp_s1 = K.ones((13, self.num_nodes), dtype=tf.float32)
        tmp_s0 = K.zeros((13, self.num_nodes), dtype=tf.float32)
        mask_u = tf.concat((tmp_u1, tmp_u0), -2)
        mask_u = tf.concat((mask_u, tmp_e0), -2)
        mask_u = tf.concat((mask_u, tmp_s0), -2)
        Mask = []
        Mask.append(tf.tile(tf.expand_dims(mask_u, 0),[batch_size, 1, 1]))
        mask_f = tf.concat((tmp_u0, tmp_u1), -2)
        mask_f = tf.concat((mask_f, tmp_e0), -2)
        mask_f = tf.concat((mask_f, tmp_s0), -2)
        Mask.append(tf.tile(tf.expand_dims(mask_f, 0),[batch_size, 1, 1]))
        mask_e = tf.concat((tmp_u0, tmp_u0), -2)
        mask_e = tf.concat((mask_e, tmp_e1), -2)
        mask_e = tf.concat((mask_e, tmp_s0), -2)
        Mask.append(tf.tile(tf.expand_dims(mask_e, 0),[batch_size, 1, 1]))
        mask_s = tf.concat((tmp_u0, tmp_u0), -2)
        mask_s = tf.concat((mask_s, tmp_e0), -2)
        mask_s = tf.concat((mask_s, tmp_s1), -2)
        Mask.append(tf.tile(tf.expand_dims(mask_s, 0),[batch_size, 1, 1]))


        if self.consecutive_links:
            shifted = tf.manip.roll(eye, shift=1, axis=0)
            A_.append(shifted)

        if self.backward_links:
            for i in range(len(A)):
                A_.append(K.permute_dimensions(A[i], [0, 2, 1]))

        if self.self_links:
            A_.append(eye)
        A_sum = K.stack(A_, axis=1)
        A_sum = K.sum(A_sum, axis=1)
        #A_u = A_sum
        AHWs = list()
        for i in range(self.node_types):
            if self.edge_weighting:
                features *= self.W_edges[i]
            type_A = Mask[i] * A_sum 
            HW = K.dot(features, self.W[i]) # Shape: (None, num_nodes, output_dim)

            AHW = K.batch_dot(type_A, HW) + self.B[i] # Shape: (None, num_nodes, num_features)
            AHWs.append(AHW)
        AHWs_stacked = K.stack(AHWs, axis=1) # Shape: (None, num_supports, num_nodes, num_features)
        output = K.sum(AHWs_stacked, axis=1) # Shape: (None, num_nodes, output_dim)
        #code.interact(local=locals())
#        if self.bias:
#            output += self.b
        return self.activation(output)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'num_bases': self.num_bases,
                  'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

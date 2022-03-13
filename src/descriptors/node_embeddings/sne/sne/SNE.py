from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import logging
import pickle
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from itertools import tee
from six.moves import xrange
import numpy as np
import tensorflow.compat.v1 as tf
# import tensorflow as tf

flags = tf.app.flags
# # These line can't be set when sne_embedding module is working:
# flags.DEFINE_string("save_path", './', "Directory to write the model and training sammaries.")
# flags.DEFINE_string("train_data", 'data/wiki_edit.txt', "Training text file.")
# flags.DEFINE_string("label_data", 'data/wiki_usr_labels.txt', "Nodes labels text file.")
# flags.DEFINE_string("walks_data", 'data/wiki_edit_num_40.walk', "Random walks on data")
# flags.DEFINE_integer("embedding_size", 100, "The embedding dimension size.")
# flags.DEFINE_integer("samples_to_train", 25, "Number of samples to train(*Million).")
# flags.DEFINE_float("learning_rate", 0.025, "Initial learning rate.")
# flags.DEFINE_integer("num_sampled", 512, "The number of classes to randomly sample per batch.")
# flags.DEFINE_integer("context_size", 3, "The number of context nodes .")
# flags.DEFINE_integer("batch_size", 50, "Number of training examples processed per step.")
# flags.DEFINE_boolean("is_train", True, "Train or restore")
# # TODO: Let it be uncommented when sne_embedding module is working.
FLAGS = flags.FLAGS


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

class Options(object):
    '''Options used by LINE model.'''

    def __init__(self):
        # Model options.

        # Embedding dimension.
        self.emb_dim = FLAGS.embedding_size

        # training text file.
        self.train_data = FLAGS.train_data
        self.label_data = FLAGS.label_data
        self.walks_data = FLAGS.walks_data

        # The initial learning rate.
        self.learning_rate = FLAGS.learning_rate

        # Number of samples to train.
        self.samples_to_train = FLAGS.samples_to_train * 1000000

        self.num_sampled = FLAGS.num_sampled
        self.context_size = FLAGS.context_size

        # Number of examples for one training step.
        self.batch_size = FLAGS.batch_size

        # Where to write out embeddings.
        self.save_path = FLAGS.save_path
        self.is_train = FLAGS.is_train


class SNE(object):

    def __init__(self, options, session):
        self._options = options
        self._session = session
        self._vertex2id = {}
        self._id2vertex = []
        self._vid2degree = []
        self._edge_source_id = []
        self._edge_target_id = []
        self._edge_weight = []
        self._edge_sign = {}
        self._alias = []
        self._prob = []
        self._edge_count = 0
        self._read_data(self._options.train_data, self._options.label_data)
        # self.InitAliasTable()
        self.build_graph()

    def _read_data(self, filename, label_path):
        '''Read data from training file.'''
        weight = 1.
        for line in open(filename):
            chunkes = line.strip().split()
            assert len(chunkes) == 3
            [name_v1, name_v2, sign] = chunkes
            '''Add current vertex to dict if it is not included yet.'''
            if name_v1 not in self._vertex2id:
                self._vertex2id[name_v1] = len(self._vertex2id)
                self._id2vertex.append(name_v1)
                self._vid2degree.append(0)

            if name_v2 not in self._vertex2id:
                self._vertex2id[name_v2] = len(self._vertex2id)
                self._id2vertex.append(name_v2)
                self._vid2degree.append(0)

            vid1 = self._vertex2id[name_v1]
            self._vid2degree[vid1] += weight
            self._edge_source_id.append(vid1)

            vid2 = self._vertex2id[name_v2]
            self._vid2degree[vid2] += weight
            self._edge_target_id.append(vid2)

            self._edge_sign[(vid1, vid2)] = (int(sign)+1)/2
            self._edge_weight.append(weight)

        self._edge_num = len(self._edge_weight)
        self._options.vertex_size = len(self._id2vertex)
        s_idx, t_idx, signs = [], [], []
        for (s, t), y in self._edge_sign.items():
            s_idx.append(s)
            t_idx.append(t)
            signs.append(y)
        self._y = np.asarray(signs)
        self._s_idx = np.asarray(s_idx)
        self._t_idx = np.asarray(t_idx)
        logging.info("Edge number : %d" % (self._edge_num))
        logging.info("Vertex number : %d" % (self._options.vertex_size))

        print("!!!!!")
        print(label_path)
        df = pd.read_csv(label_path, sep='\t', header=None)
        df = df.applymap(str)
        self._node_labels = df.set_index(0)[1].to_dict()
        print(self._node_labels)
        print(len(self._node_labels))
        print(self._id2vertex)
        print(len(self._id2vertex))
        assert len(self._node_labels) == len(self._id2vertex)


    def make_instances(self, f_walks,):
        with open(f_walks, 'r') as f:
            corpus = f.readlines()
        data = []
        labels = []
        signs = []
        for sentence in corpus:
            id_sentence = [self._vertex2id[str(node)] for node in sentence.split()]
            for instance in zip(*(id_sentence[i:] for i in xrange(self._options.context_size + 1))):
                sign = []
                for u, v in pairwise(instance):
                    if (u, v) in self._edge_sign:
                        sign.append(self._edge_sign[(u, v)])
                    elif (v, u) in self._edge_sign:
                        sign.append(self._edge_sign[(v, u)])
                    else:
                        raise ValueError((u,v))
                assert len(sign) == len(instance)-1
                data.append(instance[:-1])
                labels.append(instance[-1])
                signs.append(sign)
        logging.info("number of data %d" %len(data))
        return data, labels, signs


    def forward(self, sources, targets, signs):
        '''Build the graph for the forward pass.'''
        opts = self._options

        targets = tf.reshape(targets, [-1, 1])

        # embedding:[vertex_size,emb_dim]
        init_width = 0.5

        emb_vertex = tf.Variable(tf.random_uniform([opts.vertex_size, opts.emb_dim], -init_width, init_width),
                                 name='emb_vertex')
        w = tf.Variable(tf.random_uniform([opts.vertex_size, opts.emb_dim], dtype=tf.float32), name='proj_w')
        b = tf.Variable(tf.zeros([opts.vertex_size], dtype=tf.float32), name="proj_b")

        sign_weights = tf.Variable(tf.random_uniform([2, opts.emb_dim], -init_width, init_width), name='weights')

        self._emb_vertex = emb_vertex
        self._sign_w = sign_weights
        self._emb_context = w
        self._proj_b = b

        self.global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int32)

        example_emb = tf.nn.embedding_lookup(emb_vertex, sources)
        weight = tf.nn.embedding_lookup(sign_weights, signs)

        # vector bilinear
        bilinear = tf.reduce_sum(tf.multiply(example_emb, weight), 1)

        loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=w, biases=b, inputs=bilinear, labels=targets,
                                                         num_sampled=opts.num_sampled, num_classes=opts.vertex_size))
        return loss

    def optimize(self, loss):
        '''build the graph to optimize the loss function.'''
        optimizer = tf.train.AdagradOptimizer(1.0)
        train = optimizer.minimize(loss,
                                   global_step=self.global_step,
                                   gate_gradients=optimizer.GATE_NONE
                                   )
        self._train = train

    def build_graph(self):
        '''build the graph for the full model.'''
        opts = self._options
        sources = tf.placeholder(tf.int32, shape=[opts.batch_size, opts.context_size])
        self._sources = sources
        targets = tf.placeholder(tf.int64, shape=[opts.batch_size])
        self._targets = targets
        signs = tf.placeholder(tf.int32, shape=[opts.batch_size, opts.context_size])
        self._signs = signs

        loss = self.forward(sources, targets, signs)
        self._loss = loss
        self.optimize(loss)

        # create a saver.
        self.saver = tf.train.Saver()

        # Initialize all variables.
        tf.global_variables_initializer().run()


    def train(self):
        '''trian the model.'''
        opts = self._options

        # print training info.
        loss_list = []
        last_count, last_time = self._edge_count, time.time()
        self._edge_count = 0
        walks, target, signs = self.make_instances(f_walks=opts.walks_data)
        n_train_batches = np.round(int(len(walks)/opts.batch_size))
        print("Total batch number %d" %n_train_batches)
        for minibatch_index in xrange(n_train_batches):

            _sources, _targets, _signs = walks[minibatch_index*opts.batch_size:(minibatch_index+1)*opts.batch_size], \
                                         target[minibatch_index*opts.batch_size:(minibatch_index+1)*opts.batch_size],\
                                         signs[minibatch_index*opts.batch_size:(minibatch_index+1)*opts.batch_size]
            feed_dict = {self._sources: _sources, self._targets: _targets, self._signs: _signs}
            (loss, _) = self._session.run([self._loss, self._train], feed_dict=feed_dict)
            loss_list.append(loss)
            self._edge_count += opts.batch_size
            if self._edge_count % 2000 == 0:
                now = time.time()
                rate = (self._edge_count - last_count) / (now - last_time)
                progress = 100 * (self._edge_count) / float(opts.samples_to_train)
                last_time = now
                last_count = self._edge_count
                average_loss = np.mean(np.array(loss_list))
                print("loss:%6.2f average loss: %f edges/sec:%8.0f%%\r" % (
                    loss, average_loss, rate
                ), end="")
                sys.stdout.flush()
                loss_list = []
            if self._edge_count >= opts.samples_to_train:
                break
        print("")

    def get_sne(self):
        """
        Get the SNE which was trained in order to create the prediction model

        :return: The embedding
        :rtype: list
        """
        return self._emb_vertex.eval()

    def get_options(self):
        """
        Get the options which are used to train SNE

        :return: The options
        :rtype: Options
        """
        return self._options

    def save_model(self,):
        with open(os.path.join(self._options.save_path, "lbl_wiki_edit_emb.pkl"), 'wb') as f:
            pickle.dump(self._emb_vertex.eval(), f)
            pickle.dump(self._sign_w.eval(), f)
            pickle.dump(self._emb_context.eval(), f)
            pickle.dump(self._id2vertex, f)
            pickle.dump(self._vertex2id, f)
            pickle.dump(self._edge_source_id, f)
            pickle.dump(self._edge_target_id, f)
            pickle.dump(self._edge_sign, f)
            pickle.dump(self._node_labels, f)


def main(_):
    logging.basicConfig(level=logging.INFO)
    if not FLAGS.train_data:
        logging.error('no train file.')
        sys.exit(1)
    opts = Options()
    with tf.Graph().as_default(), tf.Session() as session:
        model = SNE(opts, session)
        if opts.is_train:
            model.train()
            model.save_model()


if __name__ == '__main__':
    '''This is for test.'''
    tf.app.run()
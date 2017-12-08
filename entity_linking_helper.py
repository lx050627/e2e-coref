#!/usr/bin/env python

import os
import re
import sys
sys.path.append(os.getcwd())
import time
import random

import numpy as np
import tensorflow as tf
import coref_model as cm
import util

def neural_network(config, is_training=True):

    dense_units = config["el_dense_units"]
    dropout_rate = config["el_dropout_rate"]
    n_classes = config["el_n_classes"]

    # Placeholders
    x_mention = tf.placeholder(tf.float32, shape=[None, 1320])
    x_cluster_m = tf.placeholder(tf.float32, shape=[None, 2, 1320, 1])
    x_cluster_p = tf.placeholder(tf.float32, shape=[None, 2, 1320, 1])
    y_labels = tf.placeholder(tf.float32, shape=[None, n_classes])

    input_tensors = [x_mention, x_cluster_m, x_cluster_p]
    output_tensor = y_labels

    return input_tensors, output_tensor, tf.layers.dense(x_mention, n_classes)

    # Network Structure
    conv_m = tf.layers.conv2d(x_cluster_m, 1, 2, activation=tf.tanh)
    conv_p = tf.layers.conv2d(x_cluster_p, 1, 2, activation=tf.tanh)

    flat_cluster_m = tf.contrib.layers.flatten(conv_m)
    flat_cluster_p = tf.contrib.layers.flatten(conv_p)
    concat = tf.concat([x_mention, flat_cluster_m, flat_cluster_p], 1)

    fc1 = tf.layers.dropout(concat, rate=dropout_rate, training=is_training) 
    fc1 = tf.layers.dense(fc1, dense_units)
    fc1 = tf.layers.batch_normalization(fc1)
    fc1 = tf.nn.relu(fc1)

    fc2 = tf.layers.dropout(fc1, rate=dropout_rate, training=is_training) 
    fc2 = tf.layers.dense(fc2, n_classes)
    fc2 = tf.layers.batch_normalization(fc2)
    output_layer = tf.nn.softmax(fc2)

    return input_tensors, output_tensor, output_layer

def reduce_labels(data):
    data  = [data['arr_{}'.format(i)] for i in range(4)]

    roles = ['Ross', 'Joey', 'Chandler', 'Monica', 'Phoebe', 'Rachel']
    ids = [335, 183, 59, 248, 292, 306]

    # data = list(map(lambda key: test_data[key], sorted([key for key in test_data])))
    labels = np.argmax(data[3], axis=1)
    labels = np.array(map(lambda id: ids.index(id) if id in ids else 6, labels))
    one_hot = np.zeros((len(labels), 7))
    one_hot[np.arange(len(labels)), labels] = 1
    data[3] = one_hot

    return data
def get_entity_linking_data(config, data, model, session): 
    n_classes = config["el_n_classes"]

    mention_embs = []
    cluster_embs = []
    mention_pair_embs = []
    entity_ids = []

    for example_num, (tensorized_example, example) in enumerate(data):
        feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
        mention_tensors = [model.candidate_starts, model.candidate_ends, model.candidate_mention_emb] 
        mention_starts, mention_ends, mention_emb = session.run(mention_tensors, feed_dict=feed_dict)

        mention_idx = zip(mention_starts, mention_ends)
        clusters = example["clusters"]
        cluster_ids = example["cluster_ids"]

        for cluster_id, cluster in zip(cluster_ids, clusters):
            ### CLUSTER_EMB
            cluster_m_emb = []
            for start, end in cluster:
                idx = mention_idx.index((start, end))
                cluster_m_emb.append(mention_emb[idx])
            cluster_m_emb = np.array(cluster_m_emb)

            avg_pool = np.average(cluster_m_emb, axis=0)
            max_pool = np.amax(cluster_m_emb, axis=0)
            cluster_m_pool = np.stack((avg_pool, max_pool))

            ### MENTION_PAIR_EMB
            for i, mention_emb_i in enumerate(cluster_m_emb):
                cluster_p_emb = []
                for j, mention_emb_j in enumerate(cluster_m_emb):
                    if i == j: continue
                    # controversial here, paper didn't mention how to combine mention pair features
                    mention_emb_pair = (mention_emb_i + mention_emb_j) / 2
                    cluster_p_emb.append(mention_emb_pair)

                # only one mention in the cluster
                if len(cluster_p_emb) == 0: 
                    cluster_p_emb = np.array([mention_emb_i])
                else:
                    cluster_p_emb = np.array(cluster_p_emb)

                # mention pair cluster pooling
                avg_pool = np.average(cluster_p_emb, axis=0)
                max_pool = np.amax(cluster_p_emb, axis=0)
                cluster_p_pool = np.stack((avg_pool, max_pool))

                # entity_id
                entity_id = np.zeros(401)
                entity_id[cluster_id[0]] = 1

                # concat features
                mention_embs.append(mention_emb_i)
                cluster_embs.append(cluster_m_pool)
                mention_pair_embs.append(cluster_p_pool)
                entity_ids.append(entity_id)
    
    # reduce labels to 6 roles and Unknown
    if n_classes == 7:
        roles = ['Ross', 'Joey', 'Chandler', 'Monica', 'Phoebe', 'Rachel']
        ids = [335, 183, 59, 248, 292, 306]
        entity_ids = np.argmax(entity_ids, axis=1)
        entity_ids = list(map(lambda id: ids.index(id) if id in ids else 6, entity_ids))

    dataset = [mention_embs, cluster_embs, mention_pair_embs, entity_ids]
    dataset = map(np.array, dataset)

    # set CNN channel to 1
    dataset[1:3] = list(map(lambda d: d.reshape(d.shape + (1, )), dataset[1:3]))

    return dataset

if __name__ == "__main__": 
    if "GPU" in os.environ:
        util.set_gpus(int(os.environ["GPU"]))
    else:
        util.set_gpus()
  
    if len(sys.argv) > 1:
        name = sys.argv[1]
        print "Running experiment: {} (from command-line argument).".format(name)
    else:
        name = os.environ["EXP"]
        print "Running experiment: {} (from environment variable).".format(name)
  
    # CONFIGURATION
    config = util.get_config("experiments.conf")[name]
    config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], name))
  
    util.print_config(config)
    model = cm.CorefModel(config)
  
    saver = tf.train.Saver()

    log_dir = config["log_dir"]
    train_fp = os.path.join(log_dir, config["el_train_path"])
    test_fp = os.path.join(log_dir, config["el_test_path"])
  
    with tf.Session() as session:
        checkpoint_path = os.path.join(log_dir, "model.max.ckpt")

        print "Evaluating {}".format(checkpoint_path)
        saver.restore(session, checkpoint_path)
        
        model.load_train_data()
        model.load_eval_data()

        train_data = get_entity_linking_data(config, model.train_data, model, session)
        test_data = get_entity_linking_data(config, model.eval_data, model, session)

        np.savez(train_fp, *train_data)
        np.savez(test_fp, *test_data)

        # for d in test_data:
            # print(d.shape)

        # mention_embs, cluster_embs, mention_pair_embs, entity_ids = test_data
        # print(mention_embs.shape)
        # print(cluster_embs.shape)
        # print(mention_pair_embs.shape)
        # print(entity_ids.shape)

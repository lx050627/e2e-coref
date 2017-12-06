#!/usr/bin/env python
import random
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

import util
import sys

def neural_network(config, tensors, is_training=True):
    dense_units = config["el_dense_units"]
    dropout_rate = config["el_dropout_rate"]
    n_classes = config["el_n_classes"]

    x_mention, x_cluster_m, x_cluster_p, y_labels = tensors

    conv_m = tf.layers.conv2d(x_cluster_m, 1, 2, activation=tf.tanh)
    conv_p = tf.layers.conv2d(x_cluster_p, 1, 2, activation=tf.tanh)

    flat_cluster_m = tf.contrib.layers.flatten(conv_m)
    flat_cluster_p = tf.contrib.layers.flatten(conv_p)
    concat = tf.concat([x_mention, flat_cluster_m, flat_cluster_p], 1)

    fc1 = tf.layers.dropout(concat, rate=dropout_rate, training=is_training) 
    fc1 = tf.layers.dense(fc1, dense_units)
    fc1 = tf.nn.relu(fc1)

    fc2 = tf.layers.dropout(fc1, rate=dropout_rate, training=is_training) 
    fc2 = tf.layers.dense(fc2, n_classes)
    output = tf.nn.softmax(fc2)

    return output

if __name__ == "__main__":

    ### CONFIGURATION
    if len(sys.argv) > 1:
        name = sys.argv[1]
        print "Running experiment: {} (from command-line argument).".format(name)
    else:
        name = os.environ["EXP"]
        print "Running experiment: {} (from environment variable).".format(name)

    config = util.get_config("experiments.conf")[name]
    config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], name))

    log_dir = config["log_dir"]
    train_fp = os.path.join(log_dir, config["el_train_path"])
    test_fp = os.path.join(log_dir, config["el_test_path"])
    n_classes = config["el_n_classes"]
    learning_rate = config["el_learning_rate"]
    epoches = config["el_epoches"]
    batch_size = config["el_batch_size"]

    ### DATA
    train_data = np.load(train_fp)
    train_data  = [train_data['arr_{}'.format(i)] for i in range(4)]
    # train_data = list(map(lambda key: test_data[key], sorted([key for key in test_data])))

    test_data = np.load(test_fp)
    test_data = [test_data['arr_{}'.format(i)] for i in range(4)]

    ### NETWORK GRAPH
    x_mention = tf.placeholder(tf.float32, shape=[None, 1320])
    x_cluster_m = tf.placeholder(tf.float32, shape=[None, 2, 1320, 1])
    x_cluster_p = tf.placeholder(tf.float32, shape=[None, 2, 1320, 1])
    y_labels = tf.placeholder(tf.float32, shape=[None, n_classes])

    tensors = [x_mention, x_cluster_m, x_cluster_p, y_labels]
    
    y_train = neural_network(config, tensors, is_training=True)
    y_test = neural_network(config, tensors, is_training=False)

    ### LOSS
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(y_labels, y_train))
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

    ### TRAINING
    print("TRAINING: ")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epoches):
            epoch_loss = 0
            for i in range(0, train_data[0].shape[0], batch_size):
                feed_dict = dict(zip(tensors, [d[i:i+batch_size] for d in train_data]))
                c, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
                epoch_loss += c
            print('epoch', epoch, 'completed out of',epoches,'loss:',epoch_loss)
        
    ### TESTING
    print("\nTESTING: ")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        feed_dict = dict(zip(tensors, test_data))
        prediction = sess.run(y_test, feed_dict=feed_dict)

        y_pred = np.argmax(prediction, axis=1)
        y_true = np.argmax(test_data[-1], axis=1)
        print(y_pred)
        print(y_true)
        print("Accuracy: ", accuracy_score(y_true, y_pred))

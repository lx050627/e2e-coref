#!/usr/bin/env python
import os
import re
import sys
sys.path.append(os.getcwd())
import time
import random

import numpy as np
import tensorflow as tf

from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    data_size = 50
    dense_units = 200
    n_classes = 7

    learning_rate = 0.1

    mention = np.random.rand(data_size, 1320)
    cluster_m = np.random.rand(data_size, 2, 1320, 1)
    cluster_p = np.random.rand(data_size, 2, 1320, 1)

    target = np.random.randint(0, n_classes, data_size)
    labels = np.zeros((data_size, n_classes))
    labels[np.arange(data_size), target] = 1

    x_mention = tf.placeholder(tf.float32, shape=[None, 1320])
    x_cluster_m = tf.placeholder(tf.float32, shape=[None, 2, 1320, 1])
    x_cluster_p = tf.placeholder(tf.float32, shape=[None, 2, 1320, 1])
    y_labels = tf.placeholder(tf.int32, shape=[None, 7])

    # NETWORK GRAPH
    # Convolution Layer with 32 filters and a kernel size of 5
    conv_m = tf.layers.conv2d(x_cluster_m, 1, 2, activation=tf.tanh)
    conv_p = tf.layers.conv2d(x_cluster_p, 1, 2, activation=tf.tanh)

    flat_cluster_m = tf.contrib.layers.flatten(conv_m)
    flat_cluster_p = tf.contrib.layers.flatten(conv_p)
    concat = tf.concat([x_mention, flat_cluster_m, flat_cluster_p], 1)

    fc1 = tf.layers.dense(concat, dense_units)
    fc1 = tf.nn.relu(fc1)
    output = tf.layers.dense(fc1, n_classes)

    pred_probas = tf.nn.softmax(output)
    cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(y_labels, pred_probas))
    
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)


    # TRAINING
    epoches = 10
    batch_size = 10
    with tf.Session() as session:
        session.run(tf.initialize_all_variables())

        for epoch in range(epoches):
            epoch_loss = 0
            for _ in range(0, data_size, batch_size):
                _, c, prediction = session.run([optimizer, cost, pred_probas], \
                        feed_dict={
                            x_mention: mention,
                            x_cluster_m: cluster_m,
                            x_cluster_p: cluster_p,
                            y_labels: labels
                        })
                epoch_loss += c

            print('epoch', epoch, 'completed out of',epoches,'loss:',epoch_loss)

        
        print("Accuracy:")
        print(accuracy_score(np.argmax(labels, axis=1), np.argmax(prediction, axis=1)))
        # correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        # print('accuracy:',accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


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

def neural_network(input_tensors, is_training=True):
    x_mention, x_cluster_m, x_cluster_p = input_tensors

    conv_m = tf.layers.conv2d(x_cluster_m, 1, 2, activation=tf.tanh)
    conv_p = tf.layers.conv2d(x_cluster_p, 1, 2, activation=tf.tanh)

    flat_cluster_m = tf.contrib.layers.flatten(conv_m)
    flat_cluster_p = tf.contrib.layers.flatten(conv_p)
    concat = tf.concat([x_mention, flat_cluster_m, flat_cluster_p], 1)

    fc1 = tf.layers.dropout(concat, rate=dropout_rate, training=True) 
    fc1 = tf.layers.dense(fc1, dense_units)
    fc1 = tf.nn.relu(fc1)

    fc2 = tf.layers.dropout(fc1, rate=dropout_rate, training=True) 
    fc2 = tf.layers.dense(fc2, n_classes)
    output = tf.nn.softmax(fc2)

    return output

if __name__ == "__main__":
    ### CONFIGURATION
    data_size = 50
    dense_units = 100
    n_classes = 7
    learning_rate = 0.05
    dropout_rate = 0.8

    epoches = 20
    batch_size = 10

    ### DATA
    mention = np.random.rand(data_size, 1320)
    cluster_m = np.random.rand(data_size, 2, 1320, 1)
    cluster_p = np.random.rand(data_size, 2, 1320, 1)

    y_true = np.random.randint(0, n_classes, data_size)
    labels = np.zeros((data_size, n_classes))
    labels[np.arange(data_size), y_true] = 1

    ### NETWORK GRAPH
    # input_tensors.append(tf.placeholder(tf.float32, shape=[None, 1320])) #x_mention
    # input_tensors.append(tf.placeholder(tf.float32, shape=[None, 2, 1320, 1])) #x_cluster_m
    # input_tensors.append(tf.placeholder(tf.float32, shape=[None, 2, 1320, 1])) #x_cluster_p
    x_mention = tf.placeholder(tf.float32, shape=[None, 1320])
    x_cluster_m = tf.placeholder(tf.float32, shape=[None, 2, 1320, 1])
    x_cluster_p = tf.placeholder(tf.float32, shape=[None, 2, 1320, 1])
    input_tensors = [x_mention, x_cluster_m, x_cluster_p]
    # output_tensor = tf.placeholder(tf.float32, shape=[None, n_classes])
    y_labels = tf.placeholder(tf.float32, shape=[None, n_classes])

    y_train = neural_network(input_tensors, is_training=True)

    ### LOSS
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(y_labels, y_train))
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

    ### TRAINING
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epoches):
            epoch_loss = 0
            for i in range(0, data_size, batch_size):
                feed_dict = {
                        x_mention: mention[i:i+batch_size],
                        x_cluster_m: cluster_m[i:i+batch_size],
                        x_cluster_p: cluster_p[i:i+batch_size],
                        y_labels: labels[i:i+batch_size]
                }
                c, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
                epoch_loss += c

            print('epoch', epoch, 'completed out of',epoches,'loss:',epoch_loss)
        
    ### TESTING
    y_test = neural_network(input_tensors, is_training=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        prediction = sess.run(y_test, feed_dict= {
            x_mention: mention,
            x_cluster_m: cluster_m,
            x_cluster_p: cluster_p,
            y_labels: labels
        })

        y_pred = np.argmax(prediction, axis=1)
        print(y_pred)
        print(y_true)
        print("Accuracy: ", accuracy_score(y_true, y_pred))

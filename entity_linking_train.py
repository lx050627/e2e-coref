#!/usr/bin/env python
import random
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score

import util
import sys
import entity_linking_helper as helper

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
    model_fp = os.path.join(log_dir, "model.entity_linking.ckpt")

    ### DATA
    train_data = helper.reduce_labels(np.load(train_fp))

    ### NETWORK GRAPH
    input_tensors, y_labels, y_train = helper.neural_network(config, is_training=True)
    tensors = input_tensors + [y_labels]

    ### LOSS
    cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(y_labels, y_train))
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cross_entropy)
    # cross_entropy = tf.reduce_mean(
        # tf.nn.softmax_cross_entropy_with_logits(labels=y_labels, logits=y_train))
    # optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_labels, 1), tf.argmax(y_train, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    with tf.Session() as sess:
        ### training
        print("TRAINING: ")
        sess.run(tf.global_variables_initializer())
        epoches = 10
        for epoch in range(epoches):
            epoch_loss = 0
            precision = 0 
            data_size = train_data[0].shape[0]
            total_batch = int(data_size/batch_size)
            for i in range(0, data_size, batch_size):
                feed_dict = dict(zip(tensors, [d[i:i+batch_size] for d in train_data]))
                acc, c, _ = sess.run([accuracy, cross_entropy, optimizer], feed_dict=feed_dict)

                epoch_loss += c / total_batch
                precision += acc / total_batch
            print('epoch', epoch, 'completed out of',epoches, 'precision: ', precision, 'loss:',epoch_loss)

        save_path = saver.save(sess, model_fp)
        print("model saved in file: %s" % save_path)

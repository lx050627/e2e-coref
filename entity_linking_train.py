#!/usr/bin/env python
import random
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score

import util
import sys
from entity_linking_helper import neural_network, reduce_labels

if __name__ == "__main__":

    ### CONFIGURATION
    if len(sys.argv) > 1:
        name = sys.argv[1]
        print "Running experiment: {} (from command-line argument).".format(name)
    else:
        name = 'best'

    config = util.get_config("experiments.conf")[name]
    config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], name))

    log_dir = config["log_dir"]
    train_fp = os.path.join(log_dir, config["el_train_path"])
    n_classes = config["el_n_classes"]
    learning_rate = config["el_learning_rate"]
    epoches = config["el_epoches"]
    batch_size = config["el_batch_size"]
    model_fp = os.path.join(log_dir, "model.entity_linking.ckpt")

    ### DATA
    train_data = reduce_labels(np.load(train_fp))
    # train_data = np.load(train_fp)

    ### NETWORK GRAPH
    input_tensors, y_labels, y_train = neural_network(config, is_training=True)
    tensors = input_tensors + [y_labels]

    ### LOSS
    cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(y_labels, y_train))
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_labels, 1), tf.argmax(y_train, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    with tf.Session() as sess:
        print("TRAINING: ")
        sess.run(tf.global_variables_initializer())
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

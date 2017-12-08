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
    test_data = helper.reduce_labels(np.load(test_fp))

    ### NETWORK GRAPH

    # If there is a built graph
    # tf.reset_default_graph()
    input_tensors, y_labels, y_test = helper.neural_network(config, is_training=False)
    tensors = input_tensors + [y_labels]

    # call after graph construction
    saver = tf.train.Saver()
    with tf.Session() as sess:
        ### TESTING
        print("\nTESTING: ")

        saver.restore(sess, model_fp)
        sess.run(tf.global_variables_initializer())
        print("Model restored.")

        feed_dict = dict(zip(tensors, test_data))
        prediction = sess.run(y_test, feed_dict=feed_dict)

        y_pred = np.argmax(prediction, axis=1)
        y_true = np.argmax(test_data[-1], axis=1)
        print(confusion_matrix(y_true, y_pred))
        print("Accuracy: ", accuracy_score(y_true, y_pred))

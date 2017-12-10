#!/usr/bin/env python
import random
import sys
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score

import util
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
    test_fp = os.path.join(log_dir, config["el_test_path"])
    model_fp = os.path.join(log_dir, "model.entity_linking.ckpt")

    ### DATA
    test_data = reduce_labels(np.load(test_fp))

    ### NETWORK GRAPH

    # If there is a built graph
    # tf.reset_default_graph()
    input_tensors, y_labels, y_test = neural_network(config, is_training=False)
    tensors = input_tensors + [y_labels]

    # call after graph construction
    saver = tf.train.Saver()
    with tf.Session() as sess:
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

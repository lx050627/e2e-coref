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

def generate_mention_emb(model, session): 
  # model.load_eval_data()
  # for example_num, (tensorized_example, example) in enumerate(model.eval_data):
  model.load_train_data()
  for example_num, (tensorized_example, example) in enumerate(model.train_data):
    _, _, _, _, _, _, gold_starts, gold_ends, _ = tensorized_example
    feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}

  mention_starts, mention_ends, mention_emb = \
          session.run([model.candidate_starts, model.candidate_ends, model.candidate_mention_emb], feed_dict=feed_dict)
  return mention_starts, mention_ends, mention_emb

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

  config = util.get_config("experiments.conf")[name]
  config["log_dir"] = util.mkdirs(os.path.join(config["log_root"], name))

  util.print_config(config)
  model = cm.CorefModel(config)

  saver = tf.train.Saver()
  log_dir = config["log_dir"]

  with tf.Session() as session:
    checkpoint_path = os.path.join(log_dir, "model.max.ckpt")
    print "Evaluating {}".format(checkpoint_path)
    saver.restore(session, checkpoint_path)

    mention_starts, mention_ends, mention_emb = generate_mention_emb(model, session)
    print(mention_starts, mention_ends, mention_emb)
    print(mention_emb.shape)

    # dataset_dic = {}
    # for n in range(0,len(mention_starts)):
      # dataset_dic[(mention_starts[n], mention_ends[n])] = mention_emb[n]

    # clusters_emb = []
    # for cluster in clusters:
      # cluster_emb = [dataset_dic[mention] for mention in cluster]
      # clusters_emb.append(cluster_emb)
    # clusters_emb = np.array(clusters_emb)

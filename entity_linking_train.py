#!/usr/bin/env python

import os
import re
import sys
sys.path.append(os.getcwd())
import time
import random

import numpy as np
import tensorflow as tf

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

    mention_starts, mention_ends, mention_emb, clusters = generate_mention_emb(model, session)
    print(mention_starts, mention_ends, mention_emb, clusters)

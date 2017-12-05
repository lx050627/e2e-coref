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

def get_entity_linking_data(model, session): 
    # model.load_eval_data()
    # for example_num, (tensorized_example, example) in enumerate(model.eval_data):
    model.load_train_data()

    mention_embs = []
    cluster_embs = []
    mention_pair_embs = []
    for example_num, (tensorized_example, example) in enumerate(model.train_data):
        feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
        mention_tensors = [model.candidate_starts, model.candidate_ends, model.candidate_mention_emb] 
        mention_starts, mention_ends, mention_emb = session.run(mention_tensors, feed_dict=feed_dict)

        mention_idx = zip(mention_starts, mention_ends)
        clusters = example["clusters"]

        for cluster in clusters:
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

                # concat features
                mention_embs.append(mention_emb_i)
                cluster_embs.append(cluster_m_pool)
                mention_pair_embs.append(cluster_p_pool)
    
    return np.array(mention_embs), np.array(cluster_embs), np.array(mention_pair_embs)

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
  
        mention_embs, cluster_embs, mention_pair_embs = get_entity_linking_data(model, session)
        print(mention_embs.shape)
        print(cluster_embs.shape)
        print(mention_pair_embs.shape)

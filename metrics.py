import numpy as np
from collections import Counter
from sklearn.utils.linear_assignment_ import linear_assignment

"""
Mostly borrowed from https://github.com/clarkkev/deep-coref/blob/master/evaluation.py
"""

def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)

class CorefEvaluator(object):
    def __init__(self):
        self.evaluators = [Evaluator(m) for m in (muc, b_cubed, ceafe)]

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        for e in self.evaluators:
            e.update(predicted, gold, mention_to_predicted, mention_to_gold)

    def get_f1(self):
        return sum(e.get_f1() for e in self.evaluators) / len(self.evaluators)

    def get_recall(self):
        return sum(e.get_recall() for e in self.evaluators) / len(self.evaluators)

    def get_precision(self):
        return sum(e.get_precision() for e in self.evaluators) / len(self.evaluators)

    def get_prf(self):
        for e in self.evaluators:
            if e.metric == ceafe:
                print "**************************  Ceafe **************************"
                print("F1:{:.2f}%".format(e.get_ceafe_f1()))
                print "Precision:{:.2f}%".format(e.get_ceafe_precision())
                print("Recall:{:.2f}%".format(e.get_ceafe_recall()))

            elif e.metric == muc:
                print "**************************  MUC **************************"
                print("F1:{:.2f}%".format(e.get_muc_f1()))
                print "Precision:{:.2f}%".format(e.get_muc_precision())
                print("Recall:{:.2f}%".format(e.get_muc_recall()))
            else:
                print "**************************  B_Cubed **************************"
                print("F1:{:.2f}%".format(e.get_b_f1()))
                print "Precision:{:.2f}%".format(e.get_b_precision())
                print("Recall:{:.2f}%".format(e.get_b_recall()))
        return self.get_precision(), self.get_recall(), self.get_f1()

class Evaluator(object):
    def __init__(self, metric, beta=1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0

        self.ceafe_pn =0
        self.ceafe_pd = 0
        self.ceafe_rn = 0
        self.ceafe_rd = 0

        self.muc_pn=0
        self.muc_pd=0
        self.muc_rn=0
        self.muc_rd=0

        self.b_pn = 0
        self.b_pd = 0
        self.b_rn = 0
        self.b_rd = 0

        self.metric = metric
        self.beta = beta

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        if self.metric == ceafe:
            pn, pd, rn, rd = self.metric(predicted, gold)
            self.ceafe_pn+=pn
            self.ceafe_pd+=pd
            self.ceafe_rn+=rn
            self.ceafe_rd+=rd
        elif self.metric == muc:
            pn, pd = self.metric(predicted, mention_to_gold)
            rn, rd = self.metric(gold, mention_to_predicted)
            self.muc_pn += pn
            self.muc_pd += pd
            self.muc_rn += rn
            self.muc_rd += rd
        else:
            pn, pd = self.metric(predicted, mention_to_gold)
            rn, rd = self.metric(gold, mention_to_predicted)
            self.b_pn += pn
            self.b_pd += pd
            self.b_rn += rn
            self.b_rd += rd

        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

    def get_muc_f1(self):
        return f1(self.muc_pn, self.muc_pd, self.muc_rn, self.muc_rd, beta=self.beta)

    def get_muc_recall(self):
        return 0 if self.muc_rn == 0 else self.muc_rn / float(self.muc_rd)

    def get_muc_precision(self):
        return 0 if self.muc_pn == 0 else self.muc_pn / float(self.muc_pd)


    def get_b_f1(self):
        return f1(self.b_pn, self.b_pd, self.b_rn, self.b_rd, beta=self.beta)

    def get_b_recall(self):
        return 0 if self.b_rn == 0 else self.b_rn / float(self.b_rd)

    def get_b_precision(self):
        return 0 if self.b_pn == 0 else self.b_pn / float(self.b_pd)


    def get_ceafe_f1(self):
        return f1(self.ceafe_pn, self.ceafe_pd, self.ceafe_rn, self.ceafe_rd, beta=self.beta)

    def get_ceafe_recall(self):
        return 0 if self.ceafe_rn == 0 else self.ceafe_rn / float(self.ceafe_rd)

    def get_ceafe_precision(self):
        return 0 if self.ceafe_pn == 0 else self.ceafe_pn / float(self.ceafe_pd)


    def get_f1(self):
        return f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def get_counts(self):
        return self.p_num, self.p_den, self.r_num, self.r_den


def evaluate_documents(documents, metric, beta=1):
    evaluator = Evaluator(metric, beta=beta)
    for document in documents:
        evaluator.update(document)
    return evaluator.get_precision(), evaluator.get_recall(), evaluator.get_f1()


def b_cubed(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        gold_counts = Counter()
        correct = 0
        for m in c:
            if m in mention_to_gold:
                gold_counts[tuple(mention_to_gold[m])] += 1
        for c2, count in gold_counts.iteritems():
            if len(c2) != 1:
                correct += count * count

        num += correct / float(len(c))
        dem += len(c)

    return num, dem


def muc(clusters, mention_to_gold):
    tp, p = 0, 0
    for c in clusters:
        p += len(c) - 1
        tp += len(c)
        linked = set()
        for m in c:
            if m in mention_to_gold:
                linked.add(mention_to_gold[m])
            else:
                tp -= 1
        tp -= len(linked)
    return tp, p


def phi4(c1, c2):
    return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))


def ceafe(clusters, gold_clusters):
    clusters = [c for c in clusters if len(c) != 1]
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi4(gold_clusters[i], clusters[j])
    matching = linear_assignment(-scores)
    similarity = sum(scores[matching[:, 0], matching[:, 1]])
    return similarity, len(clusters), similarity, len(gold_clusters)


def lea(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        common_links = 0
        all_links = len(c) * (len(c) - 1) / 2.0
        for i, m in enumerate(c):
            if m in mention_to_gold:
                for m2 in c[i + 1:]:
                    if m2 in mention_to_gold and mention_to_gold[m] == mention_to_gold[m2]:
                        common_links += 1

        num += len(c) * common_links / float(all_links)
        dem += len(c)

    return num, dem

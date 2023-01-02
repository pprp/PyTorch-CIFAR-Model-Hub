import copy
import math

import numpy as np
import scipy.stats
from scipy.stats import stats


def concordant_pair_ratio(list1, list2):
    """Proposed `cpr`"""
    assert len(list1) == len(list2)
    total_number = len(list1)
    num_concordant = 0
    for i in range(len(list1)):
        if list1[i] * list2[i] > 0:
            num_concordant += 1
    res = num_concordant / (total_number + 1e-9)
    return res


def pearson(true_vector, pred_vector):
    n = len(true_vector)
    # simple sums
    sum1 = sum(float(true_vector[i]) for i in range(n))
    sum2 = sum(float(pred_vector[i]) for i in range(n))
    # sum up the squares
    sum1_pow = sum([pow(v, 2.0) for v in true_vector])
    sum2_pow = sum([pow(v, 2.0) for v in pred_vector])
    # sum up the products
    p_sum = sum([true_vector[i] * pred_vector[i] for i in range(n)])
    # 分子num，分母den
    num = p_sum - (sum1 * sum2 / n)
    try:
        den = math.sqrt((sum1_pow - pow(sum1, 2) / n) *
                        (sum2_pow - pow(sum2, 2) / n) + 1e-8)
    except ValueError:
        return 0

    if den == 0:
        return 0.0
    return num / den


def kendalltau(true_vector, pred_vector):
    tau, p_value = scipy.stats.kendalltau(true_vector, pred_vector)
    return tau


def spearman(true_vector, pred_vector):
    coef, p_value = scipy.stats.spearmanr(true_vector, pred_vector)
    return coef


def rank_difference(true_vector, pred_vector):
    """Compute the underestimate degrade.
    RD = ri - ni
    ri: true rank
    ni: estimated rank
    ranging from 0 to 1:
        RD > 0: supernet underestimate the performance.
        RD < 0: supernet overestimate the performance.
    """

    def get_rank(vector):
        v = np.array(vector)
        v_ = copy.deepcopy(v)
        v_.sort()
        rank = []

        for i in v:
            rank.append(list(v_).index(i))
        return rank

    rank1 = get_rank(true_vector)
    rank2 = get_rank(pred_vector)

    length = len(true_vector)

    sum_rd = 0.
    for i in range(length):
        sum_rd += rank1[i] - rank2[i] if rank1[i] > rank2[i] else 0

    return sum_rd / length


# Calculate the BR@K, WR@K [0-1]
def minmax_n_at_k(true_scores, predict_scores, ks=[0.01, 0.05, 0.10, 0.50]):
    true_scores = np.array(true_scores)
    predict_scores = np.array(predict_scores)
    num_archs = len(true_scores)
    true_ranks = np.zeros(num_archs)
    true_ranks[np.argsort(true_scores)] = np.arange(num_archs)[::-1]
    predict_best_inds = np.argsort(predict_scores)[::-1]
    minn_at_ks = []
    for k in ks:
        ranks = true_ranks[predict_best_inds[:int(k * len(true_scores))]]
        if len(ranks) < 1:
            continue
        minn = int(np.min(ranks)) + 1
        maxn = int(np.max(ranks)) + 1
        minn_at_ks.append(
            (k, minn, float(minn) / num_archs, maxn, float(maxn) / num_archs))
    return minn_at_ks


# Calculate the P@topK, P@bottomK, and Kendall-Tau in predicted topK/bottomK
# [0-1]
def p_at_tb_k(true_scores, predict_scores, ratios=[0.01, 0.05, 0.1, 0.5]):
    predict_scores = np.array(predict_scores)
    true_scores = np.array(true_scores)
    predict_inds = np.argsort(predict_scores)[::-1]
    num_archs = len(predict_scores)
    true_ranks = np.zeros(num_archs)
    true_ranks[np.argsort(true_scores)] = np.arange(num_archs)[::-1]
    patks = []
    for ratio in ratios:
        k = int(num_archs * ratio)
        if k < 1:
            continue
        top_inds = predict_inds[:k]
        bottom_inds = predict_inds[num_archs - k:]
        p_at_topk = len(np.where(true_ranks[top_inds] < k)[0]) / float(k)
        p_at_bottomk = len(
            np.where(true_ranks[bottom_inds] >= num_archs - k)[0]) / float(k)
        kd_at_topk = stats.kendalltau(predict_scores[top_inds],
                                      true_scores[top_inds]).correlation
        kd_at_bottomk = stats.kendalltau(predict_scores[bottom_inds],
                                         true_scores[bottom_inds]).correlation
        # [ratio, k, P@topK, P@bottomK, KT in predicted topK, KT in predicted bottomK]
        patks.append(
            (ratio, k, p_at_topk, p_at_bottomk, kd_at_topk, kd_at_bottomk))
    return patks


if __name__ == '__main__':
    a = np.array([np.random.randn() for _ in range(100)])
    b = np.array([np.random.randn() for _ in range(100)])

    print(rank_difference(a[:20], b[:20]), rank_difference(a[20:40], b[20:40]))
    print(p_at_tb_k(a, b))
    print(minmax_n_at_k(a, b))
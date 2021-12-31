import numpy as np
import random


def Dominates(x, y):
    """Check if x dominates y.

    :param x: a sample
    :type x: array
    :param y: a sample
    :type y: array

    the smaller value, the better performance
    """
    return np.all(x <= y) & np.any(x < y)

def NonDominatedSorting(pop):
    """Perform non-dominated sorting.

    :param pop: the current population
    :type pop: array
    """
    _, npop = pop.shape
    rank = np.zeros(npop)
    dominatedCount = np.zeros(npop) # 记录每个个体其被支配的数量
    # 记录每个个个体被哪些个体支配了
    # 例如 [[1,2],[3],[0]]表示 第一个个体被 [1,2] 个体支配
    dominatedSet = [[] for i in range(npop)]
    front = [[]]
    # 更新 dominatedCount 和 dominatedSet
    for i in range(npop):
        for j in range(i + 1, npop):
            crt_objs = pop[:, i]
            next_objs = pop[:, j]
            if Dominates(crt_objs, next_objs):
                dominatedSet[i].append(j)
                dominatedCount[j] += 1
            if Dominates(next_objs, crt_objs):
                dominatedSet[j].append(i)
                dominatedCount[i] += 1
        if dominatedCount[i] == 0:
            rank[i] = 1
            front[0].append(i)
    # 求解出不同批次的支配解 
    # 比如 [ [1,3], [0], [2] ] 表示 [1,3] 不被任何个体支配，0 只被一个个体支配， 2被两个个体支配
    k = 0
    while (True):
        Q = []
        for i in front[k]:
            crt_objs = pop[:, i]
            for j in dominatedSet[i]:
                dominatedCount[j] -= 1
                if dominatedCount[j] == 0:
                    Q.append(j)
                    rank[j] = k + 1
        if len(Q) == 0:
            break
        front.append(Q)
        k += 1
    return front

def CARS_NSGA(targets, objs, N):
    """pNSGA-III (CARS-NSGA).

        :param targets: the first objective, e.g. accuracy
        :type targets: array
        :param objs: the other objective, e.g. FLOPs, number of parameteres
        :type objs: array
        :param N: number of population
        :type N: int
        :return: The selected samples
        :rtype: array
    
        Example:
        targets = np.random.rand(1,10)  # accuracy
        objs = np.random.rand(2,10)     # model size, FLOPs
        pareto_front = CARS_NSGA(targets, objs, 5) # get top-5
    """
    if len(targets.shape) == 1:
        length = targets.shape[0]
    else:
        length = targets.shape[1]
    selected = np.zeros(length)
    fronts = []
    for target in targets:
        for obj in objs:
            fronts.append(NonDominatedSorting(np.vstack((1 / (target + 1e-10), obj))))
            # fronts.append(NonDominatedSorting(np.vstack((1 / (target + 1e-10), 1 / (obj + 1e-10)))))
    stage = 0
    while (np.sum(selected) < N):
        current_front = []
        for i in range(len(fronts)):
            if stage < len(fronts[i]):
                current_front.append(fronts[i][stage])
        current_front = [np.array(c) for c in current_front]
        current_front = np.hstack(current_front)
        current_front = list(set(current_front))
        if np.sum(selected) + len(current_front) <= N:
            for i in current_front:
                selected[i] = 1
        else:
            not_selected_indices = np.arange(len(selected))[selected==0]
            crt_front = [index for index in current_front if index in not_selected_indices]
            num_to_select = N - np.sum(selected).astype(np.int32)
            current_front = crt_front if len(crt_front) <= num_to_select else random.sample(crt_front, num_to_select)
            for i in current_front:
                selected[i] = 1
        stage += 1
    return np.where(selected == 1)[0]

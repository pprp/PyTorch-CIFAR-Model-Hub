import csv
import glob
import itertools
import random

import matplotlib.pyplot as plt
"""
author:pprp
description: 可视化
"""

path = 'exps'
csv_files = glob.glob('./exps/*/log.csv')

plt.title('val acc-epoch')
plt.xlabel('epoch')
plt.ylabel('val acc')

marker = itertools.cycle(
    ('+', '<', 'd', 'h', 'H', '1', '.', '2', 'D', 'o', '*', 'v', '>'))

legends = []

for csv_file in csv_files:
    name = csv_file.split('e-')[1].split('_m')[0]
    if name.startswith('1') or name.startswith('3') or name.startswith(
            '6') or name.startswith('8'):
        continue
    if name.startswith('cbam'):
        continue

    # print(name)
    tmp_idx = []
    tmp_tacc = []  # train acc
    tmp_vacc = []  # valid acc
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if i == 0:
                continue
            epoch, lr, loss, acc, val_loss, val_acc = line
            tmp_idx.append(int(float(epoch)))
            tmp_tacc.append(float(acc))
            tmp_vacc.append(float(val_acc))
        current_mark = next(marker)
        current_color = (random.random(), random.random(), random.random())
        plt.plot(tmp_idx,
                 tmp_tacc,
                 marker=current_mark,
                 markersize=1,
                 color=current_color)
        plt.plot(tmp_idx,
                 tmp_vacc,
                 marker=current_mark,
                 markersize=1,
                 color=current_color)

    legends.append(name + '_train')
    legends.append(name + '_val')

plt.legend(legends, fontsize=6)

plt.savefig('acc_epoch_analysis.png')

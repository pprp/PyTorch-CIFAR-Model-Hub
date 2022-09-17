import glob
import os
import shutil
"""
author:pprp
description： 用于删除空文件夹
"""

path = 'exps'

for d in os.listdir(path):
    exp_path = os.path.join(path, d)
    results = glob.glob(os.path.join(exp_path, '*.pth'))
    csv_result = glob.glob(os.path.join(exp_path, '*.csv'))

    if len(results) <= 5 or len(csv_result) == 0:
        print('remove %s' % exp_path)
        shutil.rmtree(exp_path)

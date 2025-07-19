# -*- coding: utf-8 -*-
# @Time    : 2024/3/30
# @Author  : Siyang Li
# @File    : statstest.py
# produce statistical testing results
import numpy as np
from scipy import stats
from scipy.stats import wilcoxon
from scipy.stats import ttest_rel

names = ['MOSI', 'TUSAS', 'TweetEval', 'WOS']

for data_name in names:
    print('#' * 30)

    data1 = np.loadtxt('./results/' + data_name + '/smlovr.csv')
    data1 += 1
    paths = ['./results/' + data_name + '/laa.csv']

    for path in paths:
        data2 = np.loadtxt(path)
        data2 += 1
        print(path)

        stat, p = wilcoxon(data1, data2)  # Wilcoxon
        #stat, p = ttest_rel(data1, data2)  # t-test
        stats.false_discovery_control(p)

        print('stat={:f}, p={:.128f}'.format(stat, p))
        if p > 0.05:
            print('!!! Probably the same distribution')
        else:
            print('Probably different distributions')

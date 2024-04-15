# Example of the Wilcoxon Signed-Rank Test
import numpy as np
from scipy import stats
from scipy.stats import wilcoxon
from scipy.stats import ttest_rel

names = ['MOSI', 'TUSAS', 'TweetEval', 'WOS']

for data_name in names:
    print('#' * 30)

    data1 = np.loadtxt('./results/' + data_name + '/smlmulti-hard.csv')
    data1 += 1
    paths = ['./results/' + data_name + '/voting.csv']

    for path in paths:
        data2 = np.loadtxt(path)
        data2 += 1
        print(path)

        stat, p = wilcoxon(data1, data2)
        #stat, p = ttest_rel(data1, data2)  # t-test
        stats.false_discovery_control(p)

        print('stat={:f}, p={:.128f}'.format(stat, p))
        if p > 0.05:
            print('!!! Probably the same distribution')
        else:
            print('Probably different distributions')

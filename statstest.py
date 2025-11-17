"""
We quantify the difference between two models’ class predictions.
Note that this purpose is different from classic statistical tests for machine learning algorithms, which usually operate on evaluation metrics such as accuracy scores.
We use the disagreement rate—the proportion of test samples on which their predicted labels differ.
We then conduct a one-sided proportion test comparing the observed disagreement against a predefined practical threshold δ (the minimum disagreement considered meaningfully large), and reject the null hypothesis at significance level α.
A significant result indicates that the two models’ predictions are sufficiently different.
This is different from CIM paper’s usage of statistical tests in Section IV.E (Statistical Tests), and we believe the current method is more appropriate for our purpose.
"""

import numpy as np

from scipy.stats import norm

def prediction_difference_test(pred1, pred2, delta=0.05, alpha=0.05):
    pred1 = np.asarray(pred1).astype(int)
    pred2 = np.asarray(pred2).astype(int)

    n = len(pred1)
    disagree = np.sum(pred1 != pred2)
    p_hat = disagree / n

    se = np.sqrt(delta * (1 - delta) / n)
    z = (p_hat - delta) / se
    p_value = 1 - norm.cdf(z)

    significant = (p_hat > delta) and (p_value < alpha)

    return {
        "n": n,
        "disagree_rate": p_hat,
        "z": z,
        "p_value": p_value,
        "significant": significant,
    }


names = ['MOSI', 'TUSAS', 'TweetEval', 'WOS']

delta = 0.05  # threshold on the disagreement rate; try values like 0.1 to see how much disagreement you require to call the models "different"
alpha = 0.05  # statistical significance level

for data_name in names:
    print("\n" + "#" * 50)
    print("Dataset:", data_name)

    predA = np.loadtxt(f'./results/{data_name}/smlovr.csv')

    compare_paths = [
        f'./results/{data_name}/dawidskene.csv',
        f'./results/{data_name}/ebcc.csv',
        f'./results/{data_name}/glad.csv',
        f'./results/{data_name}/la.csv',
        f'./results/{data_name}/laa.csv',
        f'./results/{data_name}/m-msr.csv',
        f'./results/{data_name}/mace.csv',
        f'./results/{data_name}/pm.csv',
        f'./results/{data_name}/voting.csv',
        f'./results/{data_name}/wawa.csv',
        f'./results/{data_name}/zencrowd.csv',
    ]
    for path in compare_paths:
        print(path)
        predB = np.loadtxt(path)

        res = prediction_difference_test(predA, predB, delta=delta, alpha=alpha)

        print(f"Sample num = {res['n']}")
        print(f"Disagreement rate = {res['disagree_rate']:.4f}")
        print(f"Z-value = {res['z']:.4f}")
        print(f"p-value  = {res['p_value']:.6f}")

        if res["significant"]:
            print("YES. Their predictions are sufficiently different on this test set.")
        else:
            print("NO. Their predictions cannot be regarded as sufficiently different on this test set.")


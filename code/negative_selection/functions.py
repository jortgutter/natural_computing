from sklearn.metrics import roc_auc_score, roc_curve
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import subprocess


def calc_score(train_file: str, test_file: str, n: int, r: int) -> List[float]:
    command = f"java -jar negsel2.jar -self {train_file} -n {n} -r {r} -c -l < {test_file}"
    result = subprocess.getoutput(command)
    outputs = [float(i) for i in result.split("\n")]

    return outputs


def roc_auc(train_file: str, base_test_file: str, comp_test_file: str, n: int, r: int, plot_roc: bool = False) -> float:
    getscore = lambda x: calc_score(train_file, x, n, r)
    base_score = getscore(base_test_file)
    comp_score = getscore(comp_test_file)

    # Create 'dataset' of base and comparison data. The comparison data labels are set to 1 (anomaly)
    appended = np.append(base_score, comp_score)
    labels = np.zeros(appended.shape[0])
    labels[len(base_score):] = 1

    if plot_roc:
        fpr, tpr, thresholds = roc_curve
        plt.plot(fpr, tpr, 'crimson')
        plt.title(f"ROC-Curve of {base_test_file} versus {comp_test_file}")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.show()

    auc = roc_auc_score(labels, appended)

    return auc


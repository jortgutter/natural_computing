from sklearn.metrics import roc_auc_score, roc_curve
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import os.path
import sys


def sys_roc_auc(train: str, test: str, labels: np.ndarray, r: int, jar_path: str, alphabet: str, path: str = "/.", plot_roc: bool = False) -> float:
    """
    This function calculates the area under the ROC Curve for a test file set (negative and positive samples) after
    training on the train file (negative samples only).

    :param train: The pre-processed training data filename (negative samples)
    :param test: The pre-processed test data filename containing negative and positive samples
    :param labels: The labels corresponding to the test data
    :param r: Minimum match sequence length
    :param jar_path: Path to negsel2.jar file
    :param alphabet: File containing all the characters from the used alphabet
    :param path: Directory in which the train and test files are stored
    :param plot_roc: Default: False. If True, plot the ROC Curve
    :return: Area under the ROC Curve
    """
    scores = sys_scores(train, test, r=r, jar_path=jar_path, alphabet=alphabet, path=path)

    if plot_roc:
        with open(os.path.join(path, test[0]), "r") as file:
            n = len(file.readline().strip())
        fpr, tpr, thresholds = roc_curve(labels, scores)
        plt.plot(fpr, tpr, 'crimson')
        plt.title(f"ROC-Curve of {test[0]} versus {test[1]}\nn = {n}, r = {r}")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.show()

    auc = roc_auc_score(labels, scores)

    return auc


def sys_scores(train: str, test: str, r: int, jar_path: str, alphabet: str, path: str = "./") -> np.ndarray:
    """
    This function calculates the anomaly scores per line for a given test file (negative and positive samples)
    after training on the train file (negative samples only).

    :param train: The pre-processed training data filename (negative samples)
    :param test: The pre-processed test data filename containing negative and positive samples
    :param r: Minimum match sequence length
    :param jar_path: Path to negsel2.jar file
    :param alphabet: File containing all the characters from the used alphabet
    :param path: Directory in which the train and test files are stored
    :return: Tuple of anomaly scores and the class labels
    """
    out = calc_score(train, test, r=r, jar_path=jar_path, alphabet=alphabet, path=path)

    nans = np.where(np.isnan(out))[0]
    scores = np.zeros(len(nans))

    lower = -1
    for i, upper in enumerate(nans):
        scores[i] = np.mean(out[lower + 1:upper])
        lower = upper

    return scores


def calc_score(train_file: str, test_file: str, r: int, jar_path:str, alphabet: str = None, path: str = "./") -> List[float]:
    """
    This function calculates the anomaly scores per line for a given test file after training on the train file.

    :param train_file: The pre-processed training data filename
    :param test_file: The pre-processed test data filename
    :param r: Minimum match sequence length
    :param jar_path: Path to negsel2.jar file
    :param alphabet: File containing all the characters from the used alphabet
    :param path: Directory in which the train and test file are stored
    :return: Anomaly scores per line
    """
    with open(os.path.join(path, train_file), "r") as file:
        n1 = len(file.readline().strip())
    with open(os.path.join(path, test_file), "r") as file:
        n2 = len(file.readline().strip())

    if n1 != n1 or (n1 < r or n2 < r):
        print("Line lengths do not match or are < r!", file=sys.stderr)
        exit(1)

    train_path = os.path.join(path, train_file)
    test_path = os.path.join(path, test_file)

    alphabet_string = f"-alphabet file://{os.path.join(path, alphabet)}" if alphabet else ""

    command = f"java -jar {jar_path} -self {train_path} -n {n1} -r {r} -c -l {alphabet_string} < {test_path}"
    result = subprocess.getoutput(command)
    outputs = [float(i) for i in result.split("\n")]

    return outputs


def roc_auc(train_file: str, base_test_file: str, comp_test_file: str, r: int, path: str = "/.", plot_roc: bool = False) -> float:
    """
    This function calculates the area under the ROC Curve for a true class (base file) and false class (comp file) after
    training on the train file.

    :param train_file: The pre-processed training data filename
    :param base_test_file: The pre-processed test data filename of the true class
    :param comp_test_file: The pre-processed test data filename of the false class
    :param r: Minimum match sequence length
    :param path: Directory in which the train and test files are stored
    :param plot_roc: Default: False. If True, plot the ROC Curve
    :return: Area under the ROC Curve
    """
    getscore = lambda x: calc_score(train_file, x, r, path=path)
    base_score = getscore(base_test_file)
    comp_score = getscore(comp_test_file)

    # Create 'dataset' of base and comparison data. The comparison data labels are set to 1 (anomaly)
    appended = np.append(base_score, comp_score)
    labels = np.zeros(appended.shape[0])
    labels[len(base_score):] = 1

    if plot_roc:
        with open(os.path.join(path, base_test_file), "r") as file:
            n = len(file.readline().strip())
        fpr, tpr, thresholds = roc_curve(labels, appended)
        plt.plot(fpr, tpr, 'crimson')
        plt.title(f"ROC-Curve of {base_test_file} versus {comp_test_file}\nn = {n}, r = {r}")
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.show()

    auc = roc_auc_score(labels, appended)

    return auc

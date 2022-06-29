import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import h5py


filename = "../out/ensemble_6ep_5conv_4chan_20nets_preds.h5"


def prediction_heatmap(predictions, y, cls, show_class=True):
    class_pred = predictions[:, y == cls, :].mean(axis=1)
    if not show_class:
        class_pred[:, cls] = 0
    sns.heatmap(class_pred.T)
    plt.xlabel(f'Network')
    plt.ylabel(f'Prediction')
    plt.title(f'Prediction distribution over all ensemble members for class {cls}')
    plt.show()

cls=2
arr = np.r_[0:cls, cls+1:10]
print(arr)

with h5py.File(filename, "r") as f:

    pred = f['p'][()]
    y = f['y'][()]

for cls in range(10):
    prediction_heatmap(predictions=pred, y=y, cls=cls, show_class=True)

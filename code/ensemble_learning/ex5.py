from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np




iris_data = load_digits()

X = iris_data['data']
y = iris_data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

N_RUNS = 10
max_trees = 30
depth_steps = 5
f_scores = np.zeros((depth_steps, N_RUNS, max_trees))


for depth in range(5):
    f_scores_depth = []
    for i in tqdm(range(N_RUNS)):
        f_scores_run = []
        for n_trees in range(max_trees):
            forest = RandomForestClassifier(n_estimators=n_trees+1, max_depth = depth+1, random_state=42)
            forest.fit(X_train, y_train)
            y_hat_forest = forest.predict(X_test)
            C_forest = confusion_matrix(y_test, y_hat_forest)
            f_score_forest = f1_score(y_test, y_hat_forest, average='weighted')
            f_scores[depth,i,n_trees] = f_score_forest

# Decision Tree baseline
f_scores_baselines = np.zeros((5, 10))
for i in range(2):
    for run in range(10):
        tree = DecisionTreeClassifier(max_depth=(i+1)*5, random_state=42)
        tree.fit(X_train, y_train)
        y_hat_tree = tree.predict(X_test)
        C_tree = confusion_matrix(y_test, y_hat_tree)
        f_score_tree = f1_score(y_test, y_hat_tree, average='weighted')
        f_scores_baselines[i, run] = f_score_tree


# plotting
colors = ['red', 'orange', 'yellow', 'green', 'blue']
styles = ['-', '--']
for i in range(5):
    #plt.plot(f_scores[i].T, alpha=0.2, color=colors[i])
    plt.plot(np.mean(f_scores[i], axis=0), color=colors[i], linewidth=2, label=f'Forest, depth={i+1}')
for j in range(2):
    plt.hlines(y=np.mean(f_scores_baselines[j], axis=0), xmin=0, xmax=max_trees, color='black', linestyles=styles[j], label=f'Single tree, depth={(j+1)*5}')
plt.legend()
plt.xlabel('Number of trees in forest')
plt.ylabel('F1_score')
plt.title('F1_scores of several decision trees and random forests')
plt.show()




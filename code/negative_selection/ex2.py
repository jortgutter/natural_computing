from functions import *

FILE_NR = "3"
JAR = "./negsel2.jar"

# The target dir specifies the main folder in which the files are located. All following files are relative to this path
TARGET_DIR = "./syscalls/snd-cert/n12"
ALPHABET = "../snd-cert.alpha"
TRAIN = "train"
TEST = FILE_NR
LABEL = f"../snd-cert.{FILE_NR}.labels"

with open(os.path.join(TARGET_DIR, LABEL), "r") as file:
    labels = np.array([int(line) for line in file.readlines()])

aucs = []
for r in [3]:#range(1, 12):
    auc = sys_roc_auc(
        train=TRAIN,
        test=TEST,
        labels=labels,
        r=r,
        jar_path=JAR,
        alphabet=ALPHABET,
        path=TARGET_DIR,
        plot_roc=False
    )
    # aucs.append(auc)
    print(f"AUC r={r}: {auc}")

# plt.plot(range(1, len(aucs)+1), aucs, 'k')
# plt.title("AUC for different $r$-values")
# plt.xlabel("$r$")
# plt.ylabel("AUC")
# plt.show()

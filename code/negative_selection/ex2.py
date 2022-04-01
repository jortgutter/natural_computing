import matplotlib.pyplot as plt
from functions import *

# TODO: Use separate path for negsel2.jar to avoid annoying filenames for ALPHABET, TRAIN, and TEST
# TODO: Something cool with wrappers?

TARGET_DIR = "./"
ALPHABET = "syscalls/snd-cert/snd-cert.alpha"
TRAIN = "syscalls/snd-cert/snd-cert_n12_false.train"
TEST = (
    "syscalls/snd-cert/snd-cert_1_n12_false.test",
    "syscalls/snd-cert/snd-cert_1_n12_true.test"
)

aucs = []
for r in range(1, 12):
    auc = sys_roc_auc(
        train_false=TRAIN,
        test_files=TEST,
        r=r,
        alphabet=ALPHABET,
        path=TARGET_DIR,
        plot_roc=False
    )
    aucs.append(auc)
    print(f"AUC r={r}: {auc}")

plt.plot(range(1, len(aucs)+1), aucs, 'k')
plt.title("AUC for different $r$-values")
plt.xlabel("$r$")
plt.ylabel("AUC")
plt.show()

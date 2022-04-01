from functions import *

TARGET_DIR = "./"
ALPHABET = "syscalls/snd-cert/snd-cert.alpha"
TRAIN = "syscalls/snd-cert/snd-cert_n12_false.train"
TEST = ["syscalls/snd-cert/snd-cert_1_n12_false.test",
        "syscalls/snd-cert/snd-cert_1_n12_true.test"]

scores, labels = sys_scores(TRAIN, TEST, r=5, alphabet=ALPHABET, path=TARGET_DIR)


print("Bye")

from functions import *
import os.path

TARGET_DIR = "../../../assignment4/negative-selection"

ENG_TRAIN = "english.train"
ENG_TEST = "english.test"
TAG_TEST = "tagalog.test"

auc_r1 = roc_auc(ENG_TRAIN, ENG_TEST, TAG_TEST, 10, 1, path=TARGET_DIR, plot_roc=True)
auc_r4 = roc_auc(ENG_TRAIN, ENG_TEST, TAG_TEST, 10, 4, path=TARGET_DIR, plot_roc=True)
auc_r9 = roc_auc(ENG_TRAIN, ENG_TEST, TAG_TEST, 10, 9, path=TARGET_DIR, plot_roc=True)

print(f"auc r = 1: {auc_r1}"
      f"auc r = 4: {auc_r4}"
      f"auc r = 9: {auc_r9}")

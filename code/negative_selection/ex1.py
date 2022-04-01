from functions import *
import os

TARGET_DIR = "../../../assignment4/negative-selection"

ENG_TRAIN = "english.train"
ENG_TEST = "english.test"
TAG_TEST = "tagalog.test"

for r in (1, 4, 9):
    auc = roc_auc(ENG_TRAIN, ENG_TEST, TAG_TEST, r, path=TARGET_DIR, plot_roc=True)
    print(f"AUC {TAG_TEST[:-5]}, r={r}: {auc}", end="\n")
print()

for language in os.listdir(os.path.join(TARGET_DIR, "lang")):
    for r in (1, 4, 9):
        comp_file = os.path.join("lang", language)
        auc = roc_auc(ENG_TRAIN, ENG_TEST, comp_file, r, path=TARGET_DIR)
        print(f"AUC {language[:-4]}, r={r}: {auc}")

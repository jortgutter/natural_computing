import json

with open("ensemble_10ep_5conv_5chan.json", 'r') as file:
    res = json.load(file)

vals = []
for k, vs in res.items():
    vals.append(
        (k, vs['n_params'], vs['final_accuracy_pmaj'], vs['training_time'], vs['n_nets'])
    )

vals.sort(key=lambda x: x[-1], reverse=True)

for val in vals:
    print(val[:-1])
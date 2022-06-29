import argparse
import json
import sys
import os
import re


def main(args):
    output_dict = {}

    input_regex = f"{args.classifier}_" \
                  f"{args.epochs}ep_" \
                  f"{args.conv_blocks if args.conv_blocks > 0 else '.*'}conv_" \
                  f"{args.start_channels if args.start_channels > 0 else '.*'}chan" \
                  f"{'.*' if not args.add_regex else args.add_regex}.txt"

    for filename in [f for f in os.listdir("../../out/") if re.match(input_regex, f)]:
        with open(os.path.join("../../out", filename), 'r') as file:
            content = "".join(file.readlines())
            output_dict[filename] = base_parser(content) if args.classifier == 'base' else ensemble_parser(content)

    json.dump(output_dict, open(args.output_filename, 'w'))


def base_parser(content: str):
    out = dict()

    out["n_chan"] = int(re.findall(r"\(None, 32, 32, (\d*)\)", content)[0])
    out["n_conv"] = len(re.findall(r"\n conv2d.*(Conv2D)", content))
    out["n_params"] = int(re.sub(",", "", re.findall(r"Trainable params: ([\d,]*)", content)[0]))
    for measure in ["loss", "accuracy", "val_loss", "val_accuracy"]:
        out[measure] = [float(val) for val in re.findall(fr"{measure}: \[([\d., ]*)]", content)[0].split(", ")]
    out["training_time"] = float(re.findall(r"Training time: (\d*\.\d*) seconds", content)[0])
    out["n_epochs"] = len(out["loss"])
    out["final_accuracy"] = float(re.findall(r"accuracy: ([\d.]*)$", content)[0])

    return out


def ensemble_parser(content: str):
    out = dict()

    out["n_nets"] = int(re.findall(r"===== Network 1/(\d*) =====", content)[0])
    out["n_chan"] = int(re.findall(r"\(None, 32, 32, (\d*)\)", content)[0])
    out["n_conv"] = len(re.findall(r"\n conv2d.*(Conv2D)", content))
    out["training_time"] = float(re.findall(r"Training time: (\d*\.\d*) seconds", content)[0])

    out["n_params_per_net"] = int(re.sub(",", "", re.findall(r"Trainable params: ([\d,]*)", content)[0]))
    out["n_params"] = out["n_nets"] * out["n_params_per_net"]

    out["loss"] = []
    out["accuracy"] = []
    out["val_loss"] = []
    out["val_accuracy"] = []

    for net in range(1, out["n_nets"] + 1):
        results = re.findall(rf"===== Network {net}/{out['n_nets']} =====\n"
                             rf"loss: \[([\d., ]*)]\n"
                             rf"accuracy: \[([\d., ]*)]\n"
                             rf"val_loss: \[([\d., ]*)]\n"
                             rf"val_accuracy: \[([\d., ]*)]", content)

        for i, measure in enumerate(["loss", "accuracy", "val_loss", "val_accuracy"]):
            out[measure].append([float(val) for val in results[0][i].split(", ")])

    out["final_accuracy_pmed"] = float(re.findall(r"Accuracy prob. median voting:\W([\d.]*)", content)[0])
    out["final_accuracy_pmaj"] = float(re.findall(r"Accuracy prob. majority voting:\W([\d.]*)", content)[0])
    out["final_accuracy_cmaj"] = float(re.findall(r"Accuracy class majority voting:\W([\d.]*)", content)[0])

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("classifier", type=str)
    parser.add_argument("epochs", type=int)
    parser.add_argument("conv_blocks", type=int)
    parser.add_argument("start_channels", type=int)
    parser.add_argument("output_filename", type=str)
    parser.add_argument("--add_regex", type=str, help="Regex to match after start channels in filename (exclude .txt)")

    args = parser.parse_args()

    main(args)

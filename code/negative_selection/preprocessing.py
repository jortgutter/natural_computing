# main.py
import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('path', help='path to folder containing the input files')
    parser.add_argument('data_file', type=str, help='datafile to be parsed')
    parser.add_argument('label_file', type=str, help='corresponding label file')
    parser.add_argument('string_length', metavar='n', type=int, help='target word length')
    parser.add_argument('-dest', dest='destination_path', default=False,
                        help='optional path of the output files. If not set, output files will be saved in the input folder')
    args = parser.parse_args()

    # open inout files
    data = [line.strip() for line in open(os.path.join(args.path, args.data_file)).readlines()]
    labels = [int(line.strip()) for line in open(os.path.join(args.path, args.label_file)).readlines()]

    n = args.string_length

    # create output file paths:
    parsed_data_file = args.data_file.split('.')
    parsed_data_file = '_'.join(parsed_data_file[:-1]) + '_parsed_' + str(n) + '.' + parsed_data_file[-1]
    parsed_data_path = os.path.join(args.destination_path if args.destination_path else args.path, parsed_data_file)

    parsed_label_file = args.label_file.split('.')
    parsed_label_file = '_'.join(parsed_label_file[:-1]) + '_parsed_' + str(n) + '.' + parsed_label_file[-1]
    parsed_label_path = os.path.join(args.destination_path if args.destination_path else args.path, parsed_label_file)

    print(parsed_data_path)
    print(parsed_label_path)





if __name__ == "__main__":
    main()


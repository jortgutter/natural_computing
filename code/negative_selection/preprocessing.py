import os
import math
import argparse

PADDING = 'a'


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Preprocess system call files to a fixed word length per line')

    parser.add_argument('path', help='Path to folder containing the input files')
    parser.add_argument('data_file', type=str, help='Datafile to be parsed')
    parser.add_argument('string_length', metavar='n', type=int, help='Target word length')
    parser.add_argument('-dest', dest='destination_path', default=False,
                        help='Optional path of the output files. If not set, output files will be saved in the input '
                             'folder')
    parser.add_argument('--no-overlap', dest='no_overlap', action='store_const', const=True, default=False,
                        help="Parser will use sliding windows unless this flag is set")
    args = parser.parse_args()

    n = args.string_length
    data_filename_decomposed = args.data_file.split('.')
    is_test_file = data_filename_decomposed[-1] == 'test'
    no_overlap = args.no_overlap

    # open input files
    data = [line.strip() for line in open(os.path.join(args.path, args.data_file)).readlines()]
    if is_test_file:
        labels = [int(line.strip()) for line in
                  open(os.path.join(args.path, '.'.join(data_filename_decomposed[:-1]) + '.labels')).readlines()]
    else:
        labels = [0 for i in range(len(data))]

    # create output file paths
    parsed_true_file = '_'.join(data_filename_decomposed[:-1]) + '_n' + str(n) + '_true.' + data_filename_decomposed[-1]
    parsed_true_path = os.path.join(args.destination_path if args.destination_path else args.path, parsed_true_file)

    parsed_false_file = '_'.join(data_filename_decomposed[:-1]) + '_n' + str(n) + '_false.' + data_filename_decomposed[
        -1]
    parsed_false_path = os.path.join(args.destination_path if args.destination_path else args.path, parsed_false_file)

    # open output files
    files = [open(parsed_false_path, 'w'), open(parsed_true_path, 'w')]

    # cut
    for i, (line, label) in enumerate(zip(data, labels)):
        if no_overlap:
            for j in range(math.ceil(len(line) / n)):
                chunk = line[j * n:j * n + n]
                if len(chunk) < n:
                    chunk += PADDING * (n - len(chunk))
                files[label].write(chunk + '\n')

        else:
            line += PADDING * (max(0, n - len(line)))
            for j in range(len(line) - (n - 1)):
                files[label].write(line[j:j + n] + '\n')

        files[label].write('\n')

    for file in files:
        file.close()


if __name__ == "__main__":
    main()

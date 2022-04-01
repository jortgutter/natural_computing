import os
import math
import argparse


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Preprocesses system call files to a fixed word length per line')

    parser.add_argument('file', type=str, help='Datafile to be parsed')
    parser.add_argument('string_length', metavar='n', type=int, help='Target word length')
    parser.add_argument('-padding', dest='padding', default='a', help="Symbol used for padding. Default: 'a'")
    parser.add_argument('-dest', dest='destination_path', default=False,
                        help='Optional path of the output file. If not set, output files will be saved in the input '
                             'folder')
    parser.add_argument('--no-overlap', dest='no_overlap', action='store_const', const=True, default=False,
                        help="Parser will use sliding windows unless this flag is set")
    args = parser.parse_args()

    # extract file name and path
    split_path = os.path.split(args.file)
    path = os.path.join(*split_path[:-1])
    file_name = split_path[-1]

    # extract other parameters
    n = args.string_length
    no_overlap = args.no_overlap
    padding = args.padding

    # read data
    data_file = open(args.file, 'r')
    data = [line.strip() for line in data_file.readlines()]
    data_file.close()

    # create output file path
    filename_decomposed = file_name.split('.')
    parsed_file = '_'.join(filename_decomposed[:-1]) + '_parsed_n' + str(n) + '.' + filename_decomposed[-1]
    parsed_path = os.path.join(args.destination_path if args.destination_path else path, parsed_file)

    # open output file
    file = open(parsed_path, 'w')

    # parse all lines to fixed length words, with an extra newline between the original lines
    for line in data:
        if no_overlap:
            # non-overlapping chunks
            for j in range(math.ceil(len(line) / n)):
                chunk = line[j * n:j * n + n]
                if len(chunk) < n:
                    chunk += padding * (n - len(chunk))
                file.write(chunk + '\n')
        else:
            # sliding window
            line += padding * (max(0, n - len(line)))
            for j in range(len(line) - (n - 1)):
                file.write(line[j:j + n] + '\n')
        # newline to signal the end of the original line
        file.write('\n')
    file.close()


if __name__ == "__main__":
    main()

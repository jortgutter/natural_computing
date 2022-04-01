import os
import math
import argparse


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Preprocess system call files to a fixed word length per line')

    parser.add_argument('path', help='Path to folder containing the input file')
    parser.add_argument('data_file', type=str, help='Datafile to be parsed')
    parser.add_argument('string_length', metavar='n', type=int, help='Target word length')
    parser.add_argument('-padding', dest='padding', default='a', help="Symbol used for padding. Default: 'a'")
    parser.add_argument('-dest', dest='destination_path', default=False,
                        help='Optional path of the output file. If not set, output files will be saved in the input '
                             'folder')
    parser.add_argument('--no-overlap', dest='no_overlap', action='store_const', const=True, default=False,
                        help = "Parser will use sliding windows unless this flag is set")
    args = parser.parse_args()

    n = args.string_length
    data_filename_decomposed = args.data_file.split('.')
    no_overlap = args.no_overlap
    padding = args.padding

    # read data
    data = [line.strip() for line in open(os.path.join(args.path, args.data_file)).readlines()]

    # create output file paths
    parsed_file = '_'.join(data_filename_decomposed[:-1]) + '_parsed_n' + str(n) + '.' + data_filename_decomposed[-1]
    parsed_path = os.path.join(args.destination_path if args.destination_path else args.path, parsed_file)

    # open output files
    file = open(parsed_path, 'w')

    # parse all lines to fixed length words, with an extra newline between the original lines
    for line in data:
        if no_overlap:
            # non-overlapping chunks
            for j in range(math.ceil(len(line)/n)):
                chunk = line[j*n:j*n+n]
                if len(chunk) < n:
                    chunk += padding*(n - len(chunk))
                file.write(chunk + '\n')
        else:
            # sliding window
            line+=padding*(max(0, n-len(line)))
            for j in range(len(line)-(n-1)):
                file.write(line[j:j+n] + '\n')
        # newline to signal the end of the original line
        file.write('\n')
    file.close()


if __name__ == "__main__":
    main()

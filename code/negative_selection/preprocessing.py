import os
import math
import argparse


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Preprocesses system call files to a fixed word length per line')

    parser.add_argument('path', type=str, help='path to folder containing datafiles to be parsed')
    parser.add_argument('string_length', metavar='n', type=int, help='Target word length')
    parser.add_argument('-padding', dest='padding', default='a', help="Symbol used for padding. Default: 'a'")
    parser.add_argument('-dest', dest='destination_path', default=False,
                        help='Optional path of the output files. If not set, output files will be saved in'
                             ' a separate folder within the input folder')
    parser.add_argument('--sliding-window', dest='sliding_window', action='store_const', const=True, default=False,
                        help="Parser will use sliding windows when this flag is set")
    args = parser.parse_args()

    subfolder = 'n'+str(args.string_length)
    sub_path = os.path.join(args.path, subfolder)
    dir_list = os.listdir(args.path)

    # create sub folder if it doesn't exist yet
    if subfolder not in dir_list and not args.destination_path:
        os.mkdir(sub_path)
    for f in dir_list:
        filename_decomp = f.split('.')
        if filename_decomp[-1] == 'train' or filename_decomp[-1] == 'test':
            file_path = os.path.join(args.path, f)
            target_file_name = 'train' if filename_decomp[-1] == 'train' else filename_decomp[-2]
            target_file_path = os.path.join(args.destination_path if args.destination_path else sub_path, target_file_name)
            parse_file(file_path, target_file_path, args.string_length, args.padding, args.sliding_window)


def parse_file(file_path: str, target_file_path: str, n, padding: str, sliding_window):

    # read data
    data_file = open(file_path, 'r')
    data = [line.strip() for line in data_file.readlines()]
    data_file.close()

    # open output file
    file = open(target_file_path, 'w')

    # parse all lines to fixed length words, with an extra newline between the original lines
    for line in data:
        if sliding_window:
            # sliding window
            line += padding * (max(0, n - len(line)))
            for j in range(len(line) - (n - 1)):
                file.write(line[j:j + n] + '\n')
        else:
            # non-overlapping chunks
            for j in range(math.ceil(len(line) / n)):
                chunk = line[j * n:j * n + n]
                if len(chunk) < n:
                    chunk += padding * (n - len(chunk))
                file.write(chunk + '\n')
        # newline to signal the end of the original line
        file.write('\n')
    file.close()


if __name__ == "__main__":
    main()

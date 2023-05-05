import numpy as np
import os.path as path
from utils import load_lines, save_lines
import argparse

def filter(args, split):
    src_lines = load_lines(path.join(args.data_path, f'{split}.{args.slang}'))
    tgt_lines = load_lines(path.join(args.data_path, f'{split}.{args.tlang}'))
    assert len(src_lines) == len(tgt_lines)
    src_lens = [len(line.split()) for line in src_lines]
    tgt_lens = [len(line.split()) for line in tgt_lines]
    ratio = np.array(tgt_lens) / np.array(src_lens)
    nsamples = len(src_lines)
    src_lines = [line for i, line in enumerate(src_lines) if ratio[i] <= 2.0]
    tgt_lines = [line for i, line in enumerate(tgt_lines) if ratio[i] <= 2.0]
    assert len(src_lines) == len(tgt_lines)
    nremoved = nsamples - len(src_lines)
    print(f'{nremoved} samples removed from {split}.')
    if nremoved > 0:
        save_lines(path.join(args.data_path, f'{split}.{args.slang}'), src_lines)
        save_lines(path.join(args.data_path, f'{split}.{args.tlang}'), tgt_lines)
        print('File saved:', path.join(args.data_path, f'{split}.{args.slang}'))
        print('File saved:', path.join(args.data_path, f'{split}.{args.tlang}'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', default='../exp_raw_nos/iwslt17-doc.segmented.en-de')
    parser.add_argument('--slang', default='en')
    parser.add_argument('--tlang', default='de')
    args = parser.parse_args()

    filter(args, 'train')
    filter(args, 'valid')

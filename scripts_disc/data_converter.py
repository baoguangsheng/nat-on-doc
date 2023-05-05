import gc
import glob
import argparse
import logging
import os
import os.path as path
import numpy as np
from .utils import load_lines, save_lines

logger = logging.getLogger()

def load_file(file_name):
    lines = load_lines(file_name)
    docs = [line.split(' _eos ') for line in lines]
    outlines = []
    for doc in docs:
        outlines.append('<d>')
        outlines.extend(doc)
    return outlines

def convert_data(args, fromfold, tofold):
    assert args.source_lang == 'en' and args.target_lang == 'ru'
    # convert train data
    for corpus in ['train', 'dev', 'test']:
        lines_en = load_file(path.join(args.datadir, fromfold, 'en_%s' % corpus))
        lines_ru = load_file(path.join(args.datadir, fromfold, 'ru_%s' % corpus))
        assert len(lines_en) == len(lines_ru)
        destdir = path.join(args.destdir, tofold)
        os.makedirs(destdir, exist_ok=True)
        save_lines(path.join(destdir, 'concatenated_en2ru_%s_en.txt' % corpus), lines_en)
        save_lines(path.join(destdir, 'concatenated_en2ru_%s_ru.txt' % corpus), lines_ru)
        logger.info('Saved %s lines into file: %s' % (len(lines_en), path.join(destdir, 'concatenated_en2ru_%s_en.txt' % corpus)))
        logger.info('Saved %s lines into file: %s' % (len(lines_ru), path.join(destdir, 'concatenated_en2ru_%s_ru.txt' % corpus)))

def merge_data(args, tofold):
    assert args.source_lang == 'en' and args.target_lang == 'ru'
    # convert train data
    for corpus in ['train', 'dev', 'test']:
        lines_en = load_file(path.join(args.datadir, 'context_aware', 'en_%s' % corpus))
        lines_ru = load_file(path.join(args.datadir, 'context_aware', 'ru_%s' % corpus))
        lines = list(zip(lines_en, lines_ru))

        if corpus == 'train':
            lines_en = load_file(path.join(args.datadir, 'context_agnostic', 'en_%s' % corpus))
            lines_ru = load_file(path.join(args.datadir, 'context_agnostic', 'ru_%s' % corpus))
            lines_agno = list(zip(lines_en, lines_ru))

            pairs = set(lines)
            lines_agno = [line for line in lines_agno if line not in pairs]
            for line in lines_agno:
                lines.append(('<d>', '<d>'))
                lines.append(line)

        lines_en = [en for en, ru in lines]
        lines_ru = [ru for en, ru in lines]
        assert len(lines_en) == len(lines_ru)
        destdir = path.join(args.destdir, tofold)
        os.makedirs(destdir, exist_ok=True)
        save_lines(path.join(destdir, 'concatenated_en2ru_%s_en.txt' % corpus), lines_en)
        save_lines(path.join(destdir, 'concatenated_en2ru_%s_ru.txt' % corpus), lines_ru)
        logger.info('Saved %s lines into file: %s' % (len(lines_en), path.join(destdir, 'concatenated_en2ru_%s_en.txt' % corpus)))
        logger.info('Saved %s lines into file: %s' % (len(lines_ru), path.join(destdir, 'concatenated_en2ru_%s_ru.txt' % corpus)))


def convert_scoringdata(args, fromfold, tofold):
    assert args.source_lang == 'en' and args.target_lang == 'ru'
    # convert train data
    corpuses = {'deixis': 'deixis_test', 'ellinfl': 'ellipsis_infl', 'ellvp': 'ellipsis_vp', 'lexcohe': 'lex_cohesion_test'}
    for corpus in corpuses:
        lines_en = load_file(path.join(args.datadir, fromfold, '%s.src' % corpuses[corpus]))
        lines_ru = load_file(path.join(args.datadir, fromfold, '%s.dst' % corpuses[corpus]))
        assert len(lines_en) == len(lines_ru)
        destdir = path.join(args.destdir, tofold)
        os.makedirs(destdir, exist_ok=True)
        save_lines(path.join(destdir, 'concatenated_en2ru_%s_en.txt' % corpus), lines_en)
        save_lines(path.join(destdir, 'concatenated_en2ru_%s_ru.txt' % corpus), lines_ru)
        logger.info('Saved %s lines into file: %s' % (len(lines_en), path.join(destdir, 'concatenated_en2ru_%s_en.txt' % corpus)))
        logger.info('Saved %s lines into file: %s' % (len(lines_ru), path.join(destdir, 'concatenated_en2ru_%s_ru.txt' % corpus)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-lang", default='en')
    parser.add_argument("--target-lang", default='ru')
    parser.add_argument('--datadir', default='data-cadec/orig/')
    parser.add_argument("--destdir", default='data-cadec/')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, filename='./data_converter.log', format="[%(asctime)s %(levelname)s] %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("[%(asctime)s %(levelname)s] %(message)s"))
    logger.addHandler(console_handler)

    os.makedirs(args.destdir, exist_ok=True)

    convert_scoringdata(args, 'scoring_data', 'score')
    convert_data(args, 'context_aware', 'doc')
    convert_data(args, 'context_agnostic', 'sent')

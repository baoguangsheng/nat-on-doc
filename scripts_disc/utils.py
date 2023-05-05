import numpy as np
import codecs
import re

def load_lines_special(file_name):
    with codecs.open(file_name, 'r', 'utf-8') as fin:
        lines = [line.strip() for line in fin.read().split('\n')]
        if len(lines[-1]) == 0:
            lines = lines[:-1]
        return lines

def load_lines(file_name):
    with codecs.open(file_name, 'r', 'utf-8') as fin:
        lines = [line.strip() for line in fin.readlines()]
        return lines

def save_lines(file_name, lines):
    with codecs.open(file_name, 'w', 'utf-8') as fout:
        for line in lines:
            print(line, file=fout)

def remove_seps(text):
    sents = [s.strip() for s in text.split('</s> <s>')]
    sents = [s.replace('<s>', '').replace('</s>', '').strip() for s in sents]
    return sents

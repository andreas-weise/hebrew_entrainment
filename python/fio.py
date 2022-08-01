import csv
import nltk
import os
import subprocess

import aux
import cfg
import db



################################################################################
#                            GET PATH AND FILE NAMES                           #
################################################################################

def get_lmn_pfn(corpus_id, tsk_or_ses, tsk_ses_id, a_or_b):
    ''' returns path and file name (w/o extension) for lm/ngram files '''
    return cfg.get_lmn_path(corpus_id), \
        '%s_%d_%s' % (tsk_or_ses, tsk_ses_id, a_or_b)



################################################################################
#                                 WRITE FILES                                  #
################################################################################

def remove_lmn_files(corpus_id, tsk_or_ses=None, extension='txt'):
    ''' removes lm/ngram files with given extension for tasks/sessions/both '''
    if tsk_or_ses is None:
        # if tsk_or_ses not specified, do both
        remove_lmn_files(corpus_id, 'tsk', extension)
        remove_lmn_files(corpus_id, 'ses', extension)
    else:
        for tsk_ses_id in db.get_tsk_ses_ids(tsk_or_ses):
            for a_or_b in ['A', 'B']:
                path, fname = get_lmn_pfn(
                    corpus_id, tsk_or_ses, tsk_ses_id, a_or_b) 
                if os.path.isfile(path + fname + '.' + extension):
                    os.remove(path + fname + '.' + extension)


def store_tokens(corpus_id, tsk_or_ses, lem):
    ''' writes tokens per speaker to txt file for each task/session '''
    if tsk_or_ses is None:
        store_tokens(corpus_id, 'tsk', lem)
        store_tokens(corpus_id, 'ses', lem)
    else:
        remove_lmn_files(corpus_id, tsk_or_ses, extension='txt')
        all_lemmata = []
        for tsk_ses_id in db.get_tsk_ses_ids(tsk_or_ses):
            tur_id_prev = -1
            cnts = {'A': 0, 'B': 0}
            for tur_id, a_or_b, words in db.get_words(tsk_or_ses, tsk_ses_id):
                path, fname = get_lmn_pfn(
                    corpus_id, tsk_or_ses, tsk_ses_id, a_or_b) 
                with open(path + fname + '.txt', 'a') as txt_file:
                    if cnts[a_or_b] > 0:
                        if tur_id_prev != tur_id:
                            txt_file.write('\n')
                        else:
                            txt_file.write(' ')
                    words_lem = lem(words)
                    all_lemmata += words_lem
                    txt_file.write(' '.join(words_lem))
                cnts[a_or_b] += 1
                tur_id_prev = tur_id
        vocab = sorted(list(nltk.FreqDist(all_lemmata).keys()))
        with open(cfg.get_vocab_fname(corpus_id), 'w') as vocab_file:
            vocab_file.write('\n'.join(vocab))



################################################################################
#                                  READ FILES                                  #
################################################################################

def readlines(path, fname):
    ''' yields all lines in given file '''
    with open(path + fname) as file:
        for line in file.readlines():
            yield(line)


def read_csv(path, fname, delimiter=",", quotechar='"', skip_header=False):
    ''' yields all rows in given file, interpreted as csv '''
    with open(path + fname, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=delimiter, quotechar=quotechar)
        if skip_header:
            next(reader)
        for row in reader:
            yield(row)


def load_tokens(
        corpus_id, tsk_or_ses, tsk_ses_id, a_or_b, excl=[], func=lambda t: t):
    ''' loads tokens of given task/session & speaker, returns filtered list '''
    path, fname = get_lmn_pfn(corpus_id, tsk_or_ses, tsk_ses_id, a_or_b)
    if os.path.isfile(path + fname + '.txt'):
        tokens = '\n'.join(readlines(path, fname + '.txt')).replace('\n', ' ')
        tokens = [func(t) for t in tokens.split() if t not in excl]
    else:
        tokens = []
    return tokens


def load_all_tokens(corpus_id):
    ''' loads list of tokens for all sessions and speakers from files '''
    tokens = []
    for ses_id in db.get_ses_ids():
        for a_or_b in ['A', 'B']:
            path, fname = get_lmn_pfn(corpus_id, 'ses', ses_id, a_or_b)
            lines = readlines(path, fname + '.txt')
            tokens += '\n'.join(lines).replace('\n', ' ').split()
    return tokens



################################################################################
#                                     OTHER                                    #
################################################################################

def extract_features(in_path, in_fname, tsk_id, chu_id, words, start, end):
    ''' runs feature extraction for given chunk section, returns features '''
    # determine tmp filenames
    cut_fname = '%d_%d.wav' % (tsk_id, chu_id)
    out_fname = '%d_%d.txt' % (tsk_id, chu_id)
    # extract audio and features
    subprocess.check_call(['sox', 
                           in_path + in_fname, 
                           cfg.TMP_PATH + cut_fname, 
                           'trim', str(start), '=' + str(end)])
    subprocess.check_call(['praat', '--run', 
                           cfg.PRAAT_SCRIPT_FNAME,
                           cfg.TMP_PATH + cut_fname, 
                           cfg.TMP_PATH + out_fname])
    # read output
    features = {}
    for line in readlines(cfg.TMP_PATH, out_fname):
        key, val = line.replace('\n', '').split(',')
        try:
            val = float(val)
        except:
            val = None
        features[key] = val
    features['rate_syl'] = aux.count_syllables(words) / (end - start)
    # clean up
    os.remove(cfg.TMP_PATH + cut_fname)
    os.remove(cfg.TMP_PATH + out_fname)
    
    return features






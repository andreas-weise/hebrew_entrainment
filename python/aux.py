import hyphenate
import math
import nltk
from nltk.corpus import wordnet
import pandas as pd
import scipy

import cfg

cmu_dict = nltk.corpus.cmudict.dict()



def preprocess_transcript(transcript):
    ''' prepocessing of transcripts of individual utterances '''
    words = transcript.replace('\n', '')
    # @ marks unintelligible text; not very common, simply ignore
    words = words.replace('@ ', '')
    words = words.replace('@', '')
    # remove markup (silence, noises)
    while words.find('<') != -1:
        words = words[:words.find('<')] + words[words.find('>')+1:]
    # 'condense' double spaces
    words = ' '.join(words.split())
    # remove empty overlaps
    if words == '[start_overlap][end_overlap]':
        words = ''
    return words


def count_syllables(words):
    ''' returns number of syllables in given string '''
    # each vowel and each consonant-only word counts as one syllable
    return sum([max(1, sum(map(w.lower().count, 'aeiou'))) 
                for w in words.split()])


def get_df(data, index_names):
    ''' creates pandas dataframe from given data with given index names '''
    df = pd.DataFrame(data)
    df.index.set_names(index_names, inplace=True)
    return df


def ttest_ind(a, b):
    ''' scipy.stats.ttest_ind(a, b) with degrees of freedom also returned '''
    return scipy.stats.ttest_ind(a, b) + (len(a) + len(b) - 2,)


def ttest_rel(a, b):
    ''' scipy.stats.ttest_rel(a, b) with degrees of freedom also returned '''
    return scipy.stats.ttest_rel(a, b) + (len(a) - 1,)


def pearsonr(x, y):
    ''' scipy.stats.pearsonr(x, y) with degrees of freedom also returned '''
    return scipy.stats.pearsonr(x, y) + (len(x) - 2,)


def r2z(r):
    ''' fisher z-transformation of a pearson correlation coefficient '''
    return 0.5 * (math.log(1 + r) - math.log(1 - r))



































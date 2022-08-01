import collections
import math
import nltk
import os
import pandas as pd
import subprocess

import aux
import cfg
import db
import fio


def load_lex(fname):
    ''' load a standard lexicon file, a list of lemmata, one per line '''
    return [line.replace('\n', '') 
            for line in fio.readlines(cfg.LEX_PATH_HMT, fname)]


def load_lex_w_lem(fname):
    ''' load a lexicon with lemmatization '''
    lem = collections.defaultdict(str)
    for line in load_lex(fname):
        items = line.split('\t')
        if items[0] != '':
            # lines with items[0] mark the beginning of a new section; types in
            # this and subsequent lines are all grammatical forms of the lemma
            lemma = items[0]
        lem[items[2]] = lemma
    lem_func = lambda t: [lem[w] for w in t.split()]
    return list(set(lem.values())), list(lem.keys()), lem_func


def store_lms_ngrams(corpus_id, tsk_or_ses=None):
    ''' computes lm and ngram files for all tasks/sessions using srilm

    based on txt file per session/task stored beforehand (fio.store_tokens)'''
    if tsk_or_ses is None:
        store_lms_ngrams(corpus_id, 'tsk')
        store_lms_ngrams(corpus_id, 'ses')
    else:
        fio.remove_lmn_files(corpus_id, tsk_or_ses, 'lm')
        fio.remove_lmn_files(corpus_id, tsk_or_ses, 'cnt')
        for tsk_ses_id in db.get_tsk_ses_ids(tsk_or_ses):
            for a_or_b in ['A', 'B']:
                path, fname = fio.get_lmn_pfn(
                    corpus_id, tsk_or_ses, tsk_ses_id, a_or_b)
                subprocess.check_call(
                    ['ngram-count', 
                     '-lm', path + fname + '.lm', 
                     '-text', path + fname + '.txt', 
                     '-vocab', cfg.get_vocab_fname(corpus_id)])
                subprocess.check_call(
                    ['ngram-count', 
                     '-text', path + fname + '.txt', 
                     '-write', path + fname + '.cnt', 
                     '-vocab', cfg.get_vocab_fname(corpus_id)])


def mem_entropy(f):
    ''' memoization function for get_entropy '''
    memo = {}
    def helper(corpus_id, tsk_or_ses, tsk_ses_id, spk_id):
        params = (corpus_id, tsk_or_ses, tsk_ses_id, spk_id)
        if params not in memo:
            memo[params] = f(*params)
        return memo[params]
    return helper


@mem_entropy
def get_entropy(corpus_id, tsk_or_ses, tsk_ses_id, spk_id):
    ''' computes entropy for given task/session and speaker using srilm 

    based on lm and cnt files stored beforehand (store_lms_ngrams)'''
    a_or_b = db.get_a_or_b(tsk_or_ses, tsk_ses_id, spk_id)
    path, fname = fio.get_lmn_pfn(corpus_id, tsk_or_ses, tsk_ses_id, a_or_b)
    proc = subprocess.run(
        ['ngram', 
         '-lm', path + fname + '.lm', 
         '-counts', path + fname + '.cnt', 
         '-counts-entropy'], 
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
        universal_newlines=True, check=True)
    outputs = proc.stdout.split('\n')[1].split()
    if outputs[2] != 'logprob=':
        print(outputs)
        raise Exception('unexpected output for entropy!')
    return -float(outputs[3])


def get_entropy_pairs(corpus_id, df_spk_pairs):
    ''' gets entropies for speaker pairs (only actual partners) '''
    # auxiliary column lists
    loc_cols1 = ['ses_id', 'tsk_id', 'spk_id']
    loc_cols2 = ['ses_id_paired', 'tsk_id_paired', 'spk_id_paired']
    # filter for relevant rows (partners only, exclude non-partners) and columns
    df_ent2 = df_spk_pairs[df_spk_pairs['p_or_x'] == 'p']
    df_ent2 = df_ent2.loc[:,loc_cols1+loc_cols2]
    # entropy of main speaker
    func = lambda x: \
        get_entropy(
            corpus_id, 
            'ses' if x['tsk_id'] == 0 else 'tsk',
            int(x['ses_id']) if x['tsk_id'] == 0 else int(x['tsk_id']),
            int(x['spk_id']))
    df_ent2['entropy'] = df_ent2.apply(func, axis=1)
    # entropy of partner
    func = lambda x: \
        get_entropy(
            corpus_id, 
            'ses' if x['tsk_id'] == 0 else 'tsk',
            int(x['ses_id_paired']) if x['tsk_id'] == 0 
                else int(x['tsk_id_paired']),
            int(x['spk_id_paired']))
    df_ent2['entropy_partner'] = df_ent2.apply(func, axis=1)
    df_ent2.set_index(loc_cols1, inplace=True)
    df_ent2.drop(loc_cols2, axis=1, inplace=True)
    return df_ent2


def get_entropy_triplets(corpus_id, df_spk_pairs):
    ''' gets entropies for speaker, original partner, and (non)-partner '''
    # auxiliary column lists
    loc_cols = ['ses_id', 'tsk_id', 'spk_id']
    loc_cols2 = ['ses_id_paired', 'tsk_id_paired', 'spk_id_paired']
    # get entropy pairs for speaker and actual partner
    df_ent2 = get_entropy_pairs(corpus_id, df_spk_pairs)
    # get entropy triplets for speaker, partner, and partner or non-partner
    # (partner entropy is repeated for 'p' speaker pairs)
    df_ent3 = df_spk_pairs.join(df_ent2, on=loc_cols)
    df_ent2.drop('entropy_partner', axis=1, inplace=True)
    df_ent3 = df_ent3.join(df_ent2, on=loc_cols2, rsuffix='_paired')
    return df_ent3


def get_entropy_weights(corpus_id, df_spk_pairs):
    ''' computes entropy diff. weights for non-partners replacing given speaker 

    for each non-partner, the difference in entropy with the actual partner is
    computed, then all non-partners are weighted based on those differences, 
    with non-partners of more similar entropy receiving greater weight

    args:
        corpus_id: one of the constants defined in cfg, identifying the corpus
        df_spk_pairs: pandas dataframe with columns identifying speaker pairs
            (partners and non-partners), loaded through cfg.SQL_SP_FNAME script
    '''
    # get entropies of speaker, partner, and (non)-partner in each row
    df_ent3 = get_entropy_triplets(corpus_id, df_spk_pairs)
    # compute difference between actual partner and paired speaker
    # (0 if partner compared to partner; this is intentional and handled below)
    diffs = abs(df_ent3['entropy_partner'] - df_ent3['entropy_paired'])
    diffs.name = 'entropy_diff'
    df_ent3 = df_ent3.join(diffs)
    # determine minimum difference per session/task and speaker
    grp_cols = ['p_or_x', 'ses_id', 'tsk_id', 'spk_id']
    df_mins = df_ent3.loc[:,grp_cols+['entropy_diff']].groupby(grp_cols).min()
    df_ent3 = df_ent3.join(df_mins, on=grp_cols, rsuffix='_min')
    # compute raw weights based on how entropy difference compares to minimum
    func = lambda x: \
        1 if x['entropy_diff'] == 0 \
        else x['entropy_diff_min'] / x['entropy_diff']
    df_ent3['weight_raw'] = df_ent3.apply(func, axis=1)
    # compute final weight as fraction of overall weights
    df_sum = df_ent3.loc[:,grp_cols+['weight_raw']].groupby(grp_cols).sum()
    df_ent3 = df_ent3.join(df_sum, on=grp_cols, rsuffix='_sum')
    func = lambda x: x['weight_raw'] / x['weight_raw_sum']
    df_ent3['weight'] = df_ent3.apply(func, axis=1)
    # return final results, with intermediate results removed
    aux_cols = [
        'entropy', 'entropy_partner', 'entropy_paired', 'entropy_diff',
        'entropy_diff_min', 'weight_raw', 'weight_raw_sum'
    ]
    return df_ent3.drop(aux_cols, axis=1)


def mem_perplexity(f):
    ''' memoization function for get_perplexity '''
    memo = {}
    def helper(
            corpus_id, tsk_or_ses, tsk_ses_id1, a_or_b1, tsk_ses_id2, a_or_b2):
        params = (
            corpus_id, tsk_or_ses, tsk_ses_id1, a_or_b1, tsk_ses_id2, a_or_b2)
        if params not in memo:
            memo[params] = f(*params)
        return memo[params]
    return helper


@mem_perplexity
def get_perplexity(
        corpus_id, tsk_or_ses, tsk_ses_id1, a_or_b1, tsk_ses_id2, a_or_b2):
    ''' computes perplexity of one speaker's lm predicting another's utterances 

    language model of one speaker in one interaction used to predict utterances
    of other speaker in other interaction, perplexity computed using srilm

    args:
        corpus_id: one of the constants defined in cfg, identifying the corpus
        tsk_or_ses: whether given interactions are tasks or sessions
        tsk_ses_id1: tsk_id/ses_id (see tsk_or_ses) for language model 
        a_or_b1: which speaker's (from tsk_ses_id1) language model is used
        tsk_ses_id2: tsk_id/ses_id (see tsk_or_ses) for utterances to predict
        a_or_b1: which speaker's (from tsk_ses_id2) utterances are predicted
    returns:
        float representing perplexity (lower values mean greater sim)
    '''
    path1, fname1 = fio.get_lmn_pfn(corpus_id, tsk_or_ses, tsk_ses_id1, a_or_b1)
    fname1 = path1 + fname1 + '.lm'
    path2, fname2 = fio.get_lmn_pfn(corpus_id, tsk_or_ses, tsk_ses_id2, a_or_b2)
    fname2 = path2 + fname2 + '.txt'

    if os.path.isfile(fname1) and os.path.isfile(fname2):
        comp_proc = subprocess.run(
            ['ngram', '-lm', fname1, '-ppl', fname2, 
             '-vocab', cfg.get_vocab_fname(corpus_id)], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
            universal_newlines=True, check=True)
        outputs = comp_proc.stdout.split('\n')[1].split()
        if outputs[4] != 'ppl=':
            print(outputs)
            raise Exception('unexpected output for perplexity!')
        perplexity = float(outputs[5])
    else:
        perplexity = math.nan
    return perplexity


def mem_token_count(f):
    ''' memoization function for get_token_count '''
    memo = {}
    def helper(corpus_id, tsk_or_ses, tsk_ses_id, spk_id):
        params = (corpus_id, tsk_or_ses, tsk_ses_id, spk_id)
        if params not in memo:
            memo[params] = f(*params)
        return memo[params]
    return helper


@mem_token_count
def get_token_count(corpus_id, tsk_or_ses, tsk_ses_id, a_or_b):
    ''' loads txt for given session/task and speaker and returns token count '''
    return len(fio.load_tokens(corpus_id, tsk_or_ses, tsk_ses_id, a_or_b))


def get_mf_types(corpus_id, count=25, neg=[]):
    ''' returns given count of most frequent types, ignoring given list '''
    tokens = [t for t in fio.load_all_tokens(corpus_id) if t not in neg]
    return [v[0] for v in nltk.FreqDist(tokens).most_common(count)]


def mem_dist(f):
    ''' memoization function for get_dist '''
    memo = {}
    def helper(corpus_id, tsk_or_ses, tsk_ses_id, a_or_b, types_id=None, 
               excl_id=None, include_zero=True, func=lambda t: t, types=[], 
               excl=[]):
        params = (corpus_id, tsk_or_ses, tsk_ses_id, a_or_b, types_id, excl_id, 
                  include_zero, func)
        if params not in memo:
            memo[params] = f(*params, types, excl)
        return memo[params]
    return helper


@mem_dist
def get_dist(
        corpus_id, tsk_or_ses, tsk_ses_id, a_or_b, types_id=None, 
        excl_id=None, include_zero=True, func=lambda t: t, types=[], excl=[]):
    ''' returns distribution of types for given task/session and speaker

    determines how often each type from optional list was spoken in given
    interaction by given speaker and divides it by total number of tokens
    from that speaker in that interaction (with filters and processing)

    types_id, excl_id included for memoization; dict that stores past results 
    cannot be indexed by lists since lists are unhashable
    
    args:
        corpus_id: one of the constants defined in cfg, identifying the corpus
        tsk_or_ses: whether to load tokens for a task or session
        tsk_ses_id: id of task or session whose tokens are loaded
        a_or_b: which speaker's (from tsk_ses_id) tokens are loaded
        types_id: identifier corresponding to given types (for memoization)
        excl_id: identifier corresponding to given excl (for memoization)
        include_zero: whether to include types that did not occur among tokens
            in output with fraction 0.0 or not at all
        types: whitelist of types for which to determine distribution (sum 1)
        excl: blacklist of types to exclude from analysis
    returns:
        dict mapping types to percentages of speaker tokens represented by them
    '''
    tokens = fio.load_tokens(
        corpus_id, tsk_or_ses, tsk_ses_id, a_or_b, excl, func)
    if len(types) > 0:
        tokens = [t for t in tokens if t in types]
    else:
        types = list(set(tokens))
    freqs = dict(nltk.FreqDist(tokens))
    if include_zero:
        res = {t: (freqs[t]/len(tokens) if t in freqs else 0.0) for t in types}
    else:
        res = {t: freqs[t]/len(tokens) for t in types if t in freqs}
    return res


def mem_dist_comp(f):
    ''' memoization function for compare_dists '''
    memo = {}
    def helper(corpus_id, tsk_or_ses, tsk_ses_id1, a_or_b1, tsk_ses_id2, 
               a_or_b2, types_id=None, excl_id=None, include_zero=True, 
               func=lambda t: t, types=[], excl=[]):
        params1 = (corpus_id, tsk_or_ses, tsk_ses_id1, a_or_b1, tsk_ses_id2, 
                   a_or_b2, types_id, excl_id, include_zero, func)
        params2 = (corpus_id, tsk_or_ses, tsk_ses_id2, a_or_b2, tsk_ses_id1, 
                   a_or_b1, types_id, excl_id, include_zero, func)
        if params1 not in memo and params2 not in memo:
            memo[params1] = f(*params1, types, excl)
        # at this point, either params1 or params2 is in memo
        params = params1 if params1 in memo else params2
        return memo[params]
    return helper


@mem_dist_comp
def compare_dists(
        corpus_id, tsk_or_ses, tsk_ses_id1, a_or_b1, tsk_ses_id2, a_or_b2, 
        types_id=None, excl_id=None, include_zero=True, func=lambda t: t, 
        types=[], excl=[]):
    ''' loads and compares token dists for two given tasks/sessions & speakers 

    distributions loaded instead of given directly to allow for memoization
    for args, see get_dist
    '''
    params = (types_id, excl_id, include_zero, func, types, excl)
    dist1 = get_dist(corpus_id, tsk_or_ses, tsk_ses_id1, a_or_b1, *params)
    dist2 = get_dist(corpus_id, tsk_or_ses, tsk_ses_id2, a_or_b2, *params)
    types = types if len(types) > 0 \
        else list(set(dist1.keys()).union(set(dist2.keys())))
    return sum([-abs((dist1[t] if t in dist1 else 0.0) 
                    -(dist2[t] if t in dist2 else 0.0)) 
                for t in types])


def ppl(corpus_id, df_spk_pairs):
    ''' perplexity of predicting partner's utterances from speaker lm '''
    df_spk_pairs = df_spk_pairs.copy()
    
    # compute negated perplexity for all partner and non-partner pairs
    func = lambda x: -get_perplexity(
        corpus_id,
        'ses' if x['tsk_id'] == 0 else 'tsk',
        x['ses_id'] if x['tsk_id'] == 0 else x['tsk_id'],
        x['a_or_b'],
        x['ses_id_paired'] if x['tsk_id'] == 0 else x['tsk_id_paired'],
        x['a_or_b_paired'])
    df_spk_pairs['ppl'] = df_spk_pairs.apply(func, axis=1)
    # weight perplexity values by entropy difference with actual partner
    weighted = df_spk_pairs['ppl'] * df_spk_pairs['weight']
    weighted.name = 'ppl_wgh'
    df_spk_pairs = df_spk_pairs.join(weighted)
    # determine token count per task/session and speaker for normalization
    # (many redundant calls to get_token_count, but it is memoized)
    func = lambda x: get_token_count(
        corpus_id,
        'ses' if x['tsk_id'] == 0 else 'tsk',
        x['ses_id']  if x['tsk_id'] == 0 else x['tsk_id'], 
        x['a_or_b'])
    df_spk_pairs['token_count'] = df_spk_pairs.apply(func, axis=1)
    # get mean perplexity and token count for partners and non-partners per
    # task/session and speaker (mean of one value for partners, several for non;
    # *same* token count for all non-partners per task/session and speaker)
    grp_cols = [
        'p_or_x', 'ses_type', 'ses_id', 'gender_pair', 'tsk_id', 'spk_id']
    loc_cols = grp_cols + ['ppl', 'token_count']
    df_ppl = df_spk_pairs.loc[:,loc_cols].groupby(grp_cols).mean()
    # get *weighted* mean perplexity for partners and non-partners per
    # task/session and speaker (cannot use mean(), need to use sum())
    loc_cols = grp_cols + ['ppl_wgh']
    df_ppl_wgh = df_spk_pairs.loc[:,loc_cols].groupby(grp_cols).sum()
    df_ppl = df_ppl.join(df_ppl_wgh)
    # self-join to get values for both partners and non-partners in each row
    df_ppl = pd.DataFrame(df_ppl.xs('p', level=0)).join( 
        df_ppl.xs('x', level=0), lsuffix='_p', rsuffix='_x')
    # compute raw and weighted measures normalized by token count
    func = lambda x: x['ppl_p'] / x['token_count_p']
    df_ppl['ppl_nrm_p'] = df_ppl.apply(func, axis=1)
    func = lambda x: x['ppl_x'] / x['token_count_x']
    df_ppl['ppl_nrm_x'] = df_ppl.apply(func, axis=1)
    func = lambda x: x['ppl_wgh_p'] / x['token_count_p']
    df_ppl['ppl_wgh_nrm_p'] = df_ppl.apply(func, axis=1)
    func = lambda x: x['ppl_wgh_x'] / x['token_count_x']
    df_ppl['ppl_wgh_nrm_x'] = df_ppl.apply(func, axis=1)
    df_ppl.drop(['token_count_p', 'token_count_x'], axis=1, inplace=True)
    # compute mean per task/session as symmetric version of the measure
    # (not used below but included in df_results_raw for later use)
    df_ppl2 = df_ppl.groupby(
        ['ses_type', 'ses_id', 'gender_pair', 'tsk_id']).mean()
    # add spk_id to index for consistent interface
    df_ppl2['spk_id'] = 0
    df_ppl2.set_index('spk_id', append=True, inplace=True)
    df_results_raw = pd.concat([df_ppl, df_ppl2], axis=0)
    # add baseline normalized versions of the measure
    df_results_raw['ppl_nrm'] = \
        -df_results_raw['ppl_p'] / df_results_raw['ppl_x']
    df_results_raw['ppl_wgh_nrm'] = \
        -df_results_raw['ppl_wgh_p'] / df_results_raw['ppl_wgh_x']
    # run statistical tests (on sessions only, i.e., tsk_id == 0)
    df_sub = df_ppl.xs(0, level=3)
    # exclude rows with missing non-partner values
    df_sub = df_sub[pd.notna(df_sub['ppl_x'])]
    results = {'raw': {}, 'wgh': {}}
    results['raw'][0] = aux.ttest_rel(df_sub['ppl_p'], df_sub['ppl_x'])
    results['wgh'][0] = aux.ttest_rel(df_sub['ppl_wgh_p'], df_sub['ppl_wgh_x'])
    return aux.get_df(results, ['_']), df_results_raw


def dist_sim(corpus_id, df_spk_pairs, types_id=None, excl_id=None, 
             func=lambda t: t, types=[], excl=[]):
    df_spk_pairs = df_spk_pairs.copy()

    # compute similarity of distributions for all partner and non-partner pairs
    params = (types_id, excl_id, True, func, types, excl)
    dsim_func = lambda x: \
        compare_dists(
            corpus_id,
            'ses' if x['tsk_id'] == 0 else 'tsk',
            x['ses_id'] if x['tsk_id'] == 0 else x['tsk_id'],
            x['a_or_b'],
            x['ses_id_paired'] if x['tsk_id'] == 0 else x['tsk_id_paired'],
            x['a_or_b_paired'],
            *params)
    df_spk_pairs['dsim'] = df_spk_pairs.apply(dsim_func, axis=1)
    # weight similarity values by entropy difference with actual partner
    weighted = df_spk_pairs['dsim'] * df_spk_pairs['weight']
    weighted.name = 'dsim_wgh'
    df_spk_pairs = df_spk_pairs.join(weighted)
    # get mean similarity for partners and non-partners per task/session and 
    # speaker (mean over one value for partners, several for non)
    grp_cols = [
        'ses_type', 'p_or_x', 'ses_id', 'gender_pair', 'tsk_id', 'spk_id']
    loc_cols = grp_cols + ['dsim']
    df_dsim = df_spk_pairs.loc[:,loc_cols].groupby(grp_cols).mean()
    # get *weighted* mean similarity for partners and non-partners per
    # task/session and speaker (cannot use mean(), need to use sum())
    loc_cols = grp_cols + ['dsim_wgh']
    df_dsim_wgh = df_spk_pairs.loc[:,loc_cols].groupby(grp_cols).sum()
    df_dsim = df_dsim.join(df_dsim_wgh)
    # self-join to get values for both partners and non-partners in each row
    df_dsim = pd.DataFrame(df_dsim.xs('p', level=1)).join( 
        df_dsim.xs('x', level=1), lsuffix='_p', rsuffix='_x')
    # compute mean per task/session as symmetric version of the measure
    # (not used below but included in df_results_raw for later use)
    df_dsim2 = df_dsim.groupby(
        ['ses_type', 'ses_id', 'gender_pair', 'tsk_id']).mean()
    # add spk_id to index for consistent interface
    df_dsim2['spk_id'] = 0
    df_dsim2.set_index('spk_id', append=True, inplace=True)
    df_results_raw = pd.concat([df_dsim, df_dsim2], axis=0)
    # add baseline normalized versions of the measure
    df_results_raw['dsim_nrm'] = \
        -df_results_raw['dsim_p'] / df_results_raw['dsim_x']
    df_results_raw['dsim_wgh_nrm'] = \
        -df_results_raw['dsim_wgh_p'] / df_results_raw['dsim_wgh_x']
    # run statistical tests (on sessions only, i.e., tsk_id == 0)
    df_sub = df_dsim.xs(0, level=3)
    # exclude rows with missing non-partner values
    df_sub = df_sub[pd.notna(df_sub['dsim_x'])]
    results = {'raw': {}, 'wgh': {}}
    results['raw'][0] = aux.ttest_rel(df_sub['dsim_p'], df_sub['dsim_x'])
    results['wgh'][0] = aux.ttest_rel(df_sub['dsim_wgh_p'], 
                                      df_sub['dsim_wgh_x'])
    return aux.get_df(results, ['_']), df_results_raw


def mem_kld(f):
    ''' memoization function for compute_kld '''
    memo = {}
    def helper(
            corpus_id, tsk_or_ses, tsk_ses_id1, a_or_b1, tsk_ses_id2=None,
            a_or_b2=None):
        params = (
            corpus_id, tsk_or_ses, tsk_ses_id1, a_or_b1, tsk_ses_id2, a_or_b2)
        if params not in memo:
            memo[params] = f(*params)
        return memo[params]
    return helper


@mem_kld
def compute_kld(
        corpus_id, tsk_or_ses, tsk_ses_id1, a_or_b1, tsk_ses_id2=None, 
        a_or_b2=None):
    # based on: Bigi, B. (2003). Using Kullback-Leibler distance for text 
    #     categorization. European Conference on Information Retrieval, 305–319.
    
    # shorthand for whether info for second speaker given
    two_given = tsk_ses_id2 and a_or_b2
    # P: type probabilities for both speakers
    P = [
        get_dist(corpus_id, tsk_or_ses, tsk_ses_id1, a_or_b1),
        get_dist(corpus_id, tsk_or_ses, tsk_ses_id2, a_or_b2) if two_given
            else collections.defaultdict(float)
    ]
    
    # V: overall vocabulary
    V = set(P[0].keys()).union(set(P[1].keys()))
    # epsilon: prob. for unseen types (lowest prob. in either list / 10)
    # (paper does not specify denominator 10, just says epsilon has to 
    #  be "smaller" than the minima and empirically determined)
    if two_given:
        epsilon = min(min(P[0].values()), min(P[1].values())) / 10.0
    else:
        epsilon = min(P[0].values()) / 10.0
    # apply backoff scheme 
    for i in [0, 1]:
        # compute and apply normalization coefficient (beta/gamma in paper)
        coeff = 1 - (len(V) - len(P[i].keys())) * epsilon
        for key, val in P[i].items():
            P[i][key] = coeff * val
        # set probabilities for unseen types
        for t in V:
            if t not in P[i]:
                P[i][t] = epsilon
    # actually compute kld
    kld = 0.0
    for t in V:
        kld += (P[0][t]) * math.log(P[0][t] / P[1][t])
    
    # normalize by kld with empty document (defaultdict with 0.0 for all keys)
    if two_given:
        kld = kld / compute_kld(corpus_id, tsk_or_ses, tsk_ses_id1, a_or_b1)
    return kld


def kld(corpus_id, df_spk_pairs):
    df_spk_pairs = df_spk_pairs.copy()

    # compute negated kld for all partner and non-partner pairs
    # (inverted speaker order for kld)
    func = lambda x: \
        -compute_kld(
            corpus_id,
            'ses' if x['tsk_id'] == 0 else 'tsk',
            x['ses_id_paired'] if x['tsk_id'] == 0 else x['tsk_id_paired'],
            x['a_or_b_paired'],
            x['ses_id'] if x['tsk_id'] == 0 else x['tsk_id'],
            x['a_or_b'])
    df_spk_pairs['kld'] = df_spk_pairs.apply(func, axis=1)
    # weight similarity values by entropy difference with actual partner
    weighted = df_spk_pairs['kld'] * df_spk_pairs['weight']
    weighted.name = 'kld_wgh'
    df_spk_pairs = df_spk_pairs.join(weighted)
    # get mean kld for partners and non-partners per task/session and speaker
    # (mean over one value for partners, several for non)
    grp_cols = [
        'ses_type', 'p_or_x', 'ses_id', 'gender_pair', 'tsk_id', 'spk_id']
    loc_cols = grp_cols + ['kld']
    df_kld = df_spk_pairs.loc[:,loc_cols].groupby(grp_cols).mean()
    # get *weighted* mean kld for partners and non-partners per task/session and
    # speaker (cannot use mean(), need to use sum())
    loc_cols = grp_cols + ['kld_wgh']
    df_kld_wgh = df_spk_pairs.loc[:,loc_cols].groupby(grp_cols).sum()
    df_kld = df_kld.join(df_kld_wgh)
    # self-join to get values for both partners and non-partners in each row
    df_kld = pd.DataFrame(df_kld.xs('p', level=1)).join( 
        df_kld.xs('x', level=1), lsuffix='_p', rsuffix='_x')
    # compute mean per task/session as symmetric version of the measure
    # (not used below but included in df_results_raw for later use;
    #  symmetric version of kld is usually the sum, using mean for consistency)
    df_kld2 = df_kld.groupby(
        ['ses_type', 'ses_id', 'gender_pair', 'tsk_id']).mean()
    # add spk_id to index for consistent interface
    df_kld2['spk_id'] = 0
    df_kld2.set_index('spk_id', append=True, inplace=True)
    df_results_raw = pd.concat([df_kld, df_kld2], axis=0)
    # add baseline normalized versions of the measure
    df_results_raw['kld_nrm'] = \
        -df_results_raw['kld_p'] / df_results_raw['kld_x']
    df_results_raw['kld_wgh_nrm'] = \
        -df_results_raw['kld_wgh_p'] / df_results_raw['kld_wgh_x']
    # run statistical tests (on sessions only, i.e., tsk_id == 0)
    df_sub = df_kld.xs(0, level=3)
    # exclude rows with missing non-partner values
    df_sub = df_sub[pd.notna(df_sub['kld_x'])]
    results = {'raw': {}, 'wgh': {}}
    results['raw'][0] = aux.ttest_rel(df_sub['kld_p'], df_sub['kld_x'])
    results['wgh'][0] = aux.ttest_rel(df_sub['kld_wgh_p'], df_sub['kld_wgh_x'])
    return aux.get_df(results, ['_']), df_results_raw









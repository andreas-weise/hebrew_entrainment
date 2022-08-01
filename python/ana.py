import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import aux
import cfg

# this module implements functions for the analysis of entrainment results



identity = lambda x:x

def _get_data(series, stat_idx, func=identity):
    ''' extracts data from tuples and applies func if needed '''
    data = series
    if stat_idx is not None:
        data = list(map(func, list(zip(*data))[stat_idx]))
    return data


def _get_chu_cnt(df_bt, tsk_or_ses, tx_only=False):
    ''' computes number of chunks per session/task, returns dataframe '''
    assert tsk_or_ses in ['tsk', 'ses'], 'unknown tsk_or_ses value'
    loc_cols = [tsk_or_ses + '_id', 'chu_id']
    if tx_only:
        # only count turn exchanges
        df_tmp = df_bt[df_bt['p_or_x'] == 'p'].loc[:,loc_cols].copy()
    else:
        # count all ipus
        df_tmp = df_bt[df_bt['p_or_x'] != 'x'].loc[:,loc_cols].copy()
    df_tmp.rename(columns={'chu_id': 'chu_cnt'}, inplace=True)
    return df_tmp.groupby(tsk_or_ses + '_id').count()


def bh(x, alpha=0.05, p_id=1):
    ''' returns significance threshold for given dataframe row 
    
    uses benjamini-hochberg procedure for significance level alpha; 
    cells assumed to contain tuples, p values stored in index p_id '''
    p_vals = sorted([x[f][p_id] for f in cfg.FEATURES])
    k = max([k if p < (k+1) * alpha / len(p_vals) else -1
             for k, p in enumerate(p_vals)])
    return p_vals[k] if k != -1 else 0.0


def check_significance(
        df, idx, index_names, cols=cfg.FEATURES, stat_idxs=[None, None]):
    ''' computes significance for measures in select columns in given data

    for each given column, uses paired t-test to either compare correlation 
    coefficients for real data with those for permutated data (components of 
    tuples in each cell), or compares partner and non-partner pair of columns

    args:
        df: pandas dataframe with a session in each row, and relevant values 
            in the given columns (as tuple) or in partner and non-partner pair
            of columns (as float)
        idx: identifier for the single row of the returned dataframe
        index_names: labels for the index dimensions of the returned dataframe
        cols: list of columns that each contain pairs of values to compare (as 
            tuple) or for which there is a partner and non-partner column in df
        stat_idxs: pair of indices of relevant entries in tuple cells; None if 
            using 
    returns:
        pandas dataframe with t-test results, one per column, in a single row
    '''
    def __get_data(df, col, stat_idx, p_or_x):
        if stat_idx is not None:
            data = list(map(aux.r2z, list(zip(*df[col]))[stat_idx]))
        else:
            data = df[col + '_' + p_or_x]
        return data
    res = {col: {} for col in cols}
    for col in cols:
        res[col][idx] = aux.ttest_rel(
            __get_data(df, col, stat_idxs[0], 'p'),
            __get_data(df, col, stat_idxs[1], 'x'))
    return aux.get_df(res, index_names)


def check_significance_gp(df, cols=cfg.FEATURES, stat_idxs=[None, None]):
    ''' computes significance per gender pair in select columns in given data

    see check_significance (+ df has to have gender pair in index level 2) 
    '''
    df_res = pd.DataFrame()
    for gp in ['F', 'M', 'X']:
        df_res = pd.concat([df_res, check_significance(
            df.xs(gp, level=2), gp, ['gender_pair'], cols, stat_idxs)], axis=0)
    return df_res


def compare_gps(df, cols=cfg.FEATURES, stat_idx=None, func=identity):
    ''' compares test statistics for groups of sessions based on gender pairs

    can handle float or tuple cell contents; for tuples, component idx is needed

    args:
        df: pandas dataframe with columns containing relevant test statistics,
            a session in each row, and gender pair identified in index level 2
        cols: names of columns whose contents are compared across gender pairs
        stat_idx: index of relevant component in each cell (for tuple contents)
        func: function to apply to tuple contents (used to apply aux.r2z)
    returns:
        pandas dataframe with results of independent t-test, three rows,
        one column per entry in cols
    '''
    res = {col: {} for col in cols}
    for col in cols:
        # get measures per gender pair
        f = _get_data(df.xs('F', level=2)[col], stat_idx, func)
        m = _get_data(df.xs('M', level=2)[col], stat_idx, func)
        x = _get_data(df.xs('X', level=2)[col], stat_idx, func)
        # pairwise compare gender pairs
        res[col]['F:M'] = aux.ttest_ind(f, m)
        res[col]['F:X'] = aux.ttest_ind(f, x)
        res[col]['M:X'] = aux.ttest_ind(m, x)
    return aux.get_df(res, 'comp')


def _compare_binary(df, df_bt, tsk_or_ses, cols, stat_idx, func, 
                    factor_col, lvl0, lvl1):
    ''' compares entr. measure regarding a binary factor w/ independent t-tests

    args:
        df: pandas dataframe as returned by ap/lex entrainment measure function
        df_bt: "big table" as returned by ap.load_data
        tsk_or_ses: whether to compare task or session measures ('tsk'/'ses')
        cols: names of columns whose contents are compared with regard to factor
        stat_idx: index of relevant component in each cell (for tuple contents)
        func: function to apply to tuple contents (used to apply aux.r2z)
        factor_col: column representing the factor for which to compare 
            (gender, gender_paired, speaker_role, familiarity)
        lvl0: reference level of the binary factor
        lvl1: comparison level of the binary factor
    returns:
        pandas dataframe with results of independent t-tests, single row,
        one column per entry in cols
    ''' 
    assert tsk_or_ses in ['tsk', 'ses'], 'unknown tsk_or_ses value'
    assert factor_col in ['gender', 'gender_paired', 'speaker_role', 
                          'familiarity'], 'unknown factor'
    # filter data as needed
    df = df.reset_index()
    if tsk_or_ses == 'tsk':
        df = df[df['tsk_id']!=0]
    else:
        df = df[df['tsk_id']==0]
    if factor_col in ['gender', 'gender_paired', 'speaker_role']:
        df = df[df['spk_id']!=0]
    else:
        df = df[df['spk_id']==0]
    # join relevant column for comparison
    on = ['spk_id'] if factor_col in ['gender', 'gender_paired'] \
        else ['tsk_id', 'spk_id'] if factor_col == 'speaker_role' \
        else [tsk_or_ses + '_id']
    fltr = df_bt['p_or_x'] != 'x'
    df = df.join(df_bt[fltr].loc[:,on+[factor_col]].groupby(on).first(), on=on)
    # perform comparisons
    res = {col: {} for col in cols}
    for col in cols:
        res[col][lvl0+':'+lvl1] = aux.ttest_ind(
            _get_data(df[df[factor_col]==lvl0][col], stat_idx, func),
            _get_data(df[df[factor_col]==lvl1][col], stat_idx, func))
    return aux.get_df(res, 'comp')


def compare_genders(df, df_bt, cols=cfg.FEATURES, stat_idx=None, func=identity):
    ''' convenience function to call _compare_binary for gender comparison '''
    return _compare_binary(
        df, df_bt, 'ses', cols, stat_idx, func, 'gender', 'f', 'm')


def compare_partner_genders(
        df, df_bt, cols=cfg.FEATURES, stat_idx=None, func=identity):
    ''' convenience func. to call _compare_binary for partner gender comp. '''
    return _compare_binary(
        df, df_bt, 'ses', cols, stat_idx, func, 'gender_paired', 'f', 'm')


def compare_roles(df, df_bt, cols=cfg.FEATURES, stat_idx=None, func=identity):
    ''' convenience function to call _compare_binary for role comparison '''
    return _compare_binary(
        df, df_bt, 'tsk', cols, stat_idx, func, 'speaker_role', 'd', 'f')


def compare_familiarities(
        df, df_bt, tsk_or_ses, cols=cfg.FEATURES, stat_idx=None, func=identity):
    ''' convenience function to call _compare_binary for famil. comparison '''
    return _compare_binary(
        df, df_bt, tsk_or_ses, cols, stat_idx, func, 'familiarity', 'hi', 'lo')


def check_chu_cnt_correlations(df, df_bt, tsk_or_ses, tx_only=False,
        cols=cfg.FEATURES, stat_idx=None, func=identity):
    ''' checks for correlations of measures with number of chunks '''
    assert tsk_or_ses in ['tsk', 'ses'], 'unknown tsk_or_ses value'
    df_tmp = df.join(
        _get_chu_cnt(df_bt, tsk_or_ses, tx_only), on=tsk_or_ses+'_id')
    for col in cols:
        data = _get_data(df_tmp[col], stat_idx)
        print(col, aux.pearsonr(data, df_tmp['chu_cnt']))
    




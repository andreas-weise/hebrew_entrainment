import itertools
import numpy as np
import pandas as pd

import ana
import aux
import cfg
import db
import fio

# this module implements the five acoustic-prosodic entrainment measures we use
# note: in the result dataframes, an index of 0 for tsk_id or spk_id or an index
#       -1 for ses_id (since ses_id 0 really exists) indicates "all"; 
#       e.g., tsk_id 0 means "for all tasks in this session"



################################################################################
#                                AUX FUNCTIONS                                 #
################################################################################
# auxiliary functions used only internally within this module

def _normalize_features(df, nrm_type):
    ''' normalizes features in given dataframe in specified way 

    args:
        df: pandas dataframe with a "*_raw" column per feature
        nrm_type: how to normalize features (see cfg.NRM_TYPES)
    returns:
        input dataframe, with new columns with normalized features,
        "*_raw" columns removed
    '''
    assert nrm_type in cfg.NRM_TYPES, 'unknown normalization type'
    if nrm_type == cfg.NRM_RAW:
        print('no normalization requested, only renaming "_raw" columns')
        df.rename(columns={f + '_raw': f for f in cfg.FEATURES}, inplace=True)
    else:    
        # determine mean and standard deviation per speaker or gender
        grp_col = 'spk_id' if nrm_type == cfg.NRM_SPK else 'gender'
        df_raw = df.loc[:, [grp_col] + ['%s_raw' % f for f in cfg.FEATURES]]
        df_means = df_raw.groupby([grp_col]).mean()
        df_means.rename(
            columns={c:c[:-4] + '_mean' for c in df_means.columns}, 
            inplace=True)
        df = df.join(df_means, on=grp_col)
        df_stds = df_raw.groupby([grp_col]).std()
        df_stds.rename(
            columns={c:c[:-4] + '_std' for c in df_stds.columns}, inplace=True)
        df = df.join(df_stds, on=grp_col)
        # z-score normalize based on means and standard deviations
        for f in cfg.FEATURES:
            df_zf = (df[f + '_raw'] - df[f + '_mean']) / df[f + '_std']
            df_zf.name = f
            df = df.join(df_zf)
    # remove columns with raw features and feature stats
    cols = [[f + '_raw', f + '_mean', f + '_std'] for f in cfg.FEATURES_ALL]
    cols = list(itertools.chain(*cols))
    df.drop(cols, axis=1, inplace=True, errors='ignore')
    return df


def _compute_sims(df, extra_cols=[]):
    ''' computes similarity betw. paired chunks/tasks/sessions for all features 

    args:
        df: pandas dataframe with pairs of feature values, either individual 
            values (for chunk pairs) or means (for speakers in tasks/sessions) 
    returns:
        input dataframe with new, added "*_sim" columns, one per feature
    '''
    for f in cfg.FEATURES:
        sims_f = -abs(df[f] - df[f + '_paired'])
        sims_f.name = f + '_sim'
        df = df.join(sims_f)
    return df


def _load_pairs(df, extra_cols=[]):
    ''' loads chunk pairs and joins with given data, features, and extra columns

    args:
        df: pandas dataframe with normalized features per chunk
        extra_paired_cols: extra columns, in addition to features, to include 
            regarding paired chunks
    returns:
        pandas dataframe with chunk pairs (with features) per row 
    '''
    # tmp1: pairs of chunk ids (adjacent and non-adjacent turn exchange chunks)
    tmp1 = db.pd_read_sql_query(
        'SELECT p_or_x, chu_id1, chu_id2, rid FROM chunk_pairs')
    # tmp2: all chu_ids with respective feature values and extra columns
    loc_cols = ['chu_id'] + cfg.FEATURES + extra_cols
    tmp2 = df.loc[:, loc_cols].set_index('chu_id')
    # tmp3: df joined with pairs -> df with second, turn-final, paired chu_id
    tmp3 = df.join(tmp1.set_index('chu_id2'), on='chu_id')
    # final step: full df data joined with features of paired, turn-final chu_id
    df = tmp3.join(tmp2, rsuffix='_paired', on='chu_id1')
    df.rename(columns={'chu_id1': 'chu_id_paired'}, inplace=True)
    # reset to running index
    df.reset_index(drop=True, inplace=True)
    # add columns for similarity between paired chunks for all features
    df = _compute_sims(df)
    return df


def _run_permutations(col1, col2, n=10):
    ''' computes correlations for random permutations of given columns 

    args:
        col1: data column 1
        col2: data column 2, has to have same length as col1
        n: number of permutations that are computed
    returns:
        mean correlation coefficient and number of p values below 0.05
    '''
    # need a copy because shuffle is inplace
    col1 = np.copy(col1)
    cnt = 0
    rs = []
    for i in range(n):
        np.random.shuffle(col1)
        r, p, _ = aux.pearsonr(col1, col2)
        if p < 0.05:
            cnt += 1
        rs += [r]
    return np.mean(rs), cnt



################################################################################
#                                MAIN FUNCTIONS                                #
################################################################################

def load_data(nrm_type, extra_paired_cols=[]):
    ''' loads data into one wide dataframe with redundant info 
    
    args: 
        nrm_type: how to normalize features (see cfg.NRM_TYPES)
        extra_paired_cols: extra columns to include regarding paired speakers
    returns:
        pandas dataframe with data per chunk (or chunk pair, where applicable),
        with running index (not chu_id because non-adjacent chunk pairs lead to 
        multiple rows per chunk)
    '''
    # load raw data ("big table" dataframe with redundant info)
    df_bt = db.pd_read_sql_query(sql_fname=cfg.SQL_BT_FNAME)
    # normalize features as needed
    df_bt = _normalize_features(df_bt, nrm_type)
    # add features of paired chunks (partner and non-partner) to each row and
    # compute similarity for each pair and all features
    df_bt = _load_pairs(df_bt, extra_paired_cols)
    return df_bt


def lsim(df_bt, grp_by=cfg.GRP_BYS):
    ''' computes local similarity for given data, per session, task, and speaker

    args:
        df_bt: "big table" pandas dataframe as returned by load_data
        grp_by: list of constants from cfg.GRP_BYS, for which groups of data
            the measure should be computed
    returns:
        pandas dataframe with results (t-statistic, p-value, degrees of 
        freedom, mean normalized sim), indexed by ses_id, tsk_id, and spk_id
        (0 for any index component means "all", e.g., all tasks in a session)
    '''
    def __lsim_test(df, f, key, level):
        ''' runs sanity check and the actual t-test for local similarity '''
        # exclude lists that are too short (pearsonr undefined)
        if len(df.xs(key, level=level)) < 2:
            return (np.nan, np.nan, len(df.xs(key, level=level))-1, np.nan)
        else:
            return aux.ttest_rel(df.xs(key, level=level)[f + '_sim_p'], 
                                 df.xs(key, level=level)[f + '_sim_x']) \
                + (df.xs(key, level=level)[f + '_lsim'].mean(), 
                   df.xs(key, level=level)[f + '_sim_p'].mean(), 
                   df.xs(key, level=level)[f + '_sim_x'].mean())
    cfg.check_grp_by(grp_by)
    # per turn-initial chunk and feature, compute mean similarity with
    # adjacent (mean of 1 val) and non-adjacent (mean of 10+ vals) paired chunks
    grp_cols = ['ses_type', 'ses_id', 'gender_pair', 'tsk_id', 'spk_id', 
                'chu_id', 'p_or_x']
    df_sims = df_bt.loc[:, grp_cols + ['%s_sim' % f for f in cfg.FEATURES]]
    df_sims = df_sims.groupby(grp_cols).mean()
    # self-join to get values for both adjacent and non-adjacent in each row
    df_sims = pd.DataFrame(df_sims.xs('p', level=6)).join( 
        df_sims.xs('x', level=6), lsuffix='_p', rsuffix='_x')
    # compute local similarity per feature, overall and per task, session, spk
    results = {f: {} for f in cfg.FEATURES}
    for f in cfg.FEATURES:
        lsims = -df_sims[f + '_sim_p'] / df_sims[f + '_sim_x']
        lsims.name = '%s_lsim' % f
        df_sims = df_sims.join(lsims)
        # exclude nan (NULL) feature values
        df_sims_f = df_sims[pd.notna(df_sims[f + '_sim_p'])]
        # per session, compute mean similarity at turn exchanges
        grp_cols = ['ses_type', 'ses_id', 'gender_pair']
        df_sims_f_agg = df_sims_f.reset_index(level=(0,1,2))
        df_sims_f_agg = df_sims_f_agg.groupby(grp_cols).mean()
        for ses_type in ['GAME',]:
            # result for entire session type
            if cfg.GRP_BY_SES_TYPE in grp_by:
                results[f][(ses_type, -1, 0, 0, 0)] = \
                    __lsim_test(df_sims_f_agg, f, ses_type, 0)
            # result per gender pair
            for gp in ['F', 'M', 'X']:
                results[f][(ses_type, -1, gp, 0, 0)] = \
                    __lsim_test(df_sims_f_agg, f, (ses_type, gp), (0, 2))                
            for (ses_id, gp) in \
            sorted(set([(v[0], v[1]) for v in df_sims_f.loc[ses_type].index])):
                # result per session, symmetric measure for both speakers
                if cfg.GRP_BY_SES in grp_by:
                    results[f][(ses_type, ses_id, gp, 0, 0)] = \
                        __lsim_test(df_sims_f, f, ses_id, 1)
                for spk_id \
                in set([v[3] for v in df_sims_f.xs(ses_id, level=1).index]):
                    # result per session, asymmetric measure per speaker
                    if cfg.GRP_BY_SES_SPK in grp_by:
                        results[f][(ses_type, ses_id, gp, 0, spk_id)] = \
                            __lsim_test(df_sims_f, f, (ses_id, spk_id), (1, 4))
                for tsk_id \
                in set([v[2] for v in df_sims_f.xs(ses_id, level=1).index]):
                    # result per task, symmetric measure for both speakers
                    if cfg.GRP_BY_TSK in grp_by:
                        results[f][(ses_type, ses_id, gp, tsk_id, 0)] = \
                            __lsim_test(df_sims_f, f, tsk_id, 3)
                    for spk_id \
                    in set([v[3] for v in df_sims_f.xs(tsk_id, level=3).index]):
                        # result per task, asymmetric measure per speaker
                        if cfg.GRP_BY_TSK_SPK in grp_by:
                            key = (ses_type, ses_id, gp, tsk_id, spk_id)
                            results[f][key] = \
                                __lsim_test(
                                    df_sims_f, f, (tsk_id, spk_id), (3, 4))
    return aux.get_df(
        results, ['ses_type', 'ses_id', 'gender_pair', 'tsk_id', 'spk_id'])


def syn(df_bt, grp_by=[g for g in cfg.GRP_BYS if g != cfg.GRP_BY_SES_TYPE]):
    ''' computes synchrony for given data, per session, task, and speaker

    args:
        df_bt: "big table" pandas dataframe as returned by load_data
        grp_by: list of constants from cfg.GRP_BYS, for which groups of data
            the measure should be computed
    returns:
        pandas dataframe with results (r-value, p-value, degrees of freedom, 
        mean correlation of random permutations, and count with p below 0.05), 
        indexed by ses_id, tsk_id, and spk_id
        (0 for any index component means "all", e.g., all tasks in a session)
    '''
    def __syn_test(df, f):
        ''' runs sanity checks and the actual pearsonr for synchrony '''
        # exclude lists that are too short or constant (pearsonr undefined)
        if len(df) < 3 or np.std(df[f]) == 0:
            return (np.nan, np.nan, len(df)-2, np.nan, np.nan)
        else:
            # compute correlation between turn-final and turn-initial chunks
            return aux.pearsonr(df[f], df[f + '_paired']) \
                + _run_permutations(df[f], df[f + '_paired'])
    supported = [g for g in cfg.GRP_BYS if g != cfg.GRP_BY_SES_TYPE]
    cfg.check_grp_by(grp_by, supported)
    # compute synchrony for each feature per task, session, and speaker
    results = {f: {} for f in cfg.FEATURES}
    grp_cols = ['ses_type', 'ses_id', 'gender_pair']
    for (ses_type, ses_id, gp), df_grp_ses \
    in df_bt[df_bt['p_or_x'] == 'p'].groupby(grp_cols):
        for f in cfg.FEATURES:
            # exclude nan (NULL) feature values
            df_grp_ses_f = df_grp_ses[pd.notna(df_grp_ses[f])]
            df_grp_ses_f = df_grp_ses_f[pd.notna(df_grp_ses_f[f + '_paired'])]
            # result per session, symmetric measure for both speakers
            if cfg.GRP_BY_SES in grp_by:
                results[f][(ses_type, ses_id, gp, 0, 0)] = \
                    __syn_test(df_grp_ses_f, f)
            for spk_id, df_grp_ses_spk in df_grp_ses_f.groupby('spk_id'):
                # result per session, asymmetric measure per speaker
                if cfg.GRP_BY_SES_SPK in grp_by:
                    results[f][(ses_type, ses_id, gp, 0, spk_id)] = \
                        __syn_test(df_grp_ses_spk, f)
            for tsk_id, df_grp_ses_tsk in df_grp_ses_f.groupby('tsk_id'):
                # result per task, symmetric measure for both speakers
                if cfg.GRP_BY_TSK in grp_by:
                    results[f][(ses_type, ses_id, gp, tsk_id, 0)] = \
                        __syn_test(df_grp_ses_tsk, f)
                for spk_id, df_grp_ses_tsk_spk \
                in df_grp_ses_tsk.groupby('spk_id'):
                    # result per task, asymmetric measure per speaker
                    if cfg.GRP_BY_TSK_SPK in grp_by:
                        results[f][(ses_type, ses_id, gp, tsk_id, spk_id)] = \
                            __syn_test(df_grp_ses_tsk_spk, f)
    return aux.get_df(
        results, ['ses_type', 'ses_id', 'gender_pair', 'tsk_id', 'spk_id'])


def lcon(df_bt, grp_by=[g for g in cfg.GRP_BYS if g != cfg.GRP_BY_SES_TYPE]):
    ''' computes local convergence for given data, per session and speaker

    args:
        df_bt: "big table" pandas dataframe as returned by load_data
        grp_by: list of constants from cfg.GRP_BYS, for which groups of data
            the measure should be computed
    returns:
        pandas dataframe with results (r-value, p-value, degrees of freedom, 
        mean correlation of random permutations, and count with p below 0.05),
        indexed by ses_id, and spk_id
        (0 for spk_id index means both speakers in that session)
    '''
    def __lcon_test(df, f):
        if len(df) < 3 or np.std(df[f + '_sim']) == 0:
            # exclude lists that are too short or constant (pearsonr undefined)
            return (np.nan, np.nan, len(df)-2, np.nan, np.nan)
        else:
            # compute correlation between similarity and turn-initial start time
            return aux.pearsonr(df[f + '_sim'], df['start_time']) \
                + _run_permutations(df[f + '_sim'], df['start_time'])
    supported = [g for g in cfg.GRP_BYS if g != cfg.GRP_BY_SES_TYPE]
    cfg.check_grp_by(grp_by, supported)
    # note: correlating with turn_index_ses makes very little difference
    results = {f: {} for f in cfg.FEATURES}
    grp_cols = ['ses_type', 'ses_id', 'gender_pair']
    for (ses_type, ses_id, gp), df_grp_ses \
    in df_bt[df_bt['p_or_x'] == 'p'].groupby(grp_cols):
        for f in cfg.FEATURES:
            # exclude nan (NULL) feature values
            df_grp_ses = df_grp_ses[pd.notna(df_grp_ses[f + '_sim'])]
            # result per session, symmetric measure for both speakers
            if cfg.GRP_BY_SES in grp_by:
                results[f][(ses_type, ses_id, gp, 0, 0)] = \
                    __lcon_test(df_grp_ses, f)
            for spk_id, df_grp_ses_spk in df_grp_ses.groupby('spk_id'):
                # result per session, asymmetric measure per speaker
                if cfg.GRP_BY_SES_SPK in grp_by:
                    results[f][(ses_type, ses_id, gp, 0, spk_id)] = \
                        __lcon_test(df_grp_ses_spk, f)
            for tsk_id, df_grp_ses_tsk in df_grp_ses.groupby('tsk_id'):
                # result per task, symmetric measure for both speakers
                if cfg.GRP_BY_TSK in grp_by:
                    results[f][(ses_type, ses_id, gp, tsk_id, 0)] = \
                        __lcon_test(df_grp_ses_tsk, f)
                for spk_id, df_grp_ses_tsk_spk \
                in df_grp_ses_tsk.groupby('spk_id'):
                    # result per task, asymmetric measure per speaker
                    if cfg.GRP_BY_TSK_SPK in grp_by:
                        results[f][(ses_type, ses_id, gp, tsk_id, spk_id)] = \
                            __lcon_test(df_grp_ses_tsk_spk, f)
    return aux.get_df(
        results, ['ses_type', 'ses_id', 'gender_pair', 'tsk_id', 'spk_id'])


def gcon(df_bt):
    ''' computes global convergence for given data (tasks only)

    args:
        df_bt: "big table" pandas dataframe as returned by load_data
    returns:
        pandas dataframe with results (t-statistic, p-value, degrees of 
        freedom), with a dummy index; second dataframe with raw first and
        second half distances between speakers
    '''
    df_sub = df_bt[df_bt['p_or_x'] != 'x'].copy()
    # set tsk_id to 0 if you want to compute convergence per session
    # (and change from 'tsk_half' to 'ses_half' below)
    # df_sub['tsk_id'] = 0
    tsk_or_ses = 'tsk'
    # get feature mean per speaker and half
    grp_cols = ['ses_type', 'ses_id', 'gender_pair', 'tsk_id', 'spk_id', 
                'partner_spk_id', 'tsk_half']
    df_grps = df_sub.loc[:, grp_cols + cfg.FEATURES].groupby(grp_cols).mean()
    # self-join to get means for both halves in each row
    df_grps = df_grps.xs(1, level=6).join(
        df_grps.xs(2, level=6), lsuffix='_1', rsuffix='_2')
    # self-join again to get means for both halves for both speakers in each row
    # (main speaker is labeled 'A', partner 'B', unrelated to normal A/B)
    on = ['ses_type', 'ses_id', 'gender_pair', 'tsk_id', 
          'partner_spk_id', 'spk_id']
    df_grps = df_grps.join(df_grps, on=on, lsuffix='A', rsuffix='B')
    for f in cfg.FEATURES:
        # compute distance between speaker means for each half
        for h in [1, 2]:
            dists = abs(df_grps['%s_%dA' % (f, h)] - df_grps['%s_%dB' % (f, h)])
            dists.name = '%s_dist%d' % (f, h)
            df_grps = df_grps.join(dists)
        # compute shift from 1st to 2nd half for each speaker
        for s in ['A', 'B']:
            shifts = abs(df_grps['%s_1%s' % (f, s)]-df_grps['%s_2%s' % (f, s)])
            shifts.name = '%s_shift%s' % (f, s)
            df_grps = df_grps.join(shifts)
        # compute fraction of contribution to total shift by the main speaker
        # (main speaker is always labeled 'A', see above)
        fracs = df_grps['%s_shiftA' % (f)] \
            / (df_grps['%s_shiftA' % f] + df_grps['%s_shiftB' % f])
        fracs.name = '%s_frac' % f
        df_grps = df_grps.join(fracs)
    results = {f: {} for f in cfg.FEATURES}
    # compute global conv. per session type, feature, and gender pair
    for f in cfg.FEATURES:
        for ses_type in ['GAME',]:
            # ignore redundant rows (distances are symmetric) 
            df_sub2 = df_grps.iloc[::2]
            df_sub2 = \
                df_sub2.xs(ses_type, drop_level=False) if ses_type else df_sub2
            # ignore rows with missing values
            # (should be missing for all features if missing for any)
            df_sub2 = df_sub2[pd.notna(df_sub2[f + '_dist1'])]
            df_sub2 = df_sub2[pd.notna(df_sub2[f + '_dist2'])]
            for gp in [0, 'F', 'M', 'X']:
                df_gp = df_sub2.xs(gp, level=2) if gp else df_sub2
                results[f][(ses_type, tsk_or_ses, gp)] = \
                    aux.ttest_rel(df_gp[f + '_dist1'], df_gp[f + '_dist2'])
            # compute convergence contribution per speaker and feature
            cons = df_grps['%s_frac' % f] \
                * (df_grps['%s_dist1' % f] - df_grps['%s_dist2' % f])
            cons.name = '%s_con' % f
            df_grps = df_grps.join(cons)
    loc_cols = list(itertools.chain(
        *[[f + '_dist1', f + '_dist2', f + '_con'] for f in cfg.FEATURES]))
    df_grps.set_index(df_grps.index.droplevel(5), inplace=True)
    df_results_raw = df_grps.loc[:, loc_cols]
    # add symmetric measure (equivalent to sum of both speakers' contributions)
    df_tmp = df_results_raw.iloc[::2].reset_index()
    for f in cfg.FEATURES:
        df_tmp[f + '_con'] = df_tmp[f + '_dist1'] - df_tmp[f + '_dist2']
    df_tmp['spk_id'] = 0
    df_tmp.set_index(
        ['ses_type', 'ses_id', 'gender_pair', 'tsk_id', 'spk_id'], inplace=True)
    df_results_raw = pd.concat([df_results_raw, df_tmp], axis=0)

    return aux.get_df(results, ['ses_type', 'tsk_or_ses', 'gender_pair']), \
        df_results_raw


def gsim(df_bt, df_spk_pairs_orig):
    ''' computes global similarity for given data (sessions & tasks)

    args:
        df_bt: "big table" pandas dataframe as returned by load_data
        df_spk_pairs_orig: speaker pairs dataframe based on cfg.SQL_SP_FNAME
    returns:
        pandas dataframe with results (t-statistic, p-value, degrees of 
        freedom) per feature; second df with raw means/sims per interaction 
    '''
    df_sub = df_bt[df_bt['p_or_x'] != 'x'].copy()

    results = {f: {} for f in cfg.FEATURES}
    df_results_raw = pd.DataFrame()
    for tsk_or_ses in ['tsk', 'ses']:
        if tsk_or_ses == 'ses':
            # do not differentiate between tasks for session measure
            # ('ses' run second, so df_sub can be overwritten)
            df_sub['tsk_id'] = 0
        # compute the means per session, task, and speaker for all features
        grp_cols = ['ses_type', 'ses_id', 'tsk_id', 'spk_id']
        df_means = df_sub.loc[:,grp_cols+cfg.FEATURES].groupby(grp_cols).mean()
        # get copy of df_spk_pairs for this iteration
        df_spk_pairs = df_spk_pairs_orig.copy()
        # filter out unnecessary rows from speaker pairs
        if tsk_or_ses == 'tsk':
            df_spk_pairs = df_spk_pairs[df_spk_pairs['tsk_id'] != 0]
        else:
            df_spk_pairs = df_spk_pairs[df_spk_pairs['tsk_id'] == 0]
        # join feature means
        df_spk_pairs = df_spk_pairs.join(df_means, on=grp_cols)
        on = ['ses_type', 'ses_id_paired', 'tsk_id_paired', 'spk_id_paired']
        df_spk_pairs = df_spk_pairs.join(
            df_means, on=on, lsuffix='', rsuffix='_paired')
        # compute similarity between speakers (for partners and non-partners)
        df_spk_pairs = _compute_sims(df_spk_pairs)
        # compute mean partner (mean of 1 val) and non-partner sims 
        # per session, task, *and* speaker
        grp_cols = [
            'ses_type', 'ses_id', 'gender_pair', 'p_or_x', 'tsk_id', 'spk_id']
        loc_cols = grp_cols + cfg.FEATURES + [f + '_sim' for f in cfg.FEATURES]
        df_tmp = df_spk_pairs.loc[:, loc_cols].groupby(grp_cols).mean()
        # compute mean partner (mean of 2 identical vals) and non-partner sims
        # per session and task, *without* speaker
        # (not used below but included in df_results_raw for later use)
        grp_cols = ['ses_type', 'ses_id', 'gender_pair', 'p_or_x', 'tsk_id']
        loc_cols = grp_cols + cfg.FEATURES + [f + '_sim' for f in cfg.FEATURES]
        df_spk_pairs = df_spk_pairs.loc[:, loc_cols].groupby(grp_cols).mean()
        df_spk_pairs['spk_id'] = 0
        df_spk_pairs.set_index(['spk_id'], append=True, inplace=True)
        # concatenate results for with and without speaker
        df_spk_pairs = pd.concat([df_tmp, df_spk_pairs], axis=0)
        # self-join to get values for partners and non-partners in each row
        df_spk_pairs = pd.DataFrame(df_spk_pairs.xs('p', level=3)).join( 
             df_spk_pairs.xs('x', level=3), lsuffix='_p', rsuffix='_x')
        # compute global similarity per feature & gender pair (overall sig.)
        df_spk_pairs['spk_id'] = [v[4] for v in df_spk_pairs.index]
        # ignore samples without speaker
        df_sub2 = df_spk_pairs.copy()
        df_sub2 = df_sub2[df_sub2['spk_id']!=0]
        for f in cfg.FEATURES: 
            # ignore samples with missing non-partner values
            # (should be missing for all features if missing for any)
            df_sub2 = df_sub2[pd.notna(df_sub2[f + '_sim_x'])]
            for ses_type in ['GAME',]:
                df_sub3 = df_sub2.xs(ses_type, drop_level=False) if ses_type \
                    else df_sub2
                for gp in [0, 'F', 'M', 'X']:
                    df_gp = df_sub3.xs(gp, level=2) if gp else df_sub3
                    results[f][(ses_type, tsk_or_ses, gp)] = \
                        aux.ttest_rel(df_gp[f + '_sim_p'], df_gp[f + '_sim_x'])
        df_results_raw = pd.concat([df_results_raw, df_spk_pairs], axis=0)
    # clean up columns in dataframe with intermediate columns
    df_results_raw.drop('spk_id', axis=1, inplace=True)
    df_results_raw.drop([f + '_x' for f in cfg.FEATURES], axis=1, inplace=True)
    df_results_raw.rename(
        columns={f + '_p': f for f in cfg.FEATURES}, inplace=True)
    # add column for normalized measure per feature
    for f in cfg.FEATURES: 
        df_results_raw[f + '_sim_nrm'] = \
            -df_results_raw[f + '_sim_p'] / df_results_raw[f + '_sim_x']
    return aux.get_df(results, ['ses_type', 'tsk_or_ses', 'gender_pair']), \
        df_results_raw





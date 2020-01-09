from constants import FEATURES, GENDER_PAIRS, GENDERS, ROLES
from measures import KLDSim, getTskSesIDs, getTskSesDur
import itertools
from numpy import mean
import pandas as pd
import random
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def bh(p_vals, alpha=0.05):
    ''' returns benjamini-hochberg threshold for significance at alpha '''
    p_vals = sorted(p_vals)
    k = max([k if p < (k+1) * alpha / len(p_vals) else -1
             for k, p in enumerate(p_vals)])
    return p_vals[k] if k != -1 else 0.0


def spToStr(s, p, p_sig, p_app, printS=False, dof=None):
    ''' generates output for test results and significance levels '''
    # s is a test statistic (e.g., t or r)
    
    if printS:
        res = ('+' + str(s) if s > 0 else str(s)) if p <= p_app else ' '
    else:
        res = ('+' if s > 0 else '-') if p <= p_app else ' '
    if p > p_sig and p <= p_app:
        res = '(' + res + ')'
    if dof is not None:
        res += ' ' + str(dof)
    return res


def ttest_ind(a, b):
    ''' stats.ttest_ind(a, b) with degrees of freedom also returned '''
    return stats.ttest_ind(a, b) + (len(a) + len(b) - 2,)


def ttest_rel(a, b):
    ''' stats.ttest_rel(a, b) with degrees of freedom also returned '''
    return stats.ttest_rel(a, b) + (len(a) - 1,)


def pearsonr(x, y):
    ''' stats.pearsonr(x, y) with degrees of freedom also returned '''
    return stats.pearsonr(x, y) + (len(x) - 2,)


def printRes(res, alpha=0.05):
    ''' generates standard output for series of tests that vary by feature '''
    # get threshold for significance
    p_sig = bh(list(zip(*res))[2], alpha)
    p_app = bh(list(zip(*res))[2], 2*alpha)
    # print results
    for f, s, p, dof in res:
        print(f, 
              p if p <= p_sig else '(' + str(p) + ')', 
              spToStr(s, p, p_sig, p_app, True, dof))
    print()


def testSigF(func, alpha=0.05, kwargs={}, fs=FEATURES):
    ''' tests significance of entrainment per feature and prints results
    args:
        func: entrainment measure (lSim, lCon, syn, gSim, gCon, distSim, KLDSim)
        alpha: significance level; set to 1.0 to see all "trends"
        kwargs: additional arguments for func (mostly for lexical measures)
        fs: list of features for which to run func (single entry for lexical)
    returns:
        pd.DataFrame with series of values of measure and baseline
    '''
    df = pd.DataFrame()
    # run t-tests for all features
    res = []
    for f in fs:
        data = func(f=f, **kwargs)
        res += [(f,) + ttest_rel(data['main'], data['baseline'])]
        df = df.append(data, ignore_index=True)
    printRes(res, alpha)
    return df


def testSigGP(func, alpha=0.05, kwargs={}, fs=FEATURES):
    ''' tests significance of entrainment per gender pair and feature
    args:
        see testSigF
    returns:
        pd.DataFrame with significant results per gender pair and feature
    '''
    df = pd.DataFrame()
    # get entrainment measures for all features and gender pairs
    for f in fs:
        for gp in GENDER_PAIRS:
            df = df.append(func(f=f, gp=gp, **kwargs), ignore_index=True)
    # compute tendency and significance level per feature and gender pair 
    tp = [[ttest_rel(df[(df['gp'] == gp)&(df['f'] == f)]['main'],
                     df[(df['gp'] == gp)&(df['f'] == f)]['baseline'])
           for gp in GENDER_PAIRS] 
          for f in fs]
    # compute significance threshold per gender pair
    p_sigs = [bh([v[i][1] for v in tp], alpha)
              for i in range(len(GENDER_PAIRS))]
    p_apps = [bh([v[i][1] for v in tp], 2*alpha)
              for i in range(len(GENDER_PAIRS))]
    # return significant results
    return pd.DataFrame([[spToStr(tp[i][j][0], tp[i][j][1], 
                                  p_sigs[j], p_apps[j])
                          for j in range(len(GENDER_PAIRS))] 
                         for i in range(len(fs))], 
                        fs, GENDER_PAIRS)


def testDurCorr(func, alpha=0.05, kwargs={}, fs=FEATURES):
    ''' test for correlations of entrainment with interaction duration 
    args:
        see testSigF
        kwargs: has to include 'tsk_or_ses'
    returns:
        pd.DataFrame with series of values of measure and baseline
    '''
    tsk_or_ses = kwargs['tsk_or_ses']
    tsk_ses_ids = getTskSesIDs(tsk_or_ses)
    df = pd.DataFrame()
    res = []
    for f in fs:
        data = func(f=f, **kwargs)
        x = [(mean(data[data['tsk_ses_id'] == tsk_ses_id]['main']),
              getTskSesDur(tsk_or_ses, tsk_ses_id))
             for tsk_ses_id in tsk_ses_ids]
        df = df.append(data, ignore_index=True)
        r, p, dof = pearsonr(*zip(*x))
        res += [(f, r, p, dof)]
    printRes(res, alpha)
    return df


def compareGP(func, alpha=0.05, kwargs={}, fs=FEATURES):
    ''' compares trends/strength of entrainment across gender pairs 
    args:
        see testSigF
    returns:
        pd.DataFrame with significant results per feature and comparison
    '''
    res = []
    for f in fs:
        # get measures per gender pair
        ff = func(f=f, gp='f', **kwargs)['main']
        mm = func(f=f, gp='m', **kwargs)['main']
        fm = func(f=f, gp='fm', **kwargs)['main']

        # pairwise compare gender pairs
        t1, p1, _ = ttest_ind(ff, mm)
        t2, p2, _ = ttest_ind(ff, fm)
        t3, p3, _ = ttest_ind(mm, fm)
        # compute significance level and output row for current feature
        p_sig = bh([p1, p2, p3], alpha)
        p_app = bh([p1, p2, p3], 2*alpha)
        res += [(spToStr(t1, p1, p_sig, p_app), 
                 spToStr(t2, p2, p_sig, p_app), 
                 spToStr(t3, p3, p_sig, p_app))]
    return pd.DataFrame(res, fs, ['ff vs. mm', 'ff vs. fm', 'mm vs. fm'])


def compareGP2(func, alpha=0.05, kwargs={}, fs=FEATURES):
    ''' compares trends/strength of entrainment across gender pairs 

    columns as "families" instead of rows, few results remain significant
    (use compareGP to get a better idea of trends in the data)

    args:
        see testSigF
    returns:
        pd.DataFrame with significant results per feature and comparison
    '''
    res = []
    for f in fs:
        # get measures per gender pair
        ff = func(f=f, gp='f', **kwargs)['main']
        mm = func(f=f, gp='m', **kwargs)['main']
        fm = func(f=f, gp='fm', **kwargs)['main']

        # pairwise compare gender pairs
        res += [[ttest_ind(ff, mm), ttest_ind(ff, fm), ttest_ind(mm, fm)]]
    # compute significance level per gender comparison
    p_sigs = [bh([row[i][1] for row in res], alpha) for i in range(3)]
    p_apps = [bh([row[i][1] for row in res], 2*alpha) for i in range(3)]

    out = [[spToStr(row[i][0], row[i][1], p_sigs[i], p_apps[i]) 
            for i in range(3)]
           for row in res]
    return pd.DataFrame(out, fs, ['ff vs. mm', 'ff vs. fm', 'mm vs. fm'])


def compareG_KLD(f, alpha=0.05, kwargs={}):
    ''' compares trends/strength of asymmetric KLDSim per gender
    args:
        see testSigF
    returns:
        pd.DataFrame with series of values of measure and baseline for
        asymmetric measure for male and female speakers separately
    '''
    df_m = KLDSim(f=f, g1='m', tgt=1, **kwargs).append(
               KLDSim(f=f, g2='m', tgt=2, **kwargs), ignore_index=True)
    df_f = KLDSim(f=f, g1='f', tgt=1, **kwargs).append(
               KLDSim(f=f, g2='f', tgt=2, **kwargs), ignore_index=True)
    res = [(f,) + ttest_ind(df_m['main'], df_f['main'])]
    printRes(res)
    return df_m.append(df_f, ignore_index=True)


def compareR_KLD(f, alpha=0.05, kwargs={}):
    ''' compares trends/strength of asymmetric KLDSim per role
    args:
        see testSigF
    returns:
        pd.DataFrame with series of values of measure and baseline for
        asymmetric measure for describer and follower separately
    '''
    df_d = KLDSim(f=f, g1='m', tgt=1, **kwargs).append(
               KLDSim(f=f, g1='f', tgt=1, **kwargs), ignore_index=True)
    df_f = KLDSim(f=f, g2='m', tgt=2, **kwargs).append(
               KLDSim(f=f, g2='f', tgt=2, **kwargs), ignore_index=True)
    res = [(f,) + ttest_ind(df_d['main'], df_f['main'])]
    printRes(res)
    return df_d.append(df_f, ignore_index=True)


def anova_twoway_w_rep(data):
    ''' runs two-way anova on given data

    column 0: dependent variable, column 1: 1st factor, column 2: 2nd factor
    same number of samples per condition needed

    code based on 
    https://www.marsja.se/three-ways-to-carry-out-2-way-anova-with-python/
    for example of data format, do
        data = pd.read_csv('../../data/misc/ToothGrowth.csv')
        data = data.iloc[:,1:4] 
    '''

    N = len(data.iloc[:,0])
    df_a = len(data.iloc[:,1].unique()) - 1
    df_b = len(data.iloc[:,2].unique()) - 1
    df_axb = df_a * df_b
    df_w = N - (df_a + 1) * (df_b + 1)
    
    grand_mean = data.iloc[:,0].mean()
    
    ss_a = sum([(data[data.iloc[:,1] == l].iloc[:,0].mean() - grand_mean)**2 
                for l in data.iloc[:,1]])
    ss_b = sum([(data[data.iloc[:,2] == l].iloc[:,0].mean() - grand_mean)**2 
                for l in data.iloc[:,2]])
    ss_t = sum((data.iloc[:,0] - grand_mean)**2)
    
    cols = [data[data.iloc[:,1] == v] for v in data.iloc[:,1].unique()]
    block_means = [[col[col.iloc[:,2] == v].iloc[:,0].mean() 
                    for v in col.iloc[:,2]] 
                   for col in cols]
    
    ss_w = sum([sum((cols[i].iloc[:,0] - block_means[i])**2)
                for i, col in enumerate(cols)])
    
    ss_axb = ss_t - ss_a - ss_b - ss_w
    
    ms_a = ss_a / df_a
    ms_b = ss_b / df_b
    ms_axb = ss_axb / df_axb
    ms_w = ss_w / df_w
    
    f_a = ms_a / ms_w
    f_b = ms_b / ms_w
    f_axb = ms_axb / ms_w
    
    p_a = stats.f.sf(f_a, df_a, df_w)
    p_b = stats.f.sf(f_b, df_b, df_w)
    p_axb = stats.f.sf(f_axb, df_axb, df_w)
    
    results = {'ss': [ss_a, ss_b, ss_axb, ss_w],
               'df': [df_a, df_b, df_axb, df_w],
               'F': [f_a, f_b, f_axb, 'NaN'],
               'PR(>F)': [p_a, p_b, p_axb, 'NaN']}
    columns=['ss', 'df', 'F', 'PR(>F)']
    
    index = [
        data.columns.values[1], 
        data.columns.values[2], 
        data.columns.values[1] + ':' + data.columns.values[2],
        'Residual'
    ]
    return pd.DataFrame(results, columns=columns, index=index)


def get_two_factor_kwargs(f, tsk_or_ses, g1, g2, role):
    ''' aux function to get kwargs to invoke measure for 2-factor analysis '''
    kwargs = {'f': f, 'tsk_or_ses': tsk_or_ses}
    if g1 is not None:
        kwargs['g1'] = g1
    if g2 is not None:
        kwargs['g2'] = g2
    if role is not None:
        kwargs['role'] = role
    return kwargs


def two_factor_analysis(func, tsk_or_ses='tsk', alpha=0.05, 
                        do_g1=False, do_g2=False, do_role=False, 
                        print_all=False, seed=0):
    ''' runs 2-factor analysis for given measure, interaction type, and factors

    uses asymmetric measures to attribute entrainment to specific speakers

    args:
        func: entrainment measure function (lSim, lCon, syn, gSim, gCon)
        tsk_or_ses: whether to analyze for tasks or sessions ('tsk'/'ses')
        alpha: significance level
        do_g1: whether to use g1 (see doc. for resp. func) as a factor
        do_g2: whether to use g2 (see doc. for resp. func) as a factor
        do_role: whether to use role (see doc. for resp. func) as a factor
            (for func in [lSim, lCon, syn] only)
        seed: initializer for random number generator for reproducibility
            set to None for random results (not reproducible!)
    returns:
        dict of pd.DataFrames containing the selected data samples per feature
    '''
    assert sum([do_g1, do_g2, do_role]) > 1, 'need at least two factors'
    
    # get get cartesian product of genders and roles (as needed)
    product = list(itertools.product(GENDERS if do_g1 else [None], 
                                     GENDERS if do_g2 else [None], 
                                     ROLES if do_role else [None]))
    data_all = {}
    results_all = pd.DataFrame()
    for f in FEATURES:
        random.seed(seed)
        data_one = pd.DataFrame()

        # determine minimum number of samples across all factor combinations 
        # for given function (=measure), feature, and interaction type
        sample_sizes = []
        for g1, g2, role in product:
            kwargs = get_two_factor_kwargs(f, tsk_or_ses, g1, g2, role)
            sample_sizes += [len(func(**kwargs))]
        sample_size = min(sample_sizes)
        print('%s (sample size: %d)' % (f, sample_size))
        # get samples for all combinations of factor values
        for g1, g2, role in product:
            df = func(**get_two_factor_kwargs(f, tsk_or_ses, g1, g2, role))
            df_new = pd.DataFrame()
            df_new['v'] = random.sample(list(df['main'].values), sample_size)
            if do_g1 and do_g2 and do_role:
                # doing genders and roles, merge genders
                df_new['g'] = g1 + g2
                df_new['r'] = role
            else:
                # doing only genders or role and one gender, no merge
                if do_g1:
                    df_new['g1'] = g1
                if do_g2:
                    df_new['g2'] = g2
                if do_role:
                    df_new['role'] = role
            data_one = data_one.append(df_new, ignore_index=True)
        results_one = anova_twoway_w_rep(data_one)
        results_one['f'] = f
        # store results for repeated measure analysis (store only two-factor)
        results_all = results_all.append(results_one[2:3])
        if print_all:
            print(results_one, end='\n\n\n')
        else:
            print(results_one[2:3], end='\n\n\n')
        data_all[f] = data_one
    p_sig = bh([float(x) for x in results_all.iloc[:,3]])
    print('significant results:')
    print(results_all[results_all['PR(>F)'] <= p_sig], end='\n\n\n')
    p_app = bh([float(x) for x in results_all.iloc[:,3]], 2*alpha)
    print('results approaching significance:')
    print(results_all[(results_all['PR(>F)'] > p_sig)
                     &(results_all['PR(>F)'] <= p_app)])
    print('\n\n\n')
    for f in results_all[results_all['PR(>F)'] <= p_app]['f']:
        tukey(data_all[f])
    if len(results_all[results_all['PR(>F)'] <= p_app]) > 0:
        print('(note: positive meandiff means group2 entrains more\n'
              '       strongly or more positively (check means))') 
    return data_all


def tukey(data):
    ''' runs post hoc tukey's hsd

    args:
        data: pd.DataFrame with 
            column 0: dependent variable,
            column 1: 1st factor, 
            column 2: 2nd factor
    '''
    vals = []
    lbls = []
    col1 = data.columns.values[1]
    col2 = data.columns.values[2]
    for a, b in itertools.product(data.iloc[:,1].unique(),
                                  data.iloc[:,2].unique()):
        tmp = list(data[(data[col1] == a)&(data[col2] == b)].iloc[:,0].values)
        vals += tmp
        lbls += [a + '_' + b for i in range(len(tmp))]
    print(pairwise_tukeyhsd(vals, lbls), end='\n\n\n')



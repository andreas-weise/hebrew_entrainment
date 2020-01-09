from collections import defaultdict
from constants import DB_FNAME, LEX_DNAME, FEATURES, MF_COUNT
from constants import GENDER_PAIRS, GENDERS, ROLES
from math import log
import nltk
import numpy as np
from numpy import mean
import pandas as pd
from random import sample, seed
from scipy.stats import pearsonr, ttest_ind, ttest_rel
import sqlite3


################################################################################
#                          GENERAL AUXILIARY FUNCTIONS                         #
################################################################################

def getTskIDs():
    ''' returns sorted list of integer task ids '''
    return [v[0] for v in 
            execAndFetch('SELECT tsk_id FROM tasks ORDER BY tsk_id;')]


def getSesIDs():
    ''' returns sorted list of integer session ids '''
    return [v[0] for v in 
            execAndFetch('SELECT ses_id FROM sessions ORDER BY ses_id;')]


def getTskSesIDs(tsk_or_ses):
    return getTskIDs() if tsk_or_ses == 'tsk' else getSesIDs()


def getTskDur(tsk_id):
    ''' returns duration of given task (last time stamp) '''
    sql_stmt = \
        'SELECT MAX(chu.end_time)\n' \
        'FROM   chunks chu\n' \
        'JOIN   turns tur\n' \
        'ON     chu.tur_id == tur.tur_id\n' \
        'WHERE  tur.tsk_id == ' + str(tsk_id)
    return execAndFetch(sql_stmt)[0][0]


def getSesDur(ses_id):
    ''' returns duration of given session (sum of tasks' durations) '''
    sql_stmt = \
        'SELECT tsk_id\n' \
        'FROM   tasks\n' \
        'WHERE  ses_id == ' + str(ses_id)
    return sum([getTskDur(v[0]) for v in execAndFetch(sql_stmt)])


def getTskSesDur(tsk_or_ses, tsk_ses_id):
    checkInput(None, tsk_or_ses, tsk_ses_id)
    return getTskDur(tsk_ses_id) if tsk_or_ses == 'tsk' \
      else getSesDur(tsk_ses_id)


def checkInput(f=None, tsk_or_ses=None, tsk_ses_id=None, g1=None, g2=None, 
               gp=None, role=None, spk_id=None, tname=None, leq_or_gt=None):
    ''' checks common inputs to the functions of this module

    catches some usage errors and prevents execution of unwanted sql;
    called by all lower-level functions that directly generate sql
    '''
    assert f in FEATURES + [None], 'unknown feature'
    assert tsk_or_ses in ['tsk', 'ses', None], 'unknown tsk_or_ses value'
    assert tsk_ses_id is None or isinstance(tsk_ses_id, int), \
        'tsk_ses_id must be an integer or None'
    assert g1 in GENDERS + [None], 'unknown gender1'
    assert g2 in GENDERS + [None], 'unknown gender2'
    assert gp in GENDER_PAIRS + [None], 'unknown gender pair'
    assert role in ROLES + [None], 'unknown role'
    assert spk_id is None or isinstance(spk_id, int), \
        'spk_id must be an integer or None'
    assert tname in ['chp', 'p', 'x', 'sub', None], 'unknown table'
    assert leq_or_gt in ['leq', 'gt', None], 'unknown leq_or_gt value'
    
    assert not(gp and (g1 or g2)), 'genders should be given as pair (for ' \
        'symmetric measure) or individually (asymmetric measure), not both'


def execAndFetch(sql_stmt):
    ''' executes a given sql statement and returns fetchall() result '''
    with sqlite3.connect(DB_FNAME) as conn:
        return conn.cursor().execute(sql_stmt).fetchall()


def genFilterPartGP(tname, gp, role=None):
    ''' generates WHERE clause part for gender pair 

    args:
        tname: name of the table to which the filter is applied
        gp: gender pair of speakers; male, female, or mixed ('m'/'f'/'fm')
        role: role ('d'escriber/'f'ollower) of the female speaker for mixed
            pairs; ignored for male and female pairs
    '''
    if gp == 'm':
        part = tname + '.gender1 == "m" AND ' + tname + '.gender2 == "m"'
    elif gp == 'f':
        part = tname + '.gender1 == "f" AND ' + tname + '.gender2 == "f"'
    else: # gp == 'fm'
        # note: role refers to role of female speaker in this case
        sql_role = ''
        sql_role2 = ''
        if role:
            sql_role = ' AND ' + tname + '.speaker_role == "' + role + '"'
            role2 = 'd' if role == 'f' else 'f'
            sql_role2 = ' AND ' + tname + '.speaker_role == "' + role2 + '"'
        part = \
            '((' + tname + '.gender1 == "m" AND ' + \
                   tname + '.gender2 == "f"' + sql_role + ')\n' + \
            '       OR(' + tname + '.gender1 == "f" AND ' + \
                           tname + '.gender2 == "m"' + sql_role2 + '))'
    return part


def genFilter(tname, tsk_or_ses=None, tsk_ses_id=None, g1=None, g2=None, 
              gp=None, role=None, spk_id=None, leq_or_gt=None, start=True):
    ''' generates sql for filtering a given table by criteria
    
    "start" parameter determines whether statement should begin a clause 
    (with "WHERE") or continue one (with "AND")
    '''
    checkInput(None, tsk_or_ses, tsk_ses_id, g1, g2, gp, role, spk_id, 
               tname, leq_or_gt=leq_or_gt)
    
    parts = []
    if g1:
        parts.append(tname + '.gender1 == "' + g1 + '"')
    if g2:
        parts.append(tname + '.gender2 == "' + g2 + '"')
    if gp:
        parts.append(genFilterPartGP(tname, gp, role))
    if role and not gp:
        parts.append(tname + '.speaker_role == "' + role + '"')
    if spk_id is not None:
        parts.append(tname + '.spk_id == ' + str(spk_id))
    if tsk_or_ses and tsk_ses_id is not None:
        parts.append(tname + '.' + tsk_or_ses + '_id == ' + str(tsk_ses_id))
    if leq_or_gt:
        parts.append(tname + '.leq_or_gt == "' + leq_or_gt + '"')
    if len(parts) > 0:
        sql_filter = 'WHERE  ' if start else 'AND    '
        sql_filter += '\nAND    '.join(parts)
    else:
        sql_filter = ''
    return sql_filter


def loadLex(fname):
    ''' loads words from given lexicon file '''
    with open(LEX_DNAME + fname) as file:
        return [line.replace('\n', '') for line in file.readlines()]


def loadTokens(sql_stmt, pos=[], neg=[], lem=lambda t: t):
    ''' loads list of tokens from given sql, with filters and lemmatization '''
    return [lem(t)
            for t in ' '.join([v[0] for v in execAndFetch(sql_stmt)]).split()
            if (len(pos) == 0 or t in pos) and t not in neg]


def loadSpkTokens(tsk_or_ses, tsk_ses_id, spk_id, 
                  pos=[], neg=[], lem=lambda t: t):
    ''' loads list of tokens for given interaction and speaker '''
    checkInput(None, tsk_or_ses, tsk_ses_id, spk_id=spk_id)
    sql_stmt = \
        'WITH sub AS\n' \
        '(\n' \
        ' SELECT chu.words words,\n' \
        '        CASE\n' \
        '            WHEN tsk.a_or_b == "A"\n' \
        '            AND tur.speaker_role == "d"\n' \
        '            THEN spk_a.spk_id\n' \
        '            WHEN tsk.a_or_b == "B"\n' \
        '            AND tur.speaker_role == "f"\n' \
        '            THEN spk_a.spk_id\n' \
        '            ELSE spk_b.spk_id\n' \
        '        END spk_id,\n' \
        '        ses.ses_id,\n' \
        '        tsk.tsk_id\n' \
        ' FROM   chunks chu\n' \
        ' JOIN   turns tur\n' \
        ' ON     chu.tur_id == tur.tur_id\n' \
        ' JOIN   tasks tsk\n' \
        ' ON     tur.tsk_id == tsk.tsk_id\n' \
        ' JOIN   sessions ses\n' \
        ' ON     tsk.ses_id == ses.ses_id\n' \
        ' JOIN   speakers spk_a\n' \
        ' ON     ses.spk_id_a == spk_a.spk_id\n' \
        ' JOIN   speakers spk_b\n' \
        ' ON     ses.spk_id_b == spk_b.spk_id\n' \
        ')\n' \
        'SELECT sub.words\n' \
        'FROM   sub\n' + \
        genFilter('sub', tsk_or_ses, tsk_ses_id, spk_id=spk_id)
    return loadTokens(sql_stmt, pos, neg, lem)


def loadSpatialLex():
    ''' loads lexicon of spatial words, returns lemmata, types, lem function '''
    lex = loadLex('spatial_words.txt')
    # using defaultdict so lem(t) in loadTokens works smoothly for any t
    lem = defaultdict(str)
    for line in lex:
        items = line.split('\t')
        if items[0] != '':
            # lines with items[0] mark the beginning of a new section; types in
            # this and subsequent lines are all grammatical forms of the lemma
            lemma = items[0]
        lem[items[2]] = lemma
    return list(set(lem.values())), list(lem.keys()), lambda t: lem[t]


def loadMFTypes(neg=[]):
    ''' loads MF_COUNT most frequent types overall, excluding given list '''
    tokens = loadTokens('SELECT words FROM chunks', neg=neg)
    return [v[0] for v in nltk.FreqDist(tokens).most_common(MF_COUNT)]


def getIDs(tsk_or_ses='ses', tsk_ses_id=None, g1=None, g2=None, gp=None):
    ''' finds interaction(s) and their speakers matching given criteria
    
    args:
        tsk_or_ses: whether to look for tasks or sessions
        tsk_ses_id: specific task or session id
        g1: gender of the describer (or speaker A for sessions)
        g2: gender of the follower (or speaker B for sessions)
        gp: gender pair ('m'/'f'/'fm'; roles do not matter)
    returns:
        list of tuples with:
            task or session id (based on tsk_or_ses parameter) 
            speaker id of the describer (or speaker A for sessions)
            speaker id of the follower (or speaker B for sessions)
    '''
    checkInput(None, tsk_or_ses, tsk_ses_id, g1, g2, gp)
    
    sql_stmt = \
        'WITH sub AS\n' \
        '(\n' \
        ' SELECT ses.ses_id,\n' \
        '        tsk.tsk_id,\n' \
        '        CASE\n' \
        '            WHEN tsk.a_or_b == "A"\n' \
        '            THEN spk_a.spk_id\n' \
        '            ELSE spk_b.spk_id\n' \
        '        END spk_id1,\n' \
        '        CASE\n' \
        '            WHEN tsk.a_or_b == "B"\n' \
        '            THEN spk_a.spk_id\n' \
        '            ELSE spk_b.spk_id\n' \
        '        END spk_id2,\n' \
        '        CASE\n' \
        '            WHEN tsk.a_or_b == "A"\n' \
        '            THEN spk_a.gender\n' \
        '            ELSE spk_b.gender\n' \
        '        END gender1,\n' \
        '        CASE\n' \
        '            WHEN tsk.a_or_b == "B"\n' \
        '            THEN spk_a.gender\n' \
        '            ELSE spk_b.gender\n' \
        '        END gender2,\n' \
        '        CASE\n' \
        '            WHEN tsk.a_or_b == "A"\n' \
        '            THEN 1\n' \
        '            ELSE 0\n' \
        '        END for_ses\n' \
        ' FROM   tasks tsk\n' \
        ' JOIN   sessions ses\n' \
        ' ON     tsk.ses_id == ses.ses_id\n' \
        ' JOIN   speakers spk_a\n' \
        ' ON     ses.spk_id_a == spk_a.spk_id\n' \
        ' JOIN   speakers spk_b\n' \
        ' ON     ses.spk_id_b == spk_b.spk_id\n' \
        ')\n' \
        'SELECT sub.' + tsk_or_ses + '_id,\n' \
        '       sub.spk_id1,\n' \
        '       sub.spk_id2\n' \
        'FROM   sub\n' + \
        ('WHERE  sub.for_ses == 1\n' if tsk_or_ses == 'ses' else '') + \
        genFilter('sub', tsk_or_ses, tsk_ses_id, g1, g2, gp, 
                  start=tsk_or_ses=='tsk')
    return execAndFetch(sql_stmt)


def getIDsX(tsk_or_ses, tsk_ses_id, spk_id, match_maps=False):
    ''' returns non-partner replacements for given speaker and interaction 

    non-partners have the same gender (and role, for tasks) as the given 
    speaker, the same gender as the given speaker's partner in the given 
    interaction, and have never been paired with the speaker's partner
    if necessary, non-partners can also be required to be talking about 
    the same map (for tasks only)

    args:
        tsk_or_ses: whether to look for tasks or sessions
        tsk_ses_id: specific taks or session for which to find non-partners
        spk_id: speaker for which to find non-partner replacements
        match_maps: whether non-partner interaction needs to be about same map
                    (for tasks only)
    returns:
        list of tuples with:
            task or session id (based on tsk_or_ses parameter) 
            speaker id of the non-partner in that task/session
    '''
    checkInput(None, tsk_or_ses, tsk_ses_id, spk_id=spk_id)
    assert not(tsk_or_ses == 'ses' and match_maps), \
        'requiring matching maps is not supported for sessions'
    
    # *_s and *_p in this query refer to speaker and partner, resp.;
    # subselect finds speaker gender and role as well as partner gender 
    # from either speaker's "perspective" in all tasks;
    # main select finds all interactions and speakers that match given 
    # speaker and interaction with regard to these gender/role criteria
    # and which were never paired with the given speaker's partner
    # (SELECT DISTINCT is necessary to remove duplicate session results)
    sql_stmt = \
        'WITH sub AS\n' \
        '(\n' \
        ' SELECT ses.ses_id ses_id,\n' \
        '        tsk.tsk_id tsk_id,\n' + \
        ('        tsk.map_index map_index,\n' if match_maps else '') + \
        '        ses.spk_id_a spk_id_s,\n' \
        '        ses.spk_id_b spk_id_p,\n' \
        '        spk_a.gender gender_s,\n' \
        '        spk_b.gender gender_p,\n' \
        '        CASE \n' \
        '            WHEN tsk.a_or_b == "A"\n' \
        '            THEN "d"\n' \
        '            ELSE "f"\n' \
        '        END role_s\n' \
        ' FROM   tasks tsk\n' \
        ' JOIN   sessions ses\n' \
        ' ON     tsk.ses_id == ses.ses_id\n' \
        ' JOIN   speakers spk_a\n' \
        ' ON     ses.spk_id_a == spk_a.spk_id\n' \
        ' JOIN   speakers spk_b\n' \
        ' ON     ses.spk_id_b == spk_b.spk_id\n' \
        '\n' \
        ' UNION\n' \
        '\n' \
        ' SELECT ses.ses_id ses_id,\n' \
        '        tsk.tsk_id tsk_id,\n' + \
        ('        tsk.map_index map_index,\n' if match_maps else '') + \
        '        ses.spk_id_b spk_id_s,\n' \
        '        ses.spk_id_a spk_id_p,\n' \
        '        spk_b.gender gender_s,\n' \
        '        spk_a.gender gender_p,\n' \
        '        CASE \n' \
        '            WHEN tsk.a_or_b == "B"\n' \
        '            THEN "d"\n' \
        '            ELSE "f"\n' \
        '        END role_s\n' \
        ' FROM   tasks tsk\n' \
        ' JOIN   sessions ses\n' \
        ' ON     tsk.ses_id == ses.ses_id\n' \
        ' JOIN   speakers spk_a\n' \
        ' ON     ses.spk_id_a == spk_a.spk_id\n' \
        ' JOIN   speakers spk_b\n' \
        ' ON     ses.spk_id_b == spk_b.spk_id\n' \
        ')\n' \
        'SELECT DISTINCT\n' \
        '       non.' + tsk_or_ses + '_id,\n' \
        '       non.spk_id_s\n' \
        'FROM   sub tgt\n' \
        'JOIN   sub non\n' \
        'ON     tgt.spk_id_s != non.spk_id_s\n' \
        'AND    tgt.gender_s == non.gender_s\n' \
        'AND    tgt.gender_p == non.gender_p\n' \
        'AND    tgt.role_s == non.role_s\n' + \
        ('AND    tgt.map_index == non.map_index\n' if match_maps else '') + \
        'WHERE  tgt.' + tsk_or_ses +'_id == ' + str(tsk_ses_id) + '\n' \
        'AND    tgt.spk_id_s == ' + str(spk_id) + '\n' \
        'AND    NOT EXISTS (\n' \
        '           SELECT 1\n' \
        '           FROM   sessions ses\n' \
        '           WHERE  tgt.spk_id_p IN (ses.spk_id_a, ses.spk_id_b)\n' \
        '           AND    non.spk_id_s IN (ses.spk_id_a, ses.spk_id_b));'
    
    return execAndFetch(sql_stmt)

################################################################################
#                               LOCAL MEASURES                                 #
################################################################################
# measures based on analysis of turn exchanges (turn-initial + turn-final IPUs)


### LOCAL SIMILARITY
# similarity per individual turn-exchange, no aggregation


def genLSimSubselect(f, p_or_x):
    ''' generates subselect sql for local sim. (adjacent or non-adjacent)
    
    args:
        f: feature for which local similarity is selected
        p_or_x: select partner ('p') or non-partner ('x') IPU
            select returns average distance with regard to given feature f 
            between each turn-final ipu and...
                if p_or_x == 'p': the next turn-initial ipu (average of 1 value)
                if p_or_x == 'x': 10 random other turn-initial ipus 
    '''
    checkInput(f)
    assert p_or_x in ['p', 'x', None], 'unknown p_or_x value'
    
    return \
        ' SELECT chp.ses_id,\n' \
        '        chp.tsk_id,\n' \
        '        chp.spk_id,\n' \
        '        chp.chu_id1,\n' \
        '        AVG(-ABS(chu1.' + f + '-chu2.' + f + ')) ' + f + ',\n' \
        '        gender1,\n' \
        '        gender2,\n' \
        '        speaker_role\n' \
        ' FROM   chunk_pairs chp\n' \
        ' JOIN   chunks chu1\n' \
        ' ON     chp.chu_id1 == chu1.chu_id\n' \
        ' JOIN   chunks chu2\n' \
        ' ON     chp.chu_id2 == chu2.chu_id\n' \
        ' WHERE  chp.p_or_x == "' + p_or_x + '"\n' \
        ' GROUP BY chp.chu_id1\n'


def lSim(f, tsk_or_ses='ses', tsk_ses_id=None, 
         g1=None, g2=None, gp=None, role=None, spk_id=None):
    ''' computes local sim. values for specified interaction(s)

    pairs of similarity values with regard to feature f for adjacent and 
    non-adjacent ipus from interactions that match given criteria 

    args:
        tsk_or_ses: whether to filter for tasks or sessions
        tsk_ses_id: specific task or session id
        g1: gender of the turn-final speaker ('m'/'f')
            can be used to analyze similarity asymmetrically
        g2: gender of the turn-initial speaker ('m'/'f')
            can be used to analyze similarity asymmetrically
        gp: gender pair of speakers; male, female, or mixed ('m'/'f'/'fm')
        role: role ('d'escriber/'f'ollower) of the turn-initial speaker 
            (if gp is None) or the female speaker (if gp='fm', else unused);
            can be used to analyze similarity asymmetrically
        spk_id: id of turn-initial speaker
    returns:
        pd.DataFrame with local similarities, annotated by given criteria
    '''
    checkInput(f, tsk_or_ses, tsk_ses_id, g1, g2, gp, role, spk_id)
    
    sql_stmt = \
        'WITH p AS (\n' + genLSimSubselect(f, 'p') + '),\n' \
        'x AS (\n' + genLSimSubselect(f, 'x') + ')\n' \
        'SELECT \'' + f + '\',\n' \
        '       p.ses_id,\n' \
        '       p.tsk_id,\n' \
        '       p.spk_id,\n' \
        '       p.' + f + ',\n' \
        '       x.' + f + '\n' \
        'FROM   p\n' \
        'JOIN   x\n' \
        'ON     p.chu_id1 == x.chu_id1\n' + \
        'WHERE  p.' + f + ' IS NOT NULL\n' \
        'AND    x.' + f + ' IS NOT NULL\n' + \
        genFilter('p', tsk_or_ses, tsk_ses_id, g1, g2, gp, role, spk_id, 
                  start=False)
    data = [(r[0], tsk_or_ses, int(r[1]) if tsk_or_ses == 'ses' else int(r[2]), 
             g1, g2, gp, role, int(r[3]), float(r[4]), float(r[5])) 
            for r in execAndFetch(sql_stmt)]
    columns = ['f', 'tsk_or_ses', 'tsk_ses_id', 'g1', 'g2', 'gp', 'role', 
               'spk_id', 'main', 'baseline']
    return pd.DataFrame(data, columns=columns)


### LOCAL CONVERGENCE + SYNCHRONY
# these measures come in a symmetric and an asymmetric variety:
#     symmetric: includes all ipu pairs from both speakers
#     asymmetric: includes only ipu pairs with one particular speaker responding
# which variety is computed depends on the filtering criteria (see below)


def getLConSynFeatures(f, tsk_or_ses=None, tsk_ses_id=None, pos_or_spk='pos',
                       spk_id=None):
    ''' returns values for feature f at turn exchanges (shape (2,n))
    
    only adjacent ipus at turn exchanges included (used for local con., syn.)

    args: 
        see lConSyn
        spk_id: id of responding speaker (turn-initial), for asymmetric measures
    '''
    checkInput(f, tsk_or_ses, tsk_ses_id, spk_id=spk_id)
    
    if pos_or_spk == 'pos':
        sql_fragment_select = \
            'SELECT chu1.' + f + ',\n' \
            '       chu2.' + f + '\n'
    else:
        sql_fragment_select = \
            'SELECT CASE\n' \
            '           WHEN chp.a_or_b == "A"\n' \
            '           THEN chu2.' + f + '\n' \
            '           ELSE chu1.' + f + '\n' \
            '       END,\n' \
            '       CASE\n' \
            '           WHEN chp.a_or_b == "A"\n' \
            '           THEN chu1.' + f + '\n' \
            '           ELSE chu2.' + f + '\n' \
            '       END\n'
    
    sql_stmt = \
        sql_fragment_select + \
        'FROM   chunk_pairs chp\n' \
        'JOIN   chunks chu1\n' \
        'ON     chp.chu_id1 == chu1.chu_id\n' \
        'JOIN   turns tur\n' \
        'ON     chu1.tur_id == tur.tur_id\n' \
        'JOIN   chunks chu2\n' \
        'ON     chp.chu_id2 == chu2.chu_id\n' \
        'WHERE  chp.p_or_x == "p"\n' \
        'AND    chu1.' + f + ' IS NOT NULL\n' \
        'AND    chu2.' + f + ' IS NOT NULL\n' + \
        genFilter('chp', tsk_or_ses, tsk_ses_id, spk_id=spk_id, start=False) + \
        '\n' \
        'ORDER BY chp.tsk_id, tur.turn_index, chu1.chunk_index'
    
    return np.array(execAndFetch(sql_stmt)).transpose()


def oneLConSyn(con_or_syn, f, tsk_or_ses, tsk_ses_id, pos_or_spk='pos', 
               spk_id=None):
    ''' returns local convergence / synchrony for one feature, interaction
    
    computes values for real interaction and average for 10 fake ones 

    args:
        see lConSyn
        spk_id: id of responding speaker (turn-initial), for asymmetric measures
    '''
    data = getLConSynFeatures(f, tsk_or_ses, tsk_ses_id, pos_or_spk, spk_id)
    syn = lambda data: pearsonr(data[0], data[1])
    lCon = lambda data: \
        pearsonr(np.negative(np.abs(np.subtract(data[0], data[1]))), 
                 [i+1 for i in range(len(data[0]))])
    func = lCon if con_or_syn == 'con' else syn
    r_real, p_real = func(data)
    # get values for 10 fake, i.e., shuffled, interactions
    r_fakes = []
    sig_fake = 0
    seed(0)
    for i in range(10):
        data_fake = np.array([sample(list(data[0]), len(data[0])),
                              sample(list(data[1]), len(data[1]))])
        r_fake, p_fake = func(data_fake)
        sig_fake += 1 if p_fake < 0.05 else 0
        r_fakes += [r_fake]
    fisher = lambda r: 0.5*(log(1+r)-log(1-r))
    return (fisher(r_real), fisher(mean(r_fakes)), p_real, sig_fake)


def lConSyn(con_or_syn, f, tsk_or_ses, tsk_ses_id=None, 
            g1=None, g2=None, gp=None, role=None, pos_or_spk='pos'):
    ''' computes local convergence / synchrony for specified interaction(s)
    
    determines interactions matching given gender and role specifications
    and returns convergence or synchrony computed for specified level
    of aggregation (tasks or sessions) 

    args:
        con_or_syn: whether to compute local con. or synchrony ('con'/'syn')
        f: feature for which to compute local convergence / synchrony
        tsk_or_ses: whether to look for and aggregate by task or session 
            ('tsk'/'ses')
        tsk_ses_id: specific task or session id
        g1: gender of the turn-final speaker ('m'/'f') 
            results in asymmetric measure if not None
        g2: gender of the turn-initial speaker ('m'/'f') 
            results in asymmetric measure if not None
        gp: gender pair of speakers; male, female, or mixed ('m'/'f'/'fm')
            results in symmetric measure if not None
        role: role ('d'escriber/'f'ollower) of the turn-initial speaker 
            (if gp is None) or the female speaker (if gp='fm', else unused);
            results in asymmetric measure if not None and gp is None
        pos_or_spk: group feature values (in getLConSynFeatures) by... 
            'pos': turn-final or turn-initial position (default)
            'spk': speaker "A" or "B"
            irrelevant for convergence, but not for symmetric synchrony 
    returns:
        pd.DataFrame with local con. / synchrony, annotated by given criteria
    '''
    checkInput(f, tsk_or_ses, tsk_ses_id, g1, g2, gp, role)
    assert con_or_syn in ['con', 'syn'], 'unknown con_or_syn value'
    assert pos_or_spk in ['pos', 'spk'], 'unknown pos_or_spk value'
    
    include_spk = g1 or g2 or (role and not gp)
    # sql to determine relevant tasks/sessions
    sql_stmt = \
        'SELECT DISTINCT chp.' + tsk_or_ses + '_id' + \
        (',\n       chp.spk_id\n' if include_spk else '\n') + \
        'FROM   chunk_pairs chp \n' + \
        genFilter('chp', g1=g1, g2=g2, gp=gp, role=role)
    
    # get convergence value for each relevant task/session
    data = []
    for row in execAndFetch(sql_stmt):
        tsk_ses_id = row[0]
        spk_id = row[1] if include_spk else None
        z_real, z_fake, p_real, sig_fake = oneLConSyn(
            con_or_syn, f, tsk_or_ses, tsk_ses_id, pos_or_spk, spk_id)
        data += [(f, tsk_or_ses, tsk_ses_id, g1, g2, gp, role, spk_id,
                  con_or_syn, z_real, z_fake, p_real, sig_fake)]
    columns = ['f', 'tsk_or_ses', 'tsk_ses_id', 'g1', 'g2', 'gp', 'role', 
               'spk_id', 'con_or_syn', 'main', 'baseline', 'p_real', 'sig_fake']
    return pd.DataFrame(data, columns=columns)


def lCon(f, tsk_or_ses='ses', tsk_ses_id=None, 
         g1=None, g2=None, gp=None, role=None):
    ''' wrapper for lConSyn to provide consistent interface (see doc. there) '''
    return lConSyn('con', f, tsk_or_ses, tsk_ses_id, g1, g2, gp, role)


def syn(f, tsk_or_ses='ses', tsk_ses_id=None, 
        g1=None, g2=None, gp=None, role=None, pos_or_spk='pos'):
    ''' wrapper for lConSyn to provide consistent interface (see doc. there) '''
    return lConSyn('syn', f, tsk_or_ses, tsk_ses_id, g1, g2, gp, role, 
                   pos_or_spk)


################################################################################
#                                GLOBAL MEASURES                               #
################################################################################

### AUXILIARY FUNCTIONS

def getGlobalAvg(f, tsk_or_ses, tsk_ses_id, spk_id, leq_or_gt=None):
    ''' returns average value for feature, (half of) task, and speaker
    
    aggregation by whole session is not recommended because features are 
    z-score normalized by speaker and session so result would always be ~0
    
    args:
        f: feature whose average value should be returned
        tsk_or_ses: whether to aggregate by task or session ('tsk'/'ses')
        tsk_ses_id: task/session for which to aggregate
        spk_id: speaker for which to aggregate
        leq_or_gt: limit to ipus up to / from halfway point ('leq'/'gt')
    returns:
        single numeric value, feature average
    '''
    checkInput(f, tsk_or_ses, tsk_ses_id, spk_id=spk_id, leq_or_gt=leq_or_gt)
    if tsk_or_ses == 'ses' and not leq_or_gt:
        print('warning! average for whole session is meaningless with '
              'z-score normalization by speaker and session!')
    
    sql_stmt = \
        'WITH sub AS\n' \
        '(\n' \
        ' SELECT chu.' + f + ',\n' \
        '        CASE\n' \
        '            WHEN tsk.a_or_b == "A"\n' \
        '            AND tur.speaker_role == "d"\n' \
        '            THEN spk_a.spk_id\n' \
        '            WHEN tsk.a_or_b == "B"\n' \
        '            AND tur.speaker_role == "f"\n' \
        '            THEN spk_a.spk_id\n' \
        '            ELSE spk_b.spk_id\n' \
        '        END spk_id,\n' \
        '        ses.ses_id,\n' \
        '        tsk.tsk_id,\n' \
        '        tsk.task_index,\n' \
        '        tur.turn_index,\n' \
        '        chu.chunk_index,\n' \
        '        CASE\n' \
        '            WHEN tsk.task_index > hwp.task_index\n' \
        '            THEN "gt"\n' \
        '            WHEN tsk.task_index == hwp.task_index\n' \
        '            AND  tur.turn_index > hwp.turn_index\n' \
        '            THEN "gt"\n' \
        '            WHEN tsk.task_index == hwp.task_index\n' \
        '            AND  tur.turn_index == hwp.turn_index\n' \
        '            AND  chu.chunk_index > hwp.chunk_index\n' \
        '            THEN "gt"\n' \
        '            ELSE "leq"\n' \
        '        END leq_or_gt' \
        ' FROM   chunks chu\n' \
        ' JOIN   turns tur\n' \
        ' ON     chu.tur_id == tur.tur_id\n' \
        ' JOIN   tasks tsk\n' \
        ' ON     tur.tsk_id == tsk.tsk_id\n' \
        ' JOIN   sessions ses\n' \
        ' ON     tsk.ses_id == ses.ses_id\n' \
        ' JOIN   speakers spk_a\n' \
        ' ON     ses.spk_id_a == spk_a.spk_id\n' \
        ' JOIN   speakers spk_b\n' \
        ' ON     ses.spk_id_b == spk_b.spk_id\n' \
        ' JOIN   halfway_points hwp\n' \
        ' ON     ' + tsk_or_ses + '.' + tsk_or_ses + '_id == hwp.' + \
                     tsk_or_ses + '_id\n' \
        ')\n' \
        'SELECT AVG(sub.' + f + ')\n' \
        'FROM   sub\n' + \
        genFilter('sub', tsk_or_ses, tsk_ses_id, 
                  spk_id=spk_id, leq_or_gt=leq_or_gt)
    
    return execAndFetch(sql_stmt)[0][0]


### GLOBAL SIMILARITY

def gSim(f, tsk_or_ses='ses', tsk_ses_id=None, g1=None, g2=None, gp=None):
    ''' computes global similarity for specified interaction(s)
    
    pairs of partner and non-partner similarity values with regard to 
    feature f for interactions that match given criteria 
    
    args:
        f: feature for which to compute global similarities
        tsk_or_ses: whether to aggregate by task or session ('tsk'/'ses')
        tsk_ses_id: specific task or session id
        g1: gender of the describer (for tasks only; 'm'/'f')
        g2: gender of the follower (for tasks only; 'm'/'f')
        gp: gender pair ('m'/'f'/'fm'; roles do not matter)
    returns:
        pd.DataFrame with global similarities, annotated by given criteria
    '''
    data = []
    ids = getIDs(tsk_or_ses, tsk_ses_id, g1, g2, gp)
    for tsk_ses_id, spk_id1, spk_id2 in ids:
        # speaker averages and partner similarity
        avg1 = getGlobalAvg(f, tsk_or_ses, tsk_ses_id, spk_id1)
        avg2 = getGlobalAvg(f, tsk_or_ses, tsk_ses_id, spk_id2)
        entP = -abs(avg1-avg2)
        # non-partner similarities, replace speaker 1 with similar non-partners
        X1 = []
        for tsk_ses_id_x, spk_id_x in getIDsX(tsk_or_ses, tsk_ses_id, spk_id1):
            avgX = getGlobalAvg(f, tsk_or_ses, tsk_ses_id_x, spk_id_x)
            X1 += [-abs(avg2-avgX)]
        # non-partner similarities, replace speaker 2 with similar non-partners
        X2 = []
        for tsk_ses_id_x, spk_id_x in getIDsX(tsk_or_ses, tsk_ses_id, spk_id2):
            avgX = getGlobalAvg(f, tsk_or_ses, tsk_ses_id_x, spk_id_x)
            X2 += [-abs(avg1-avgX)]
        data += [[f, tsk_or_ses, tsk_ses_id, g1, g2, gp, 
                  entP, mean(X1+X2)]]
    columns = ['f', 'tsk_or_ses', 'tsk_ses_id', 'g1', 'g2', 'gp', 
               'main', 'baseline']
    return pd.DataFrame(data, columns=columns)


### GLOBAL CONVERGENCE

def gCon(f, tsk_or_ses='ses', tsk_ses_id=None, g1=None, g2=None, gp=None,
         dod=False):
    ''' computes global convergence for specified interaction(s)
    
    pairs of differences in speaker averages with regard to feature f for 
    first and second halves of interactions that match given criteria, as well
    as the difference between those differences
    
    args:
        f: feature for which to compute global convergence
        tsk_or_ses: whether to aggregate by task or session ('tsk'/'ses')
        tsk_ses_id: specific task or session id
        g1: gender of the describer (for tasks only; 'm'/'f')
        g2: gender of the follower (for tasks only; 'm'/'f')
        gp: gender pair ('m'/'f'/'fm'; roles do not matter)
        dod: whether to return diff of diffs (True) or diffs themselves (False);
            column names for diffs: 'main' (1st half) and 'baseline' (2nd half)
            (not quite appropriate here but chosen for consistency)
    returns:
        pd.DataFrame with global convergence values, annotated by given criteria
    '''
    data = []
    ids = getIDs(tsk_or_ses, tsk_ses_id, g1, g2, gp)
    for tsk_ses_id, spk_id1, spk_id2 in ids:
        # averages per speaker (1/2) and half of interaction (a/b)
        avg1a = getGlobalAvg(f, tsk_or_ses, tsk_ses_id, spk_id1, 'leq')
        avg1b = getGlobalAvg(f, tsk_or_ses, tsk_ses_id, spk_id1, 'gt')
        avg2a = getGlobalAvg(f, tsk_or_ses, tsk_ses_id, spk_id2, 'leq')
        avg2b = getGlobalAvg(f, tsk_or_ses, tsk_ses_id, spk_id2, 'gt')
        # store criteria and diff of diffs or diffs themselves
        row = [f, tsk_or_ses, tsk_ses_id, g1, g2, gp]
        if dod:
            row += [abs(avg1a - avg2a) - abs(avg1b - avg2b)] 
        else:
            row += [abs(avg1a - avg2a), abs(avg1b - avg2b)]
        data += [row]
    columns = ['f', 'tsk_or_ses', 'tsk_ses_id', 'g1', 'g2', 'gp', 
               'main'] + (['baseline'] if not dod else [])
    return pd.DataFrame(data, columns=columns)


################################################################################
#                               LEXICAL MEASURES                               #
################################################################################

def memDist(f):
    ''' memoization function for getDist '''
    memo = {}
    def helper(tsk_or_ses, tsk_ses_id, spk_id, 
               typesID=None, posID=None, negID=None, lemID=None, 
               include_zero=True, types=[], pos=[], neg=[], lem=lambda t: t):
        params = (tsk_or_ses, tsk_ses_id, spk_id, 
                  typesID, posID, negID, lemID, include_zero)
        if params not in memo:
            memo[params] = f(*params, types, pos, neg, lem)
        return memo[params]
    return helper


@memDist
def getDist(tsk_or_ses, tsk_ses_id, spk_id,
            typesID=None, posID=None, negID=None, lemID=None, 
            include_zero=True, types=[], pos=[], neg=[], lem=lambda t: t):
    ''' returns distribution of types for given interaction and speaker

    determines how often each type from optional list was spoken in given
    interaction by given speaker and divides it by total number of tokens
    from that speaker in that interaction (with filters and lemmatization)
    '''
    tokens = loadSpkTokens(tsk_or_ses, tsk_ses_id, spk_id, pos, neg, lem)
    types = types if len(types) > 0 else list(set(tokens))
    freqs = dict(nltk.FreqDist(tokens))
    if include_zero:
        res = {t: (freqs[t] / len(tokens) if t in freqs else 0) for t in types}
    else:
        res = {t: freqs[t] / len(tokens) for t in types if t in freqs}
    return res


def getEntropy(dist):
    ''' computes entropy for a given distribution '''
    return -sum([v*log(v) for v in dist.values() if v > 0])


def getEntropyWeights(distP, distXs):
    ''' weights given dists by entropy similarity with given reference dist '''
    entP = getEntropy(distP)
    entXs = [getEntropy(distX) for distX in distXs]
    diffs = [abs(entP - entX) for entX in entXs]
    minDiff = min(diffs)
    weights_raw = [diff / min(diffs) if minDiff != 0 
                   else (1 if diff == 0 else 0)
                   for diff in diffs]
    return [w / sum(weights_raw) for w in weights_raw]
    

def getTypes(types, dist1, dist2):
    return types if len(types) > 0 \
           else list(set(dist1.keys()).union(set(dist2.keys())))


def distSim(f, tsk_or_ses='ses', tsk_ses_id=None, g1=None, g2=None, gp=None, 
            typesID=None, posID=None, negID=None, lemID=None,
            types=[], pos=[], neg=[], lem=lambda t: t):
    ''' 
    
    args:
        f: unused, only for consistency with acoustic measure interfaces
        tsk_or_ses: whether to aggregate by task or session ('tsk'/'ses')
        tsk_ses_id: specific task or session id
        g1: gender of the describer (for tasks only; 'm'/'f')
        g2: gender of the follower (for tasks only; 'm'/'f')
        gp: gender pair ('m'/'f'/'fm'; roles do not matter)
        types: types for which to compare distributions (all if empty)
        pos: whitelist of types to use to determine total (all if empty)
        neg: blacklist of types to exclude from total
        lem: lemmatization function to apply to speakers' tokens
    returns:
        pd.DataFrame with distribution similarities, annotated by given criteria
    '''
    data = []
    params = (typesID, posID, negID, lemID, True, types, pos, neg, lem)
    ids = getIDs(tsk_or_ses, tsk_ses_id, g1, g2, gp)
    for tsk_ses_id, spk_id1, spk_id2 in ids:
        # type distribution per speaker 
        dist1 = getDist(tsk_or_ses, tsk_ses_id, spk_id1, *params)
        dist2 = getDist(tsk_or_ses, tsk_ses_id, spk_id2, *params)
        # partner similarity
        entP = sum([-abs((dist1[t] if t in dist1 else 0.0) 
                        -(dist2[t] if t in dist2 else 0.0)) 
                    for t in getTypes(types, dist1, dist2)])
        # non-partner similarities, replace speaker 1 with similar non-partners
        X1 = []
        idsX1 = getIDsX(tsk_or_ses, tsk_ses_id, spk_id1, tsk_or_ses=='tsk')
        distXs1 = []
        for tsk_ses_id_x, spk_id_x in idsX1:
            distXs1 += [getDist(tsk_or_ses, tsk_ses_id_x, spk_id_x, *params)]
            X1 += [sum([-abs((dist2[t] if t in dist2 else 0.0)
                            -(distXs1[-1][t] if t in distXs1[-1] else 0.0)) 
                        for t in getTypes(types, dist2, distXs1[-1])])]
        weights = getEntropyWeights(dist2, distXs1)
        entX1 = sum([w*X1[i] for i, w in enumerate(weights)])
        # non-partner similarities, replace speaker 2 with similar non-partners
        X2 = []
        idsX2 = getIDsX(tsk_or_ses, tsk_ses_id, spk_id2, tsk_or_ses=='tsk')
        distXs2 = []
        for tsk_ses_id_x, spk_id_x in idsX2:
            distXs2 += [getDist(tsk_or_ses, tsk_ses_id_x, spk_id_x, *params)]
            X2 += [sum([-abs((dist1[t] if t in dist1 else 0.0)
                            -(distXs2[-1][t] if t in distXs2[-1] else 0.0))
                        for t in getTypes(types, dist1, distXs2[-1])])]
        weights = getEntropyWeights(dist1, distXs2)
        entX2 = sum([w*X2[i] for i, w in enumerate(weights)])
        
        data += [[f, tsk_or_ses, tsk_ses_id, g1, g2, gp, 
                  entP, mean([entX1, entX2])]]
    columns = ['f', 'tsk_or_ses', 'tsk_ses_id', 'g1', 'g2', 'gp', 
               'main', 'baseline']
    return pd.DataFrame(data, columns=columns)


def computeKLD(dist1, dist2):
    ''' 
    
    '''
    # P: type probabilities for both speakers
    P = [dist1.copy(), dist2.copy()]
    # store whether an empty dist2 was given
    empty2 = len(list(P[1].keys())) == 0
    # V: overall vocabulary
    V = set(P[0].keys()).union(set(P[1].keys()))
    # epsilon: prob. for unseen types (lowest prob in either list / 10)
    # (paper does not specify denominator 10, just says epsilon has to 
    #  be "smaller" than the minima and empirically determined)
    if not empty2:
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
        kld += (P[0][t]) * log(P[0][t] / P[1][t])
    
    # normalize by kld from empty document (defaultdict is 0.0 for all keys)
    # (sign inversion follows Gravano et al., 2014; 
    #      kld >= 0 and lower kld = more entrainment 
    #      with sign inversion: greater value = more entrainment)
    if not empty2:
        kld = -kld / computeKLD(dist1, defaultdict(float))
    return -kld


def KLDSim(f, tsk_or_ses='ses', tsk_ses_id=None, g1=None, g2=None, gp=None, 
           typesID=None, posID=None, negID=None, lemID=None,
           types=[], pos=[], neg=[], lem=lambda t: t, tgt=None):
    ''' 
    
    lem should map from pos to types only, to ensure fractions add up to 1
    '''
    assert tgt in [1, 2, None], 'unknown target'

    data = []
    params = (typesID, posID, negID, lemID, False, types, pos, neg, lem)
    ids = getIDs(tsk_or_ses, tsk_ses_id, g1, g2, gp)
    for tsk_ses_id, spk_id1, spk_id2 in ids:
        # type distribution per speaker 
        dist1 = getDist(tsk_or_ses, tsk_ses_id, spk_id1, *params)
        dist2 = getDist(tsk_or_ses, tsk_ses_id, spk_id2, *params)

        if tgt in [1, None]:
            # KLD from speaker 1 (A or describer) to partner, speaker 2
            kld1 = computeKLD(dist2, dist1)
            # KLD from speaker 1 (A or describer) to non-partner
            X1 = []
            idsX1 = getIDsX(tsk_or_ses, tsk_ses_id, spk_id2, tsk_or_ses=='tsk')
            distXs1 = []
            for tsk_ses_id_x, spk_id_x in idsX1:
                distXs1 += [getDist(tsk_or_ses, tsk_ses_id_x, spk_id_x, *params)]
                X1 += [computeKLD(distXs1[-1], dist1)]
            weights = getEntropyWeights(dist2, distXs1)
            entX1 = sum([w*X1[i] for i, w in enumerate(weights)])
        if tgt in [2, None]:
            # KLD from speaker 2 (B or follower) to partner, speaker 1
            kld2 = computeKLD(dist1, dist2)
            # KLD from speaker 2 (B or follower) to non-partner
            X2 = []
            idsX2 = getIDsX(tsk_or_ses, tsk_ses_id, spk_id1, tsk_or_ses=='tsk')
            distXs2 = []
            for tsk_ses_id_x, spk_id_x in idsX2:
                distXs2 += [getDist(tsk_or_ses, tsk_ses_id_x, spk_id_x, *params)]
                X2 += [computeKLD(distXs2[-1], dist2)]
            weights = getEntropyWeights(dist1, distXs2)
            entX2 = sum([w*X2[i] for i, w in enumerate(weights)])
        if tgt is None:
            # symmetric measure
            entP = kld1 + kld2
            entX = entX1 + entX2
        else:
            entP = kld1 if tgt == 1 else kld2
            entX = entX1 if tgt == 1 else entX2
        data += [[f, tsk_or_ses, tsk_ses_id, g1, g2, gp, entP, entX]]
    columns = ['f', 'tsk_or_ses', 'tsk_ses_id', 'g1', 'g2', 'gp', 
               'main', 'baseline']
    return pd.DataFrame(data, columns=columns)










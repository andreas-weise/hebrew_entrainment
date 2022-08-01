import pandas as pd
import sqlite3

import cfg
import fio



################################################################################
#                            CONNECTION MAINTENANCE                            #
################################################################################

class DatabaseConnection(object):
    def __init__(self, db_fname):
        self._conn = sqlite3.connect(db_fname)
        self._c = self._conn.cursor()

    def __del__(self):
        self._conn.close()

    def execute(self, sql_stmt, params=tuple()):
        return self._c.execute(sql_stmt, params)

    def executemany(self, sql_stmt, params=tuple()):
        return self._c.executemany(sql_stmt, params)

    def executescript(self, sql_script):
        return self._c.executescript(sql_script)

    def getrowcount(self):
        return self._c.rowcount

    def commit(self):
        self._conn.commit()

    def get_conn(self):
        return self._conn

# global connection object; status maintained through functions below
# all functions interacting with database (setters etc.) assume open connection
dbc = None


def connect(corpus_id):
    ''' instantiates global connection object for given corpus '''
    global dbc
    dbc = DatabaseConnection(cfg.get_db_fname(corpus_id))


def close():
    ''' closes connection by deleting global connection object '''
    global dbc
    del dbc
    dbc = None


def commit():
    ''' issues commit to database via global connection object '''
    dbc.commit()


def get_conn():
    ''' returns internal sqlite3 connection of global connection object 

    this should rarely be necessary, only if conn needs to be passed on '''
    return dbc.get_conn()



################################################################################
#                                  INSERTIONS                                  #
################################################################################

def ins_spk(spk_id, gender, age, born_in, native_lang, years_edu):
    ''' inserts individual speaker in speakers table '''
    sql_stmt = \
        'INSERT INTO speakers ' \
            '(spk_id, gender, age, born_in, native_lang, years_edu)\n' \
        'VALUES (?,?,?,?,?,?);'
    params = (spk_id, gender, age, born_in, native_lang, years_edu)
    dbc.execute(sql_stmt, params)


def ins_ses(ses_id, spk_id_a, spk_id_b, fam):
    ''' inserts individual session in sessions table '''
    sql_stmt = \
        'INSERT INTO sessions (ses_id, spk_id_a, spk_id_b, familiarity)\n' \
        'VALUES (?,?,?,?);'
    dbc.execute(sql_stmt, (ses_id, spk_id_a, spk_id_b, fam))


def ins_tsk(tsk_id, ses_id, map_index, task_index, a_or_b):
    ''' inserts individual task in tasks table '''
    sql_stmt = \
        'INSERT INTO tasks (tsk_id, ses_id, map_index, task_index, a_or_b)\n' \
        'VALUES (?,?,?,?,?);'
    dbc.execute(sql_stmt, (tsk_id, ses_id, map_index, task_index, a_or_b))


def ins_tur(tur_id, tsk_id, turn_index, speaker_role):
    ''' inserts individual turn in turns table '''
    sql_stmt = \
        'INSERT INTO turns (tur_id, tsk_id, turn_index, speaker_role)\n' \
        'VALUES (?,?,?,?);'
    dbc.execute(
        sql_stmt, (tur_id, tsk_id, turn_index, speaker_role))


def ins_chu(chu_id, tur_id, chunk_index, start_time, end_time, duration, words):
    ''' inserts individual chunk in chunks table '''
    sql_stmt = \
        'INSERT INTO chunks (chu_id, tur_id, chunk_index, start_time, ' \
            'end_time, duration, words)\n' \
        'VALUES (?,?,?,?,?,?,?);'
    dbc.execute(
        sql_stmt, 
        (chu_id, tur_id, chunk_index, start_time, end_time, duration, words))



################################################################################
#                           SETTERS (SIMPLE UPDATES)                           #
################################################################################

def set_turn_index_ses():
    ''' sets the session-wide turn index for all sessions '''
    sql_stmt = \
        'UPDATE turns\n' \
        'SET turn_index_ses = (\n' \
        '    SELECT COUNT(tur2.tur_id) + 1\n' \
        '    FROM   turns tur2\n' \
        '    JOIN   tasks tsk2\n' \
        '    ON     tur2.tsk_id == tsk2.tsk_id\n' \
        '    WHERE  tsk2.ses_id = (\n' \
        '        SELECT ses_id FROM tasks WHERE tsk_id = turns.tsk_id\n' \
        '    )\n' \
	    '    -- assumes that tsk_id is sorted by task_index within ses_id\n' \
	    '    AND   (   tur2.tsk_id < turns.tsk_id\n' \
	    '           OR (    tur2.tsk_id = turns.tsk_id\n' \
        '               AND tur2.turn_index < turns.turn_index\n' \
        '           )\n' \
	    '    )\n' \
        ');'
    dbc.execute(sql_stmt)


def set_features(chu_id, features):
    ''' sets features of given chunk '''
    sql_stmt = \
        'UPDATE chunks\n' \
        'SET    pitch_min = ?,\n' \
        '       pitch_max = ?,\n' \
        '       pitch_mean = ?,\n' \
        '       pitch_std = ?,\n' \
        '       rate_syl = ?,\n' \
        '       rate_vcd = ?,\n' \
        '       intensity_min = ?,\n' \
        '       intensity_max = ?,\n' \
        '       intensity_mean = ?,\n' \
        '       intensity_std = ?,\n' \
        '       jitter = ?,\n' \
        '       shimmer = ?,\n' \
        '       nhr = ?\n' \
        'WHERE  chu_id == ?;'
    dbc.execute(sql_stmt, 
                (features['f0_min'],
                 features['f0_max'],
                 features['f0_mean'],
                 features['f0_std'],
                 features['rate_syl'],
                 features['vcd2tot_frames'],
                 features['int_min'],
                 features['int_max'],
                 features['int_mean'],
                 features['int_std'],
                 features['jitter'],
                 features['shimmer'],
                 features['nhr'],
                 chu_id))



################################################################################
#                           GETTERS (SIMPLE SELECTS)                           #
################################################################################

def get_ses_id(tsk_id):
    ''' returns ses_id for given tsk_id '''
    sql_stmt = \
        'SELECT ses_id\n' \
        'FROM   tasks\n' \
        'WHERE  tsk_id == ?;'
    return int(dbc.execute(sql_stmt, (tsk_id,)).fetchall()[0][0])


def get_tsk_ids():
    ''' returns tsk_id for all tasks in order '''
    sql_stmt = \
        'SELECT tsk_id\n' \
        'FROM   tasks\n' \
        'ORDER BY tsk_id;'
    return [int(v[0]) for v in dbc.execute(sql_stmt).fetchall()]


def get_ses_ids():
    ''' returns ses_id for all sessions in order '''
    sql_stmt = \
        'SELECT ses_id\n' \
        'FROM   sessions\n' \
        'ORDER BY ses_id;'
    return [int(v[0]) for v in dbc.execute(sql_stmt).fetchall()]


def get_tsk_ses_ids(tsk_or_ses):
    ''' returns task or session id's as needed '''
    return get_tsk_ids() if tsk_or_ses == 'tsk' else get_ses_ids()


def get_a_or_b(tsk_or_ses, tsk_ses_id, spk_id):
    ''' returns whether given speaker is A or B in given task/session '''
    ses_id = tsk_ses_id if tsk_or_ses == 'ses' else get_ses_id(tsk_ses_id)
    sql_stmt = \
        'SELECT CASE\n' \
        '           WHEN spk_id_a == ?\n' \
        '           THEN "A"\n' \
        '           WHEN spk_id_b == ?\n' \
        '           THEN "B"\n' \
        '           ELSE ""\n' \
        '       END\n' \
        'FROM   sessions\n' \
        'WHERE  ses_id == ?;'
    return dbc.execute(sql_stmt, (spk_id, spk_id, ses_id)).fetchall()[0][0]


def get_tasks():
    ''' returns basic task data for all tasks in order of tsk_id '''
    sql_stmt = \
        'SELECT tsk_id, ses_id, task_index\n' \
        'FROM tasks\n' \
        'ORDER BY tsk_id;'
    return dbc.execute(sql_stmt).fetchall()


def get_words(tsk_or_ses, tsk_ses_id):
    ''' gets words for all chunks of given task/session in order '''
    assert tsk_or_ses in ['tsk', 'ses'], 'unknown tsk_or_ses value'
    sql_stmt = \
        'SELECT tur.tur_id,\n' \
        '       CASE\n' \
        '           WHEN tsk.a_or_b == "A" AND tur.speaker_role == "d"\n' \
        '           THEN "A"\n' \
        '           WHEN tsk.a_or_b == "B" AND tur.speaker_role == "f"\n' \
        '           THEN "A"\n' \
        '           ELSE "B"\n' \
        '       END a_or_b,\n' \
        '       chu.words\n' \
        'FROM   chunks chu\n' \
        'JOIN   turns tur\n' \
        'ON     chu.tur_id == tur.tur_id\n' \
        'JOIN   tasks tsk\n' \
        'ON     tur.tsk_id == tsk.tsk_id\n' \
        'WHERE  tsk.' + tsk_or_ses + '_id == ?\n' \
        'ORDER BY tsk.ses_id,\n' \
        '         tsk.task_index,\n' \
        '         tur.turn_index,\n' \
        '         chu.chunk_index;'
    return dbc.execute(sql_stmt, (tsk_ses_id,)).fetchall()



################################################################################
#                                    OTHER                                     #
################################################################################

def executescript(path, fname):
    ''' executes given file as script '''
    # users should obviously not have the ability to execute arbitrary scripts,  
    # but this project is not for end users, just privately run data analysis
    dbc.executescript(''.join(fio.readlines(path, fname)))


def pd_read_sql_query(sql_stmt='', sql_fname=''):
    ''' runs given sql query and returns pandas dataframe of result 

    establishes and closes db connection for each call

    args:
        sql_stmt: sql statement to execute (only run if no filename given)
        sql_fname: filename (in cfg.SQL_PATH) from where to load sql statement
    returns:
        pandas dataframe with query result set 
    '''
    assert len(sql_stmt) > 0 or len(sql_fname) > 0, 'need sql query or filename'
    if len(sql_fname) > 0:
        sql_stmt = '\n'.join(fio.readlines(cfg.SQL_PATH, sql_fname))
    df = pd.read_sql_query(sql_stmt, get_conn())
    return df


def find_chunks(tsk_id, a_or_b):
    ''' yields all chunks for given speaker (A or B) in given session ''' 
    sql_stmt = \
        'SELECT chu.chu_id,\n' \
        '       chu.words,\n' \
        '       chu.start_time,\n' \
        '       chu.end_time\n' \
        'FROM   chunks chu\n' \
        'JOIN   turns tur\n' \
        'ON     chu.tur_id == tur.tur_id\n' \
        'JOIN   tasks tsk\n' \
        'ON     tur.tsk_id == tsk.tsk_id\n' \
        'WHERE  tsk.tsk_id == ?\n' \
        'AND    CASE\n' \
        '           WHEN tur.speaker_role == "d" AND tsk.a_or_b == "A"\n' \
        '           THEN "A"\n' \
        '           WHEN tur.speaker_role == "f" AND tsk.a_or_b == "B"\n' \
        '           THEN "A"\n' \
        '           ELSE "B"\n' \
        '       END == ?\n'
    res = dbc.execute(sql_stmt, (tsk_id, a_or_b)).fetchall()
    for chu_id, words, start, end in res:
        yield(chu_id, words, start, end)







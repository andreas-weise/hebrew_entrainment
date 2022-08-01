import aux
import cfg
import db
import fio



def populate_speakers():
    for row in fio.read_csv(cfg.META_PATH_HMT, 'spk.csv', skip_header=True):
        gender = 'f' if row[1] == 'female' else \
                 'm' if row[1] == 'male' else None
        db.ins_spk(spk_id=row[0], gender=gender, age=row[2], born_in=row[3], 
                   native_lang=row[4], years_edu=row[5])
    db.commit()


def populate_sessions():
    for row in fio.read_csv(cfg.META_PATH_HMT, 'ses_tsk.csv', skip_header=True):
        if row[0][-1] == 'B':
            # each session is listed twice, skip 'B' tasks
            continue
        db.ins_ses(
            ses_id=row[0][3:6], spk_id_a=row[2], spk_id_b=row[3], fam=row[10])
    db.commit()


def populate_tasks():
    tsk_id = 1
    for row in fio.read_csv(cfg.META_PATH_HMT, 'ses_tsk.csv', skip_header=True):
        task_index = 1 if row[0][-1] == 'A' else 2
        a_or_b = 'A' if row[8] == 'master' else 'B'
        db.ins_tsk(tsk_id, ses_id=row[0][3:6], map_index=row[1], 
                   task_index=task_index, a_or_b=a_or_b)
        tsk_id += 1
    db.commit()


def _load_transcripts(ses_id, task_index):
    ''' reads both transcripts for given task, returns sorted list of chunks '''
    lines = []
    for d_or_f in ['d', 'f']:
        fname = '%s%s_%s.txt' % (ses_id, chr(64 + task_index), d_or_f)
        for line in fio.readlines(cfg.TRANS_PATH_HMT, fname):
            start = float(line.split()[0])
            end = float(line.split()[1])
            words = aux.preprocess_transcript(' '.join(line.split()[2:]))
            lines += [(start, end, words, d_or_f)]
    lines.sort()
    return lines


def populate_turns_and_chunks():
    # global ids for turns and chunks (one tur_id per speaker, see below)
    tur_ids = [0, 0]
    chu_id = 0

    for tsk_id, ses_id, task_index in db.get_tasks():
        lines = _load_transcripts(ses_id, task_index)
        
        # words, end of last chunk, turn index, and chunk index per speaker
        ends = [0.0, 0.0]
        tur_cnts = [0, 0]
        chu_cnts = [0, 0]
        
        # create chunks for all non-empty lines and turns as needed 
        for start, end, words, d_or_f in lines:
            # index of current speaker in arrays (1-idx is other speaker)
            idx = 0 if d_or_f == 'd' else 1

            if words != '':
                # check whether this is a new turn
                if ends[1-idx] > ends[idx] \
                or tur_cnts[1-idx] > tur_cnts[idx] \
                or tur_cnts[idx] == 0:
                    # new turn, update index and count
                    tur_cnts[idx] = max(tur_cnts) + 1
                    tur_ids[idx] = max(tur_ids) + 1
                    chu_cnts[idx] = 1
                else:
                    # continuation of old turn
                    chu_cnts[idx] += 1
                ends[idx] = end

                if chu_cnts[idx] == 1:
                    # first chunk in turn; insert turn first
                    db.ins_tur(tur_ids[idx], tsk_id, tur_cnts[idx], d_or_f)
                chu_id += 1
                db.ins_chu(chu_id, tur_ids[idx], chu_cnts[idx], 
                           start, end, end-start, words)
        db.commit()


def extract_features(tsk_id, ses_id, task_index):
    ''' runs feature extraction for all chunks in given task, updates db '''
    db.connect(cfg.CORPUS_ID_HMT)
    path = cfg.get_corpus_path(cfg.CORPUS_ID_HMT)
    for a_or_b in ['A', 'B']:
        fname = '%d%s.%s.wav' % (ses_id, chr(64 + task_index), a_or_b)
        all_features = {}
        for chu_id, words, start, end in db.find_chunks(tsk_id, a_or_b):
            if end - start >= 0.04: # min duration for 75Hz min pitch
                all_features[chu_id] = fio.extract_features(
                    path, fname, tsk_id, chu_id, words, start, end)
        # function is invoked in parallel, database might be locked;
        # keep trying to update until it works
        done = False
        while not done:
            try:
                for chu_id, features in all_features.items():
                    db.set_features(chu_id, features)
                db.commit()
                done = True
            except sqlite3.OperationalError:
                pass
    db.close()









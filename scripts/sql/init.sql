-- basic tables for analysis of hebrew map task corpus

-- tables created here
DROP TABLE IF EXISTS chunks;
DROP TABLE IF EXISTS turns;
DROP TABLE IF EXISTS tasks;
DROP TABLE IF EXISTS sessions;
DROP TABLE IF EXISTS speakers;

-- auxiliary tables created later
DROP TABLE IF EXISTS chunk_pairs;
DROP TABLE IF EXISTS halfway_points;



-- speakers.csv
CREATE TABLE speakers (
    spk_id          INTEGER NOT NULL,
    gender          TEXT,
    age             INTEGER,
    born_in         TEXT,
    native_lang     TEXT,
    years_edu       INTEGER,
    PRIMARY KEY (spk_id)
);



-- sessions.csv
CREATE TABLE sessions (
    -- sessions between pairs of speakers
    ses_id      INTEGER NOT NULL,
    spk_id_a    INTEGER NOT NULL,
    spk_id_b    INTEGER NOT NULL,
    -- 0 = new; 1 = transcript processed; 2 = done
    status      INTEGER DEFAULT 0,
    PRIMARY KEY (ses_id),
    FOREIGN KEY (spk_id_a) REFERENCES speakers (spk_id),
    FOREIGN KEY (spk_id_b) REFERENCES speakers (spk_id)
);



CREATE TABLE tasks (
    -- individual map tasks, two per session
    tsk_id          INTEGER NOT NULL,
    ses_id          INTEGER NOT NULL,    
    -- two different maps were used, not in the same order for all pairs
    map_index       INTEGER NOT NULL,
    task_index      INTEGER NOT NULL,
    -- who is describer for this task, a or b; other speaker is follower
    -- find specific speaker index in sessions table
    a_or_b          TEXT NOT NULL,
    PRIMARY KEY (tsk_id),
    FOREIGN KEY (ses_id) REFERENCES sessions (ses_id)
);



CREATE TABLE turns (
    -- turn = "maximal sequence of inter-pausal units from a single speaker"
    -- (Levitan and Hirschberg, 2011)
    tur_id          INTEGER NOT NULL,
    tsk_id          INTEGER NOT NULL,
    turn_type       TEXT,
    turn_index      INTEGER NOT NULL,
    -- whether "d"(escriber) or "f"(ollower) is speaking
    speaker_role    TEXT NOT NULL,
    PRIMARY KEY (tur_id),
    FOREIGN KEY (tsk_id) REFERENCES tasks (tsk_id)
);



CREATE TABLE chunks (
    -- inter-pausal units with acoustic-prosodic and lexical data
    -- "pause-free units of speech from a single speaker separated from one 
    --  another by at least 50ms" (Levitan and Hirschberg, 2011)
    chu_id            INTEGER NOT NULL,
    tur_id            INTEGER NOT NULL,
    chunk_index       INTEGER NOT NULL,
    start_time        NUMERIC,
    end_time          NUMERIC,
    duration          INTEGER,
    words             TEXT,
    pitch_min         NUMERIC,
    pitch_max         NUMERIC,
    pitch_mean        NUMERIC,
    pitch_std         NUMERIC,
    rate_syl          NUMERIC,
    rate_vcd          NUMERIC,
    intensity_min     NUMERIC,
    intensity_max     NUMERIC,
    intensity_mean    NUMERIC,
    intensity_std     NUMERIC,
    jitter            NUMERIC,
    shimmer           NUMERIC,
    nhr               NUMERIC,
    PRIMARY KEY (chu_id),
    FOREIGN KEY (tur_id) REFERENCES turns (tur_id)
);



CREATE UNIQUE INDEX chu_pk ON chunks (chu_id);
CREATE INDEX chu_tur_fk ON chunks (tur_id);
CREATE UNIQUE INDEX tur_pk ON turns (tur_id);
CREATE INDEX tur_tsk_fk ON turns (tsk_id);
CREATE UNIQUE INDEX tsk_pk ON tasks (tsk_id);
CREATE INDEX tsk_ses_fk ON tasks (ses_id);
CREATE UNIQUE INDEX ses_pk ON sessions (ses_id);
CREATE INDEX ses_spk_a_fk ON sessions (spk_id_a);
CREATE INDEX ses_spk_b_fk ON sessions (spk_id_b);
CREATE UNIQUE INDEX spk_pk ON speakers (spk_id);

# Entrainment in Hebrew

This is an updated version of the code for the following papers (with slightly different results):
Weise, A., Silber-Varod, V., Lerner, A., Hirschberg, J., & Levitan, R. (2020). Entrainment in spoken Hebrew dialogues. Journal of Phonetics, 83(November), 1–16. https://doi.org/10.1016/j.wocn.2020.101005
Weise, A., Silber-Varod, V., Lerner, A., Hirschberg, J., & Levitan, R. (2021). “Talk to me with left, right, and angles”: Lexical entrainment in spoken Hebrew dialogue. EACL 2021, 292–299.

Five acoustic-prosodic and three lexical measures are computed and applied to the Hebrew Map Task Corpus of the Open University of Israel. The corpus is not included. See documentation.pdf for details about it and how to request access.

## Directory Overview

<ul>
    <li>jupyter: a sequence of Jupyter notebooks that invoke all SQL/python code to process and analyze the corpora</li>
    <li>praat: single Praat script for feature extraction</li>
    <li>python: modules for data processing and analysis invoked from the Jupyter notebooks; file overview:
        <ul>
            <li>ana.py: functions for the analysis of all entrainment measures</li>
            <li>ap.py: implementation of five acoustic-prosodic entrainment measures</li>
            <li>aux.py: auxiliary functions</li>
            <li>cfg.py: configuration constants; if you received the corpus data (separately), configure the correct paths here</li>
            <li>db.py: interaction with the corpus database</li>
            <li>fio.py: file i/o</li>
            <li>hmt.py: functions specific to the hebrew map task corpus</li>
            <li>lex.py: implementation of three lexical entrainment measures</li>
        </ul>
    </li>
    <li>sql: core sql scripts that initialize the database files and are used during processing/analysis; file overview:
        <ul>
            <li>aux_tables.sql: creates chunk_pairs table with turn exchanges and non-adjacent IPU pairs for local entrainment measures</li>
            <li>big_table.sql: SELECT to flatten normalized, hierarchical schema into one wide, unnormalized table for analysis</li>
            <li>cleanup.sql: auxiliary script for cleanup after feature extraction</li>
            <li>fix_timestamps.sql: ensures continuous timestamps for all chunks in a session (no reset per task)</li>
            <li>init_hmt.sql: creates and documents the hierarchical database schema for the hebrew map task corpus</li>
            <li>speaker_pairs.sql: SELECT to determine partner and non-partner pairs of speakers for analysis</li>
        </ul>
    </li>
</ul>

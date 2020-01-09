-- auxiliary tables for entrainment analysis
-- turn exchanges (fake and real) plus halfway points of tasks/sessions
-- note: 
--     selection of non-adjacent turn exchanges involves randomness
--     results for local similarity somewhat depend on this 

DROP TABLE IF EXISTS consecutive_chunks;
DROP TABLE IF EXISTS tmp;
DROP TABLE IF EXISTS tmp2;
DROP TABLE IF EXISTS tmp3;
DROP TABLE IF EXISTS chu_rnd_pairs;
DROP TABLE IF EXISTS chunk_pairs;
DROP TABLE IF EXISTS halfway_points;



CREATE TABLE consecutive_chunks
-- ids of consecutive chunks, used to compute local entrainment measures
AS
WITH chu_last AS
(
 -- id for all chunks which are last in their turn
 SELECT chu.chu_id
 FROM   chunks chu
 JOIN   turns tur
 ON     chu.tur_id == tur.tur_id
 WHERE  NOT EXISTS (
                    SELECT 1
                    FROM   chunks chu2
                    JOIN   turns tur2
                    ON     chu2.tur_id == tur2.tur_id
                    WHERE  tur.tsk_id == tur2.tsk_id
                    AND    tur.turn_index == tur2.turn_index
                    AND    chu.chunk_index < chu2.chunk_index
                   )
), 
chu AS
(
 SELECT chu.chu_id,
        tur.tur_id,
        tsk.tsk_id,
        ses.ses_id,
        CASE 
            WHEN tur.speaker_role == 'd' AND tsk.a_or_b == 'A'
            THEN spk_a.spk_id
            WHEN tur.speaker_role == 'f' AND tsk.a_or_b == 'B'
            THEN spk_a.spk_id
            ELSE spk_b.spk_id
        END spk_id,
        chu.chunk_index,
        tur.turn_index,
        tur.turn_type,
        tur.speaker_role,
        CASE 
            WHEN tur.speaker_role == 'd' AND tsk.a_or_b == 'A'
            THEN 'A'
            WHEN tur.speaker_role == 'f' AND tsk.a_or_b == 'B'
            THEN 'A'
            ELSE 'B'
        END a_or_b,
        CASE 
            WHEN tur.speaker_role == 'd' AND tsk.a_or_b == 'A'
            THEN spk_a.gender
            WHEN tur.speaker_role == 'f' AND tsk.a_or_b == 'B'
            THEN spk_a.gender
            ELSE spk_b.gender
        END gender
 FROM   chunks chu
 JOIN   turns tur
 ON     chu.tur_id == tur.tur_id
 JOIN   tasks tsk
 ON     tur.tsk_id == tsk.tsk_id
 JOIN   sessions ses
 ON     tsk.ses_id == ses.ses_id
 JOIN   speakers spk_a
 ON     ses.spk_id_a == spk_a.spk_id
 JOIN   speakers spk_b
 ON     ses.spk_id_b == spk_b.spk_id
)

SELECT chu1.chu_id chu_id1,
       chu2.chu_id chu_id2,
       chu1.tur_id tur_id1,
       chu2.tur_id tur_id2,
       chu1.tsk_id tsk_id,
       chu1.ses_id ses_id,
       chu1.spk_id spk_id1,
       chu2.spk_id spk_id2,
       chu1.chunk_index chunk_index1,
       chu2.chunk_index chunk_index2,
       chu1.turn_index turn_index1,
       chu2.turn_index turn_index2,
       chu1.turn_type turn_type1,
       chu2.turn_type turn_type2,
       chu1.speaker_role speaker_role1,
       chu2.speaker_role speaker_role2,
       chu1.a_or_b speaker1_a_or_b,
       chu2.a_or_b speaker2_a_or_b,
       chu1.gender gender1,
       chu2.gender gender2
FROM   chu chu1
LEFT JOIN chu_last
ON     chu1.chu_id == chu_last.chu_id
JOIN   chu chu2
ON     chu1.tsk_id == chu2.tsk_id
AND   (
       (chu1.turn_index == chu2.turn_index 
        AND chu1.chunk_index + 1 == chu2.chunk_index)
       OR
       (chu_last.chu_id IS NOT NULL 
        AND chu1.turn_index + 1 == chu2.turn_index 
        AND chu2.chunk_index == 1)
      )
ORDER BY chu1.tsk_id,
         chu1.turn_index,
         chu1.chunk_index;

UPDATE turns
SET    turn_type = 'NOT_OVERLAPPING';

UPDATE turns
SET    turn_type = 'OVERLAPPING'
WHERE  tur_id IN (
                  SELECT tur_id2
                  FROM   consecutive_chunks con
                  JOIN   chunks chu1
                  ON     con.chu_id1 == chu1.chu_id
                  JOIN   chunks chu2
                  ON     con.chu_id2 == chu2.chu_id
                  WHERE  chunk_index2 = 1
                  AND    chu2.start_time < chu1.end_time
                 );

UPDATE consecutive_chunks
SET    turn_type1 = 'NOT_OVERLAPPING',
       turn_type2 = 'NOT_OVERLAPPING';

UPDATE consecutive_chunks
SET    turn_type1 = 'OVERLAPPING'
WHERE  tur_id1 IN (
                   SELECT tur_id
                   FROM   turns
                   WHERE  turn_type == 'OVERLAPPING'
                  );

UPDATE consecutive_chunks
SET    turn_type2 = 'OVERLAPPING'
WHERE  tur_id2 IN (
                   SELECT tur_id
                   FROM   turns
                   WHERE  turn_type == 'OVERLAPPING'
                  );



-- tmp tables for randomization 
CREATE TABLE tmp 
-- all pairs of turn-final chunks with turn-initial chunks from other speaker
-- in same role (except actual next chunk) in random order; notes:
--     excludes those with a null value for any feature
--     restriction to same task is too narrow for games corpus
--     restriction to same role is necessary due to prosodic differences
AS
WITH chu_ini AS
(
 -- all turn-initial chunks (marked by chunk_index == 1)
 -- note: union is needed because task-final chunks are never listed first
 SELECT con.chu_id1 chu_id,
        con.spk_id1 spk_id,
        con.speaker_role1 speaker_role
 FROM   consecutive_chunks con
 WHERE  con.chunk_index1 == 1
 AND    con.turn_type1 != "OVERLAPPING"

 UNION

 SELECT con.chu_id2 chu_id,
        con.spk_id2 spk_id,
        con.speaker_role2 speaker_role
 FROM   consecutive_chunks con
 WHERE  con.chunk_index2 == 1
 AND    con.turn_type2 != "OVERLAPPING"
)
SELECT con.chu_id1 chu_id1, 
       ini.chu_id chu_id2
FROM   consecutive_chunks con
JOIN   chu_ini ini
ON     con.spk_id2 == ini.spk_id
AND    con.speaker_role2 == ini.speaker_role
AND    con.chu_id2 != ini.chu_id
JOIN   chunks chu
ON     ini.chu_id == chu.chu_id
-- change in role marks chunk as turn-final
WHERE  con.speaker_role1 != con.speaker_role2
-- only those chunks without null values 
-- (ensure average over same number of non-adjacent IPUs per feature)
AND    chu.pitch_min IS NOT NULL 
AND    chu.pitch_max IS NOT NULL 
AND    chu.pitch_mean IS NOT NULL 
AND    chu.pitch_std IS NOT NULL 
AND    chu.intensity_min IS NOT NULL 
AND    chu.intensity_max IS NOT NULL 
AND    chu.intensity_mean IS NOT NULL 
AND    chu.intensity_std IS NOT NULL 
AND    chu.rate_vcd IS NOT NULL 
AND    chu.rate_syl IS NOT NULL 
AND    chu.jitter IS NOT NULL 
AND    chu.shimmer IS NOT NULL 
AND    chu.nhr IS NOT NULL 
ORDER BY con.chu_id1, random();

CREATE TABLE tmp2 
AS
SELECT chu_id1, 
       chu_id2,
       rowid rid
FROM   tmp;

CREATE TABLE tmp3
AS
SELECT chu_id1, 
       min(rowid) min_rid
FROM   tmp 
GROUP BY chu_id1;

DROP TABLE tmp;

CREATE TABLE chu_rnd_pairs
AS
-- pairs each turn-final chunk with 10 random turn-initial chunks 
-- from the speaker who started the next turn
SELECT rnd.chu_id1 chu_id1,
       rnd.chu_id2 chu_id2
FROM   tmp2 rnd
JOIN   tmp3 ids
ON     rnd.chu_id1 == ids.chu_id1
WHERE  rnd.rid < ids.min_rid + 10;

DROP TABLE tmp2;
DROP TABLE tmp3;



CREATE TABLE chunk_pairs
AS
-- pairs of turn-final and turn-initial chunks;
-- used to compute local entrainment measures;
-- some are adjacent ('p'), some are not ('x');
SELECT 'p' p_or_x, -- adjacent pairs
       -- for adjacent pairs, both chunks belong to this session
       con.ses_id ses_id,
       -- for adjacent pairs, both chunks belong to this task
       con.tsk_id tsk_id,
       -- turn-final chunk
       con.chu_id1 chu_id1,
       -- turn-initial chunk
       con.chu_id2 chu_id2,
       -- id of speaker who spoke turn-initial chunk
       con.spk_id2 spk_id,
       -- role of speaker who spoke turn-initial chunk
       con.speaker_role2 speaker_role,
       -- gender of speaker who spoke turn-final chunk
       con.gender1 gender1,
       -- gender of speaker who spoke turn-initial chunk
       con.gender2 gender2,
       -- identifier 'A' or 'B' of the speaker who spoke turn-initial chunk
       con.speaker2_a_or_b a_or_b
FROM   consecutive_chunks con
-- change in role marks chunk 1 as turn-final
WHERE  con.speaker_role1 != con.speaker_role2
AND    con.turn_type2 != 'OVERLAPPING'

UNION ALL

SELECT 'x' p_or_x, -- non-adjacent pairs
       -- for non-adjacent pairs, second chunk might not belong to this session
       -- (only if speaker participated in multiple sessions)
       con.ses_id ses_id,
       -- for non-adjacent pairs, second chunk might not belong to this task
       -- (only if speaker participated in multiple tasks with same role)
       con.tsk_id tsk_id,
       -- turn-final chunk
       con.chu_id1 chu_id1,
       -- turn-initial chunk 
       -- (non-adjacent, but same speaker in same role as the actual next chunk)
       rnd.chu_id2 chu_id2,
       -- id of speaker who spoke turn-initial chunk
       con.spk_id2 spk_id,
       -- role of speaker who spoke turn-initial chunk
       con.speaker_role2 speaker_role,
       -- gender of speaker who spoke turn-final chunk
       con.gender1 gender1,
       -- gender of speaker who spoke turn-initial chunk
       con.gender2 gender2,
       -- identifier 'A' or 'B' of speaker who spoke turn-initial chunk
       con.speaker2_a_or_b a_or_b
FROM   consecutive_chunks con
JOIN   chu_rnd_pairs rnd
ON     con.chu_id1 == rnd.chu_id1 
-- change in role marks chunk 1 as turn-final
WHERE  con.speaker_role1 != con.speaker_role2
AND    con.turn_type2 != 'OVERLAPPING';

CREATE INDEX chp_ses_fk ON chunk_pairs (ses_id);
CREATE INDEX chp_tsk_fk ON chunk_pairs (tsk_id);
CREATE INDEX chp_chu_fk1 ON chunk_pairs (chu_id1);
CREATE INDEX chp_chu_fk2 ON chunk_pairs (chu_id2);

DROP TABLE chu_rnd_pairs;
DROP TABLE consecutive_chunks;



CREATE TABLE tmp
AS
-- number of chunks per task
SELECT tur.tsk_id tsk_id,
       COUNT(*) chu_count
FROM   chunks chu
JOIN   turns tur
ON     chu.tur_id == tur.tur_id
GROUP BY tur.tsk_id;

CREATE TABLE tmp2
AS
-- number of chunks per session
SELECT tsk.ses_id ses_id,
       COUNT(*) chu_count
FROM   chunks chu
JOIN   turns tur
ON     chu.tur_id == tur.tur_id
JOIN   tasks tsk
ON     tur.tsk_id == tsk.tsk_id
GROUP BY tsk.ses_id;

CREATE TABLE halfway_points
AS
-- chunks marking halfway points per task and per session, respectively
-- (used for global convergence measure)
SELECT NULL ses_id,
       tsk.tsk_id,
       tsk.task_index,
       tur.turn_index,
       chu.chunk_index
FROM   chunks chu
JOIN   turns tur
ON     chu.tur_id == tur.tur_id
JOIN   tasks tsk
ON     tur.tsk_id == tsk.tsk_id
JOIN   tmp
ON     tsk.tsk_id == tmp.tsk_id
WHERE  ROUND(tmp.chu_count / 2.0, 0) == 
       (
           -- number of chunks up to the current one
           -- (same task, previous turn or earlier in same turn)
           SELECT COUNT(*)
           FROM   chunks chu2
           JOIN   turns tur2
           ON     chu2.tur_id == tur2.tur_id
           WHERE  tur.tsk_id == tur2.tsk_id
           AND   (tur2.turn_index < tur.turn_index
                  OR
                  (tur2.turn_index == tur.turn_index 
                   AND chu2.chunk_index <= chu.chunk_index))
       )

UNION ALL

SELECT tsk.ses_id ses_id,
       NULL tsk_id,
       tsk.task_index task_index,
       tur.turn_index turn_index,
       chu.chunk_index chunk_index
FROM   chunks chu
JOIN   turns tur
ON     chu.tur_id == tur.tur_id
JOIN   tasks tsk
ON     tur.tsk_id == tsk.tsk_id
JOIN   tmp2
ON     tsk.ses_id == tmp2.ses_id
WHERE  ROUND(tmp2.chu_count / 2.0, 0) == 
       (
           -- number of chunks up to the current one
           -- (same or previous task, previous turn or earlier in same turn)
           SELECT COUNT(*)
           FROM   chunks chu2
           JOIN   turns tur2
           ON     chu2.tur_id == tur2.tur_id
           JOIN   tasks tsk2
           ON     tur2.tsk_id == tsk2.tsk_id
           WHERE  tsk2.ses_id == tsk.ses_id
           AND   (tsk2.task_index < tsk.task_index
                  OR
                  (tsk2.task_index == tsk.task_index
                   AND tur2.turn_index < tur.turn_index)
                  OR
                  (tsk2.task_index == tsk.task_index
                   AND tur2.turn_index == tur.turn_index 
                   AND chu2.chunk_index <= chu.chunk_index))
       );

DROP TABLE tmp;
DROP TABLE tmp2;




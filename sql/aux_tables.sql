-- chunk_pairs table: turn exchanges (fake&real) for local entrainment analysis;
-- note 1:
--     overlaps between turn-final and the immediately following turn-initial 
--     chunks can be excluded entirely or limited to at most 50 percent of the 
--     turn-final chunk; in addition, for overlaps of more than 50 percent, 
--     pairs can be created for a turn-initial chunk with the chunk immediately 
--     preceding the turn-final one (an almost adjacent chunk), if that one was 
--     spoken by the same speaker as the turn-final one; this is meant to 
--     reflect that the turn-initial chunk is really a response to the almost 
--     adjacent one, not to the turn-final one which had barely started; 
--     overlaps should only be allowed if speakers are recorded on separate, 
--     isolated channels, otherwise they can cause cross-channel contamination; 
--     the three cases are represented as subselects in the 
--     "CREATE TABLE chunk_pairs" statement below; comment them in/out as needed
-- note 2: 
--     selection of non-adjacent turn exchanges involves randomness;
--     code assumes continuous timestamps per session, no reset per task!
--     (only matters if turn exchanges across tasks are relevant, which should
--      only be the case if the task boundaries are "virtual", not a real break
--      in the recording; in that case, timestamps should also be continuous;
--      in the hebrew map task corpus, turn exchanges across tasks are not 
--      relevant, so it is ok that the timestamps are not continuous)



DROP TABLE IF EXISTS consecutive_2_chunks;
DROP TABLE IF EXISTS consecutive_3_chunks;
DROP TABLE IF EXISTS tmp;
DROP TABLE IF EXISTS tmp2;
DROP TABLE IF EXISTS tmp3;
DROP TABLE IF EXISTS chunk_pairs;



CREATE TABLE consecutive_2_chunks
-- pairs of consecutive chunks with meta-data
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
tur_last AS
(
 -- id for all turns which are last in their task
 SELECT tur.tur_id
 FROM   turns tur
 WHERE  NOT EXISTS (
                    SELECT 1
                    FROM   turns tur2
                    WHERE  tur.tsk_id == tur2.tsk_id
                    AND    tur.turn_index < tur2.turn_index
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
        tsk.task_index,
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
        END gender,
        start_time,
        end_time,
        CASE
            WHEN chu_last.chu_id IS NOT NULL
            THEN 1
            ELSE 0
        END is_last_in_turn,
        CASE
            WHEN chu_last.chu_id IS NOT NULL AND tur_last.tur_id IS NOT NULL
            THEN 1
            ELSE 0
        END is_last_in_task,
        CASE
            WHEN chu.pitch_min IS NOT NULL 
            AND  chu.pitch_max IS NOT NULL 
            AND  chu.pitch_mean IS NOT NULL 
            AND  chu.pitch_std IS NOT NULL 
            AND  chu.intensity_min IS NOT NULL 
            AND  chu.intensity_max IS NOT NULL 
            AND  chu.intensity_mean IS NOT NULL 
            AND  chu.intensity_std IS NOT NULL 
            AND  chu.rate_vcd IS NOT NULL 
            AND  chu.rate_syl IS NOT NULL 
            AND  chu.jitter IS NOT NULL 
            AND  chu.shimmer IS NOT NULL 
            AND  chu.nhr IS NOT NULL 
            THEN 1
            ELSE 0
        END has_all_features
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
 LEFT JOIN chu_last
 ON     chu.chu_id == chu_last.chu_id
 LEFT JOIN tur_last
 ON     tur.tur_id == tur_last.tur_id
)
SELECT chu1.chu_id chu_id1,
       chu2.chu_id chu_id2,
       chu1.tur_id tur_id1,
       chu2.tur_id tur_id2,
       chu1.tsk_id tsk_id1,
       chu2.tsk_id tsk_id2,
       chu1.ses_id ses_id,
       chu1.spk_id spk_id1,
       chu2.spk_id spk_id2,
       chu1.chunk_index chunk_index1,
       chu2.chunk_index chunk_index2,
       chu1.turn_index turn_index1,
       chu2.turn_index turn_index2,
       chu1.task_index task_index1,
       chu2.task_index task_index2,
       chu1.speaker_role speaker_role1,
       chu2.speaker_role speaker_role2,
       chu1.a_or_b speaker1_a_or_b,
       chu2.a_or_b speaker2_a_or_b,
       chu1.gender gender1,
       chu2.gender gender2,
       chu1.start_time start1,
       chu1.end_time end1,
       chu2.start_time start2,
       chu2.end_time end2,
       chu1.has_all_features has_all_features1,
       chu2.has_all_features has_all_features2
FROM   chu chu1
JOIN   chu chu2
ON     chu1.ses_id == chu2.ses_id
AND   (
       (chu1.tsk_id == chu2.tsk_id
        AND chu1.turn_index == chu2.turn_index 
        AND chu1.chunk_index + 1 == chu2.chunk_index)
       OR
       (chu1.tsk_id == chu2.tsk_id
        AND chu1.is_last_in_turn == 1
        AND chu1.turn_index + 1 == chu2.turn_index 
        AND chu2.chunk_index == 1)
       OR
       (chu1.task_index + 1 == chu2.task_index
        AND chu1.is_last_in_task == 1
        AND chu2.turn_index == 1
        AND chu2.chunk_index == 1)
      )
ORDER BY chu1.ses_id,
         chu1.task_index,
         chu1.turn_index,
         chu1.chunk_index;

CREATE UNIQUE INDEX con_uk1 ON consecutive_2_chunks (chu_id1);
CREATE UNIQUE INDEX con_uk2 ON consecutive_2_chunks (chu_id2);



CREATE TABLE consecutive_3_chunks
AS
-- triplets of consecutive chunks with meta-data
SELECT con1.chu_id1,
       con1.chu_id2,
       con2.chu_id2 chu_id3,
       con1.tur_id1,
       con1.tur_id2,
       con2.tur_id2 tur_id3,
       con1.tsk_id1,
       con1.tsk_id2,
       con2.tsk_id2 tsk_id3,
       con1.ses_id,
       con1.spk_id1,
       con1.spk_id2,
       con2.spk_id2 spk_id3,
       con1.chunk_index1,
       con1.chunk_index2,
       con2.chunk_index2 chunk_index3,
       con1.turn_index1,
       con1.turn_index2,
       con2.turn_index2 turn_index3,
       con1.task_index1,
       con1.task_index2,
       con2.task_index2 chunk_index3,
       con1.speaker_role1,
       con1.speaker_role2,
       con2.speaker_role2 speaker_role3,
       con1.speaker1_a_or_b,
       con1.speaker2_a_or_b,
       con2.speaker2_a_or_b speaker3_a_or_b,
       con1.gender1,
       con1.gender2,
       con2.gender2 gender3,
       con1.start1,
       con1.start2,
       con2.start2 start3,
       con1.end1,
       con1.end2,
       con2.end2 end3,
       con1.has_all_features1,
       con1.has_all_features2,
       con2.has_all_features2 has_all_features3
FROM   consecutive_2_chunks con1
JOIN   consecutive_2_chunks con2
ON     con1.chu_id2 == con2.chu_id1;



CREATE TABLE chunk_pairs
AS
-- pairs of turn-final and turn-initial chunks;
-- used to compute local entrainment measures;
-- some are adjacent ('p'), some are not ('x');
-- all belong to the same session but not necessarily to the same task;
-- this initial create inserts all adjacent and almost adjacent pairs (three 
-- separate selects for different adjacency types; for hebrew corpus, all 
-- overlapping turn exchanges are excluded to avoid cross-channel contamination; 
-- for other corpora, this may not be necessary);
-- non-adjacent pairs are inserted separately below
SELECT 'p' p_or_x, -- truly adjacent pairs, no overlap between chunks
       -- turn-final chunk (adjacent to turn-initial chunk below)
       con.chu_id1 chu_id1,
       -- turn-initial chunk
       con.chu_id2 chu_id2,
       NULL rid
FROM   consecutive_2_chunks con
-- change in speaker marks chunk 1 as turn-final
WHERE  con.spk_id1 != con.spk_id2
-- turn-initial chunk starts only after turn-final is complete
AND    con.end1 <= con.start2
-- both chunks have all features
AND    con.has_all_features1
AND    con.has_all_features2

-- UNION ALL
-- 
-- SELECT 'p' p_or_x, -- truly adjacent pairs, some overlap between chunks
--        -- turn-final chunk (adjacent to turn-initial chunk below)
--        con.chu_id1 chu_id1,
--        -- turn-initial chunk
--        con.chu_id2 chu_id2,
--        NULL rid
-- FROM   consecutive_2_chunks con
-- -- change in speaker marks chunk 1 as turn-final
-- WHERE  con.spk_id1 != con.spk_id2
-- -- turn-initial chunk starts only after turn-final is at least half complete
-- -- (only works across tasks with continuous timestamps)
-- AND    con.start2 < con.end1
-- AND    con.start2 >= con.start1 + (con.end1 - con.start1) / 2
-- -- both chunks have all features
-- AND    con.has_all_features1
-- AND    con.has_all_features2
-- 
-- UNION ALL
-- 
-- SELECT 'p' p_or_x, -- almost adjacent pairs, treated as adjacent 
--        -- (chunk 3 is treated as a response to chunk 1; see note at the top)
--        -- almost turn-final chunk (almost adjacent to turn-initial chunk)
--        con.chu_id1 chu_id1,
--        -- turn-initial chunk
--        con.chu_id3 chu_id2,
--        NULL rid
-- FROM   consecutive_3_chunks con
-- -- change in speaker marks chunk 2 as turn-final
-- WHERE  con.spk_id2 != con.spk_id3
-- -- turn-initial chunk starts before turn-final is at least half complete
-- -- and chunk 1 is from same speaker as chunk 2 (3 is really a response to 1)
-- AND    con.start3 < con.start2 + (con.end2 - con.start2) / 2
-- AND    con.spk_id1 == con.spk_id2
-- -- both chunks have all features
-- AND    con.has_all_features1
-- AND    con.has_all_features3
;

CREATE INDEX chp_chu_fk1 ON chunk_pairs (chu_id1);
CREATE INDEX chp_chu_fk2 ON chunk_pairs (chu_id2);



CREATE TABLE tmp
AS
-- all possible non-adjacent choices per turn-initial chunk in random order; 
-- possible choices are those turn-final or almost turn-final chunks by the same
-- speaker in the same role in the same session as the adjacent chunk and which
-- are included in the analysis as an adjacent or almost adjacent chunk to some
-- other turn-initial chunk (depends on how much is commented out above)
WITH chu AS
(
 SELECT chp.chu_id1,
        chp.chu_id2,
        ses.ses_id,
        CASE 
            WHEN tur1.speaker_role == 'd' AND tsk1.a_or_b == 'A'
            THEN ses.spk_id_a
            WHEN tur1.speaker_role == 'f' AND tsk1.a_or_b == 'B'
            THEN ses.spk_id_a
            ELSE ses.spk_id_b
        END spk_id1,
        tur1.speaker_role speaker_role1
 FROM   chunk_pairs chp
 JOIN   chunks chu1
 ON     chp.chu_id1 == chu1.chu_id
 JOIN   turns tur1
 ON     chu1.tur_id == tur1.tur_id
 JOIN   tasks tsk1
 ON     tur1.tsk_id == tsk1.tsk_id
 JOIN   sessions ses
 ON     tsk1.ses_id == ses.ses_id
)
SELECT chu_x.chu_id1,
       chu_p.chu_id2
FROM   chu chu_p
JOIN   chu chu_x
ON     chu_p.chu_id2 != chu_x.chu_id2
AND    chu_p.ses_id == chu_x.ses_id
AND    chu_p.spk_id1 == chu_x.spk_id1
AND    chu_p.speaker_role1 == chu_x.speaker_role1
ORDER BY chu_p.chu_id2, RANDOM();



CREATE TABLE tmp2 
AS
-- rowid per non-adjacent chunk pair (after putting them in random order)
SELECT chu_id1, 
       chu_id2,
       rowid rid
FROM   tmp;

CREATE INDEX tmp_fk1 ON tmp2 (chu_id1);
CREATE INDEX tmp_fk2 ON tmp2 (chu_id2);



CREATE TABLE tmp3
AS
-- span of rowids for non-adjacent choices per turn-initial chunk
SELECT chu_id2, 
       min(rid) min_rid,
       max(rid) max_rid
FROM   tmp2
GROUP BY chu_id2;

DROP TABLE tmp;



INSERT INTO chunk_pairs
-- add non-adjacent chunk pairs, at least 10 and at least 25 percent of all 
-- possible choices (see comment on tmp table)
SELECT 'x' p_or_x,
       rnd.chu_id1 chu_id1,
       rnd.chu_id2 chu_id2,
       rnd.rid - ids.min_rid rid
FROM   tmp2 rnd
JOIN   tmp3 ids
ON     rnd.chu_id2 == ids.chu_id2
WHERE  rnd.rid < ids.min_rid + MAX(10, 0.25*(ids.max_rid-ids.min_rid+1));



DROP TABLE tmp2;
DROP TABLE tmp3;
DROP TABLE consecutive_2_chunks;
DROP TABLE consecutive_3_chunks;




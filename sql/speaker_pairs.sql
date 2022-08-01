-- finds partners and non-partners for all speakers and tasks/sessions

-- AUXILIARY query with speaker and partner information
-- finds speaker gender and role as well as partner gender from either speaker's 
-- "perspective"; *_s and *_p refer to speaker and partner, resp.
WITH sub AS
(
 SELECT ses.type ses_type,
        ses.ses_id ses_id,
        tsk.tsk_id tsk_id,
        ses.spk_id_a spk_id_s,
        ses.spk_id_b spk_id_p,
        "A" a_or_b_s,
        "B" a_or_b_p,
        spk_a.gender gender_s,
        spk_b.gender gender_p,
        CASE
            WHEN spk_a.gender == spk_b.gender
            THEN UPPER(spk_a.gender)
            ELSE 'X'
        END gender_pair,
        CASE 
            WHEN tsk.a_or_b == "A" 
            THEN "d"
            ELSE "f"
        END role_s,
        tsk.map_index
 FROM   tasks tsk
 JOIN   sessions ses
 ON     tsk.ses_id == ses.ses_id
 JOIN   speakers spk_a
 ON     ses.spk_id_a == spk_a.spk_id
 JOIN   speakers spk_b
 ON     ses.spk_id_b == spk_b.spk_id

 UNION

 SELECT ses.type ses_type,
        ses.ses_id ses_id,
        tsk.tsk_id tsk_id,
        ses.spk_id_b spk_id_s,
        ses.spk_id_a spk_id_p,
        "B" a_or_b_s,
        "A" a_or_b_p,
        spk_b.gender gender_s,
        spk_a.gender gender_p,
        CASE
            WHEN spk_a.gender == spk_b.gender
            THEN UPPER(spk_a.gender)
            ELSE 'X'
        END gender_pair,
        CASE 
            WHEN tsk.a_or_b == "B"
            THEN "d"
            ELSE "f"
        END role_s,
        tsk.map_index
 FROM   tasks tsk
 JOIN   sessions ses
 ON     tsk.ses_id == ses.ses_id
 JOIN   speakers spk_a
 ON     ses.spk_id_a == spk_a.spk_id
 JOIN   speakers spk_b
 ON     ses.spk_id_b == spk_b.spk_id
)

-- NON-PARTNER query for tasks
-- finds all non-partner tasks and speakers that match gender/role criteria
SELECT "x" p_or_x,
       tgt.ses_type,
       tgt.ses_id ses_id,
       tgt.gender_pair,
       tgt.tsk_id tsk_id,
       tgt.spk_id_s spk_id,
       tgt.a_or_b_s a_or_b,
       prd.ses_id ses_id_paired,
       prd.tsk_id tsk_id_paired,
       prd.spk_id_s spk_id_paired,
       prd.a_or_b_s a_or_b_paired
FROM   sub tgt -- "target"
JOIN   sub prd -- "paired"
-- paired speaker and target speaker must not be the same
ON     tgt.spk_id_s != prd.spk_id_s
-- paired speaker and target speaker's partner must not be the same
AND    tgt.spk_id_p != prd.spk_id_s
-- paired speaker must be engaged in the same type of interaction
AND    tgt.ses_type == prd.ses_type
-- paired speaker must be talking about the same map 
AND    tgt.map_index == prd.map_index
-- paired speaker must have the same gender as the target's partner
AND    tgt.gender_p == prd.gender_s
-- paired speaker's partner must have the same gender as the target speaker
AND    tgt.gender_s == prd.gender_p
-- paired speaker must have the same role as the target speaker's partner
-- i.e., paired speaker and target speaker must have different roles
-- (role is only meaningful if there is more than 1 task per session)
AND    (   tgt.role_s != prd.role_s
        OR (SELECT COUNT(DISTINCT task_index) FROM tasks) == 1
       )
-- target speaker and paired speaker must never have interacted
AND    NOT EXISTS (
           SELECT 1 
           FROM   sessions
           WHERE  spk_id_a IN (tgt.spk_id_s, prd.spk_id_s)
           AND    spk_id_b IN (tgt.spk_id_s, prd.spk_id_s)
       )

UNION

-- NON-PARTNER query for sessions; same as above but with one entry per session
SELECT DISTINCT
       "x" p_or_x,
       tgt.ses_type,
       tgt.ses_id ses_id,
       tgt.gender_pair,
       0 tsk_id,
       tgt.spk_id_s spk_id,
       tgt.a_or_b_s a_or_b,
       prd.ses_id ses_id_paired,
       0 tsk_id_paired,
       prd.spk_id_s spk_id_paired,
       prd.a_or_b_s a_or_b_paired
FROM   sub tgt
JOIN   sub prd
ON     tgt.spk_id_s != prd.spk_id_s
AND    tgt.spk_id_p != prd.spk_id_s
AND    tgt.ses_type == prd.ses_type
AND    tgt.map_index == prd.map_index
AND    tgt.gender_p == prd.gender_s
AND    tgt.gender_s == prd.gender_p
AND    (   tgt.role_s != prd.role_s
        OR (SELECT COUNT(DISTINCT task_index) FROM tasks) == 1
       )
AND    NOT EXISTS (
           SELECT 1 
           FROM   sessions
           WHERE  spk_id_a IN (tgt.spk_id_s, prd.spk_id_s)
           AND    spk_id_b IN (tgt.spk_id_s, prd.spk_id_s)
       )

UNION

-- PARTNER query (both speakers from same task) for tasks
SELECT "p" p_or_x,
       ses_type,
       ses_id,
       gender_pair,
       tsk_id,
       spk_id_s spk_id,
       a_or_b_s a_or_b,
       ses_id ses_id_paired,
       tsk_id tsk_id_paired,
       spk_id_p spk_id_paired,
       a_or_b_p a_or_b_paired
FROM   sub

UNION

-- PARTNER query for sessions; same as above but with one entry per session
SELECT DISTINCT 
	   "p" p_or_x,
       ses_type,
       ses_id,
       gender_pair,
       0 tsk_id,
       spk_id_s spk_id,
       a_or_b_s a_or_b,
       ses_id ses_id_paired,
       0 tsk_id_paired,
       spk_id_p spk_id_paired,
       a_or_b_p a_or_b_paired
FROM   sub






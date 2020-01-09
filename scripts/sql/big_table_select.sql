SELECT ses.ses_id session,
       CASE 
           WHEN tsk.task_index == 1 
           THEN "A"
           ELSE "B"
       END session_part,
       CASE 
           WHEN tsk.a_or_b == "A"
           THEN spk_a.spk_id
           ELSE spk_b.spk_id
       END describer_id,
       CASE 
           WHEN tsk.a_or_b == "A"
           THEN spk_a.gender
           ELSE spk_b.gender
       END describer_gender,
       CASE 
           WHEN tsk.a_or_b == "A"
           THEN spk_b.age
           ELSE spk_a.age
       END describer_age,
       CASE 
           WHEN tsk.a_or_b == "A"
           THEN spk_b.spk_id
           ELSE spk_a.spk_id
       END follower_id,
       CASE 
           WHEN tsk.a_or_b == "A"
           THEN spk_b.gender
           ELSE spk_a.gender
       END follower_gender,
       CASE 
           WHEN tsk.a_or_b == "A"
           THEN spk_b.age
           ELSE spk_a.age
       END follower_age,
       tur.turn_index turn_index,
       chu.chunk_index IPU_index,
       CASE
           WHEN tur.speaker_role == "d"
           THEN "describer"
           ELSE "follower"
       END IPU_speaker,
       chu.start_time IPU_start,
       chu.end_time IPU_end,
       chu.words IPU_transcript,
       chu.pitch_min IPU_pitch_min,
       chu.pitch_max IPU_pitch_max,
       chu.pitch_mean IPU_pitch_mean,
       chu.intensity_min IPU_intensity_min,
       chu.intensity_max IPU_intensity_max,
       chu.intensity_mean IPU_intensity_mean,
       chu.jitter IPU_jitter,
       chu.shimmer IPU_shimmer,
       chu.nhr IPU_nhr,
       chu.rate_syl IPU_speech_rate
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
--WHERE  ses.ses_id = 5
ORDER BY tsk.tsk_id, tur.turn_index, chu.chunk_index


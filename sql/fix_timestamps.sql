-- offset all timestamps in the second task of all sessions by the duration of
-- the first task, i.e., the end of its last chunk

UPDATE chunks
SET    start_time = start_time + (
           SELECT MAX(end_time)
           FROM   chunks chu2 
           JOIN   turns tur2
           ON     chu2.tur_id == tur2.tur_id
           WHERE  tur2.tsk_id == (
                      SELECT DISTINCT tsk4.tsk_id
                      FROM   turns tur3
                      JOIN   tasks tsk3
                      ON     tur3.tsk_id == tsk3.tsk_id
                      JOIN   tasks tsk4
                      ON     tsk3.ses_id == tsk4.ses_id
                      WHERE  tur3.tur_id == chunks.tur_id
                      AND    tsk4.task_index == 1
                  )
       )
WHERE  2 == (
           SELECT DISTINCT tsk5.task_index
           FROM   turns tur5
           JOIN   tasks tsk5
           ON     tur5.tsk_id == tsk5.tsk_id
           WHERE  chunks.tur_id == tur5.tur_id
       );

UPDATE chunks
SET    end_time = end_time + (
           SELECT MAX(end_time)
           FROM   chunks chu2 
           JOIN   turns tur2
           ON     chu2.tur_id == tur2.tur_id
           WHERE  tur2.tsk_id == (
                      SELECT DISTINCT tsk4.tsk_id
                      FROM   turns tur3
                      JOIN   tasks tsk3
                      ON     tur3.tsk_id == tsk3.tsk_id
                      JOIN   tasks tsk4
                      ON     tsk3.ses_id == tsk4.ses_id
                      WHERE  tur3.tur_id == chunks.tur_id
                      AND    tsk4.task_index == 1
                  )
       )
WHERE  2 == (
           SELECT DISTINCT tsk5.task_index
           FROM   turns tur5
           JOIN   tasks tsk5
           ON     tur5.tsk_id == tsk5.tsk_id
           WHERE  chunks.tur_id == tur5.tur_id
       );

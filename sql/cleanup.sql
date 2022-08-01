-- set all features null for any chunk that is missing any feature
UPDATE chunks
SET    pitch_min = NULL,
       pitch_max = NULL,
       pitch_mean = NULL,
       pitch_std = NULL,
       rate_syl = NULL,
       rate_vcd = NULL,
       intensity_min = NULL,
       intensity_max = NULL,
       intensity_mean = NULL,
       intensity_std = NULL,
       jitter = NULL,
       shimmer = NULL,
       nhr = NULL
WHERE  pitch_min IS NULL
OR     pitch_max IS NULL
OR     pitch_mean IS NULL
OR     pitch_std IS NULL
OR     rate_syl IS NULL
OR     rate_vcd IS NULL
OR     intensity_min IS NULL
OR     intensity_max IS NULL
OR     intensity_mean IS NULL
OR     intensity_std IS NULL
OR     jitter IS NULL
OR     shimmer IS NULL
OR     nhr IS NULL;

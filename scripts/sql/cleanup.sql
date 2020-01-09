-- mark missing values properly as NULL

UPDATE chunks
SET    pitch_min = NULL
WHERE  pitch_min == '--undefined--';

UPDATE chunks
SET    pitch_max = NULL
WHERE  pitch_max == '--undefined--';

UPDATE chunks
SET    pitch_mean = NULL
WHERE  pitch_mean == '--undefined--';

UPDATE chunks
SET    pitch_std = NULL
WHERE  pitch_std == '--undefined--';

UPDATE chunks
SET    intensity_std = NULL
WHERE  intensity_std == '--undefined--';

UPDATE chunks
SET    jitter = NULL
WHERE  jitter == '--undefined--'
OR     jitter == "'jitter:6'";

UPDATE chunks
SET    shimmer = NULL
WHERE  shimmer == '--undefined--'
OR     shimmer == "'shimmer:6'";

UPDATE chunks
SET    nhr = NULL
WHERE  nhr == '--undefined--';



-- explicitly compute duration (for convenience)
UPDATE chunks
SET    duration = end_time - start_time;



-- treat consonant-only "words" as individual syllables for speech rate
-- (normally, each vowel is treated as a syllable)
UPDATE chunks
SET    rate_syl = (LENGTH(words) - LENGTH(REPLACE(words, ' ', '')) + 1) / duration
WHERE  rate_syl == 0;


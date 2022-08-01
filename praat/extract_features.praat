form Feature Extraction
    word in_file 
    word out_file
endform

#############
# Load file #
#############

Read from file... 'in_file$'
Rename... sound
dur = Get total duration

#########
# Pitch #
#########

select Sound sound
To Pitch... 0 75 600
f0_min = Get minimum... 0 0 Hertz Parabolic
f0_max = Get maximum... 0 0 Hertz Parabolic
f0_mean = Get mean... 0 0 Hertz
f0_std = Get standard deviation... 0 0 Hertz
f0_mas = Get mean absolute slope... Hertz
f0_pct1 = Get quantile... 0 0 0.01 Hertz
f0_pct99 = Get quantile... 0 0 0.99 Hertz
f0_q1 = Get quantile... 0 0 0.25 Hertz
f0_q2 = Get quantile... 0 0 0.5 Hertz
f0_q3 = Get quantile... 0 0 0.75 Hertz
f0_min_time = Get time of minimum... 0 0 Hertz Parabolic
f0_max_time = Get time of maximum... 0 0 Hertz Parabolic
select Pitch sound
Remove

#############
# Intensity #
#############

select Sound sound
if dur > 6.4 / 100.0
    To Intensity... 100 0 no
    int_min = Get minimum... 0 0 Parabolic
    int_max = Get maximum... 0 0 Parabolic
    int_mean = Get mean... 0 0 energy
    int_pct1 = Get quantile... 0 0 0.01
    int_pct99 = Get quantile... 0 0 0.99
    int_q1 = Get quantile... 0 0 0.25
    int_q2 = Get quantile... 0 0 0.5
    int_q3 = Get quantile... 0 0 0.75
    int_std = Get standard deviation... 0 0
    int_min_time = Get time of minimum... 0 0 Parabolic
    int_max_time = Get time of maximum... 0 0 Parabolic
endif

#######
# NHR #
#######

select Sound sound
To Pitch... 0 75 600
To PointProcess
plus Sound sound
plus Pitch sound

voice_report$ = Voice report... 0 0 75.0 600.0 1.3 1.6 0.03 0.45
nhr = extractNumber(voice_report$, "Mean noise-to-harmonics ratio: ")
select Pitch sound
Remove
select Sound sound
To Pitch... 0 75 600

###########
# Voicing #
###########

vcd_frames = Count voiced frames
tot_frames = Get number of frames
vcd2tot_frames = vcd_frames / tot_frames

####################
# Jitter / Shimmer #
####################

if vcd_frames > 0 
	select Sound sound
	plus Pitch sound

	To PointProcess (cc)
    mean_period = 1 / f0_mean
	To TextGrid (vuv)... 0.02 mean_period

	select Sound sound
	plus TextGrid sound_sound
	Extract intervals... 1 no V
	Concatenate

	select Sound chain
    dur_vcd = Get total duration
    if dur_vcd > (6.4 / 75)
        To Pitch... 0 75 600
        To PointProcess
        jitter = Get jitter (local)... 0 0 0.0001 0.02 1.3
        plus Sound chain
        shimmer = Get shimmer (local)... 0 0 0.0001 0.02 1.3 1.6
    endif
else
    select PointProcess sound
    jitter = Get jitter (local)... 0 0 0.0001 0.02 1.3
    plus Sound sound
    shimmer = Get shimmer (local)... 0 0 0.0001 0.02 1.3 1.6
endif

##########
# Output #
##########

text$ = "dur,'dur:3''newline$'"
text$ >> 'out_file$'
text$ = "f0_min,'f0_min:3''newline$'"
text$ >> 'out_file$'
text$ = "f0_max,'f0_max:3''newline$'"
text$ >> 'out_file$'
text$ = "f0_mean,'f0_mean:3''newline$'"
text$ >> 'out_file$'
text$ = "f0_std,'f0_std:3''newline$'"
text$ >> 'out_file$'
text$ = "f0_mas,'f0_mas:3''newline$'"
text$ >> 'out_file$'
text$ = "f0_min_time,'f0_min_time:3''newline$'"
text$ >> 'out_file$'
text$ = "f0_max_time,'f0_max_time:3''newline$'"
text$ >> 'out_file$'
text$ = "f0_pct1,'f0_pct1:3''newline$'"
text$ >> 'out_file$'
text$ = "f0_pct99,'f0_pct99:3''newline$'"
text$ >> 'out_file$'
text$ = "f0_q1,'f0_q1:3''newline$'"
text$ >> 'out_file$'
text$ = "f0_q2,'f0_q2:3''newline$'"
text$ >> 'out_file$'
text$ = "f0_q3,'f0_q3:3''newline$'"
text$ >> 'out_file$'
text$ = "vcd2tot_frames,'vcd2tot_frames:3''newline$'"
text$ >> 'out_file$'
text$ = "int_min,'int_min:3''newline$'"
text$ >> 'out_file$'
text$ = "int_max,'int_max:3''newline$'"
text$ >> 'out_file$'
text$ = "int_mean,'int_mean:3''newline$'"
text$ >> 'out_file$'
text$ = "int_std,'int_std:3''newline$'"
text$ >> 'out_file$'
text$ = "int_min_time,'int_min_time:3''newline$'"
text$ >> 'out_file$'
text$ = "int_max_time,'int_max_time:3''newline$'"
text$ >> 'out_file$'
text$ = "int_pct1,'int_pct1:3''newline$'"
text$ >> 'out_file$'
text$ = "int_pct99,'int_pct99:3''newline$'"
text$ >> 'out_file$'
text$ = "int_q1,'int_q1:3''newline$'"
text$ >> 'out_file$'
text$ = "int_q2,'int_q2:3''newline$'"
text$ >> 'out_file$'
text$ = "int_q3,'int_q3:3''newline$'"
text$ >> 'out_file$'
text$ = "jitter,'jitter:6''newline$'"
text$ >> 'out_file$'
text$ = "shimmer,'shimmer:6''newline$'"
text$ >> 'out_file$'
text$ = "nhr,'nhr:6''newline$'"
text$ >> 'out_file$'



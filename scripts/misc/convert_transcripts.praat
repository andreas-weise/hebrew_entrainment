form ConvertTranscripts
    sentence filename_in
    sentence filename_out
    integer tier_id
endform

textgrid = Read from file... 'filename_in$'
interval_count = Get number of intervals... tier_id

text$ = ""
text$ > 'filename_out$'

start = 0.0
end = 0.0
for i from 1 to interval_count
    end = Get starting point... tier_id i
    text$ = "'start' 'end' <silence>'newline$'"
    text$ >> 'filename_out$'

    start = end
    end = Get end point... tier_id i
    words$ = Get label of interval... tier_id i
    text$ = "'start' 'end' 'words$''newline$'"
    text$ >> 'filename_out$'
    start = end
endfor


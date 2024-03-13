function my_callback_fcn(mark,trig_sec,trig_0)

    pause on
	outp(mark, bin2dec(trig_sec));
    pause(0.01) %in seconds
    pause off
	outp(mark, bin2dec(trig_0))
    
end


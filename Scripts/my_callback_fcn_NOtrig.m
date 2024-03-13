function my_callback_fcn_NOtrig(mark,trig_sec,trig_0)

    pause on
	fprintf('\n %d %d \n',mark, bin2dec(trig_sec));
    pause(0.01) %in seconds
    pause off
	fprintf('\n %d %d \n',mark, bin2dec(trig_0));
    
end


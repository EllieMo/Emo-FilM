function [VAS_score] = Visual_Question_Answering(Question_content,wPtr,ifi,w,h)
% INPUTS:
%- Question_content: a string variable with the question content
%- wPtr is output of the 'Screen' function if there is one question it can be nested in the Visual_Question_Answering function 
% lw: a paramter of screen e.g., 5
%- ifi is the output of the Screen('GetFlipInterval',wPtr);
%-w: window size (width): [w, h] = Screen('WindowSize', wPtr); 
%-h; window size (height): [w, h] = Screen('WindowSize', wPtr);
%-screenid: selected the screen, for example, screenid = max(Screen('Screens')); 
% OUTPUT:
% - VAS_score: numerical scores of an asnwer to the question.


%% VAS Parameters 
    HSize = [0 0 1000 3]; 
    HColor = [0 0 0];
    
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    % %% full screen
%    screenid=1;
    %screenid = max(Screen('Screens'));
%     wPtr = Screen('OpenWindow', screenid, [125 125 125]);
%     
%     [w, h] = Screen('WindowSize', wPtr);
%     lw = 5;
%     ifi = Screen('GetFlipInterval',wPtr);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    wRect = [0, 0, w, h];
    [xCenter, yCenter] = RectCenter(wRect);
    
    HLine = CenterRectOnPointd(HSize, xCenter, h * 0.6); 

    VSize = [0 0 6 80]; 
    VColor = [0 0 0];
    X     = xCenter;
    
    % Set the amount we want our square to move on each button press
    pixelsPerPress = 5;
    
    % set the flag to 1 so each run the first task iteration will be VAS
    VAS_flag = 1;
    
    VAS_duration = 5; % in seconds    
    
%% Set up response buttons
    % accepted response keys: 
    leftKey = KbName(moveKey{1});
    rightKey = KbName(moveKey{2});
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if VAS_flag == 1

        start_timer = GetSecs;
        vas_timer   = 0;
        waitframes  = 1;

        % Send Trigger Task onset
        %outp(P.parportAddr,P.triggers(8));
        % wait abit
        %WaitSecs(0.05);
        % close trigger port
        %outp(P.parportAddr,0); 

        % Loop the animation until the escape key is pressed
        vbl = Screen('Flip', wPtr);
        while vas_timer < VAS_duration

            % Check the keyboard to see if a button has been pressed
            [keyIsDown,secs, keyCode] = KbCheck;

            % Depending on the button press,
            if keyCode(leftKey)
                X = X - pixelsPerPress;
            elseif keyCode(rightKey)
                X = X + pixelsPerPress;
            end

            % Boundaries
            if X < w/2 - HSize(3)/2
                X = w/2 - HSize(3)/2;
            elseif X > w/2 + HSize(3)/2
                X = w/2 + HSize(3)/2;
            end

            % Lines
            Screen('FillRect', wPtr, HColor, HLine);
            VLine = CenterRectOnPointd(VSize, X, h * 0.6);
            Screen('FillRect', wPtr, VColor, VLine);

            % Text
            DrawFormattedText(wPtr, Question_content, 'center',h * 0.3, [0 0 0]);
            Screen('TextSize', wPtr, 40); Screen('TextFont', wPtr, 'Courier New');
            DrawFormattedText(wPtr, 'No', w * 0.15 , h * 0.61, [0 0 0]);
            Screen('TextSize', wPtr, 40); Screen('TextFont', wPtr, 'Courier New');
            DrawFormattedText(wPtr, 'Yes', w * 0.85 , h * 0.61, [0 0 0]);

            % Flip to the screen
            vbl = Screen('Flip', wPtr, vbl + (waitframes - 0.5) * ifi);

            % update timer
            vas_timer = GetSecs - start_timer;

        end
%% computing the score
        % record motivation score
        VAS_score = (X-(w-HSize(3))/2)/10;

        % set VAS flag to zero so next task blocks will just be the task 
        VAS_flag = 0;
       % %% Closing the screen if needed
         % Release texture:
%          Screen('Close', wPtr);
    end

end


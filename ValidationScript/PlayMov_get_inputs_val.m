function [VAS_score] =  PlayMov_get_inputs_val(win, moveKey, S_num)
%% Function to collect responses to questions on a continuous scale

% INPUTS:
% win
% moveKey
% S_num

% OUTPUTS:
% VAS_score: Answers to the questions

%% Load Annotation Items for Subject
load('Annotation_items.mat'); % the list of emotional items
load('ValItems.mat'); % validation items- 4 or 5 per subject
Items =  ValItems(S_num,:);


[w, h] = Screen('WindowSize', win);
ifi = Screen('GetFlipInterval',win);
Screen('FillRect', win, [125, 125, 125]);

%% VAS Parameters 
HSize = [0 0 1000 10]; 
HColor = 1;
wRect = [0, 0, w, h];
[xCenter, yCenter] = RectCenter(wRect);

HLine = CenterRectOnPointd(HSize, xCenter, h * 0.6); 

VSize = [0 0 10 80]; 
VColor = 1;
X     = xCenter;

% Set the amount we want our square to move on each button press
pixelsPerPress = 5;
VAS_duration = 5; % in seconds    

%% Set up response buttons
% accepted response keys: 
leftKey = KbName(moveKey{1});
rightKey = KbName(moveKey{2});
lock = KbName(moveKey{3});

%% Set up score
VAS_score = [];
% Loop the animation until the escape key is pressed
vbl = Screen('Flip', win);   
for i = 1:length(Items)
    
    %% Get Question info
    [Q, low, high] = Selecting_Question(Items{i}, Annotation_items);
    
    start_timer = GetSecs;
    vas_timer   = 0;
    waitframes  = 1;

    % Send Trigger Task onset
    %outp(P.parportAddr,P.triggers(8));
    % wait abit
    %WaitSecs(0.05);
    % close trigger port
    %outp(P.parportAddr,0); 
    keyCode = [];
   
    while vas_timer < VAS_duration %~keyCode(lock)

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
        Screen('FillRect', win, HColor, HLine);
        VLine = CenterRectOnPointd(VSize, X, h * 0.6);
        Screen('FillRect', win, VColor, VLine);

        % Text
        DrawFormattedText(win, Q, 'center',h * 0.3, 1);
        Screen('TextSize', win, 40); Screen('TextFont', win, 'Courier New');
        DrawFormattedText(win, low, w * 0.15 , h * 0.61, 1);
        Screen('TextSize', win, 40); Screen('TextFont', win, 'Courier New');
        DrawFormattedText(win, high, w * 0.70 , h * 0.61, 1);

        % Flip to the screen
        vbl = Screen('Flip', win, vbl + (waitframes - 0.5) * ifi);

        % update timer
        vas_timer = GetSecs - start_timer;
        if keyCode(lock)
            vas_timer = VAS_duration
            pause(0.5);
        end
        
    end
    
    %% computing the score
    score = (X-(w-HSize(3))/2)/10; 
    VAS_score = [VAS_score score]; 
    
end

Screen('Close');
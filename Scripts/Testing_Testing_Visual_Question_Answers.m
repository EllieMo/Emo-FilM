% Testing_Visual_Question_Answers
clear
clc
%% Args
%screenid=1;
screenid = max(Screen('Screens'));
Question_content1='How are you?';
Question_content2='Do you like watching movies?';
wPtr = Screen('OpenWindow', screenid, [125 125 125]);
    
    [w, h] = Screen('WindowSize', wPtr);
    lw = 5;
    ifi = Screen('GetFlipInterval',wPtr);
 [VAS_score1] = Visual_Question_Answering(Question_content1,wPtr,lw,ifi,w,h,screenid)
 [VAS_score2] = Visual_Question_Answering(Question_content2,wPtr,lw,ifi,w,h,screenid)
 % shut down the screen
 Screen('Close', wPtr);
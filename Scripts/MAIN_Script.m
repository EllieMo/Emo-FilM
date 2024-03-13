%% Charting emotion using cinema
% Main script for movie display in fMRI including triggers for Biopack 
% Authors: Michal Muszynski and Ellie Morgenroth 
% Written for Matlab 2016

% clean workspace
clc
clear all
close all

% configurations 
config_keyboard;
config_sound;
config_mouse;
cgloadlib;

%% Custom Inputs
currentPath = pwd
trig = 1; % Set to 1 if you want to use the triggers
INSCANNER = 2; % 1 = out of scanner 2 = in scanner

%% Collect inputs before starting psychtoolbox
prompt_subj_ID = '\n What is the Subject ID? \n'; %Style of this should be S01 or T01
Subj_ID = input(prompt_subj_ID,'s');
S_num = str2num(Subj_ID(2:end));

prompt_subj_session = '\n Choose Session Number: 1,2,3, or 4? \n';
Subj_session = input(prompt_subj_session,'s');
        
prompt_subj_mov = '\n Choose Movie Number (1-14)? \n';
Subj_run = input(prompt_subj_mov,'s'); 
load('Movies_subjs.mat')

%% Specification for Questions
% accepted response keys
moveKey = [{'1!'}, {'2@'}];
Questions = [{'I felt absorbed by this movie'}, 
    {'I enjoyed this movie'}, 
    {'I thought this movie was interesting'}];

%% Setting up biopack and eye-tracker
if INSCANNER==2
    % address for sending markers for the different conditions (Behavioral room)
	parallelAddress.marker = 61400; % to update
	if trig
    % ----------------------- PHYSIOLOGY ------------------------ 
        % Initialize the inpout32.dll I/O driver:
        config_io;
        % Set condition code to zero:
        outp(61400, 0); %update
        % Set automatic BIOPAC and eye tracker recording to "stop":
        outp(61402, bitset(inp(61402), 3, 0)); %update
        % Close pneumatic valve:
        outp(61402, bitset(inp(61402), 4, 1)); %update
        % Using MRI scanner
        usingMRI = 1;
        if usingMRI
            P.parportAddr = hex2dec('EFD8'); %update
        %else
        %  P.parportAddr = hex2dec('378'); % update
        end
    % Local time of reseting settings  
    c_reset=clock;
    T_reset_local_time=fix(c_reset);
    end
end

%% Wait for trigger (MRI sends 5)
% wait4me = 0;
% while wait4me == 0 
%     [keyIsDown, secs, keyCode]=KbCheck;
%     rsp=KbName(keyCode);
%     if ~(isempty(rsp))
%         if rsp=='5%' 
%             wait4me=1;
%             % Local Time of beginning of the experiement
%             c_start = clock;
%             %Xstart=GetSecs;
%             T_Exp_start_local_time=fix(c_start);
%             disp('Synchronization among devices just happened!')
%         end
%     end
% end

%% Alternative triggering
T_Exp_start_local_time = KbTriggerWait(KbName('5%'));
Local = clock;

disp('Synchronization among devices just happened!')

%% Movies and Markers for movies
movies = [{'After_The_Rain'},{'Between_Viewings'},{'Big_Buck_Bunny'},{'Chatter'},{'First_Bite'},...
    {'Lesson_Learned'},{'Payload'},{'Sintel'},{'Spaceman'},{'Superhero'},{'Tears_Of_Steel'},...
    {'The_Secret_Number'},{'To_Claire_From_Sonny'},{'You_Again'},{'Rest'}];%

% Setting trigger for each movie and washout period index 15 is for rest
movie_marker_start = [{'00000100'},{'00000110'},{'00001000'},{'00001010'},{'00001100'},{'00001110'},...
    {'00010000'},{'00010010'},{'00010100'},{'00010110'},{'00011000'},{'00011010'},{'00011100'},...
    {'00011110'},{'00000010'}]; 
movie_marker_stop = [{'00100100'},{'00100110'},{'00101000'},{'00101010'},{'00101100'},{'00101110'},...
    {'00110000'},{'00110010'},{'00110100'},{'00110110'},{'00111000'},{'00111010'},{'00111100'},...
    {'00111110'},{'01111110'}];

movie_marker_0 = '00000000'; % bin2dec(movie_marker_0)=0;

%% Selecting a movie
current_movie = moves_subjs(S_num,str2num(Subj_run)); 
current_name = strrep(movies{current_movie},'_','');

%% Local time settings
T_start_triggers_local_time_Movies = [];
T_stop_triggers_local_time_Movies = [];

%% Setup screen for display of movie:
screenid = 1; %at BBL
% Comment the following line for normal settings
Screen('Preference','VisualDebugLevel', 1); % 0.018, 50, 0.1, 5);
% Open 'windowrect' sized window on screen, with black [0] background color:
windowrect = [];
win = Screen('OpenWindow', screenid, 0, windowrect);

%% Starting display of movies with rest before and after
for mov_i = [15, current_movie, 15]  % iterating over washout periods and movies
    fprintf('\n Starting movie \n')
        
    if mov_i==15 % rest stimuli- washout period
        moviename=sprintf('%s/%s',currentPath,'washout_period.mp4');
    else % one of 14 movies
        moviename=sprintf('%s/Movies_cut/%s_exp.mp4',currentPath,movies{mov_i}); % getting movie name
    end
    Marker_start=movie_marker_start{mov_i}; % get starting trigger for a movie
    Marker_stop=movie_marker_stop{mov_i};   % get stop marker
    
    % Make sure no keys are pressed
    KbReleaseWait;
    
    try  
        % Open movie file:
        movie = Screen('OpenMovie', win, moviename);

        % Start playback engine:
        Screen('PlayMovie', movie, 1);

        if trig
            % Send start trigger
            outp(parallelAddress.marker, bin2dec(Marker_start));
            % Start trigger
            Start_time_triggers = GetSecs;
            T_start_triggers_local_time_Movies = [T_start_triggers_local_time_Movies; Start_time_triggers];
        end
               
        % Playback loop: Runs until end of movie or keypress; resricted to
        % few keys, so the scanner trigger doesn't disturb the smooth runnning
        RestrictKeysForKbCheck(['N','Q','1!','2@']);
        
        while true
            % Wait for next movie frame, retrieve texture handle to it
            tex = Screen('GetMovieImage', win, movie);

            % Valid texture returned? A negative value means end of movie reached:
            if tex<=0
                break;
            end

            % Draw the new texture immediately to screen:
            Screen('DrawTexture', win, tex);

            % Update display:
            Screen('Flip', win);

            % Release texture:
            Screen('Close', tex);
   
        end
                   
        % After the stimulus set trigger to zero
        if trig
            % Send extra stop trigger
            outp(parallelAddress.marker, bin2dec(Marker_stop));
            % Stop trigger
            Stop_time_triggers = GetSecs;
            T_stop_triggers_local_time_Movies = [T_stop_triggers_local_time_Movies;Stop_time_triggers];
            % wait
            wait(1000); 
            %Put trigger to zero
            outp(parallelAddress.marker, bin2dec(movie_marker_0));
        end
        
        % Stop playback:
        Screen('PlayMovie', movie, 0);
        
        % Close movie:
        Screen('CloseMovie', movie);

    catch %#ok<CTCH>
        sca;
        psychrethrow(psychlasterror);
    end   
end

%% Ask questions at end of display
VAS_score =  PlayMov_get_inputs(win,moveKey, Questions);

% Local Time of end of the experiment
T_Exp_end_local_time = GetSecs;

% Close Screen, we're done:
Screen('CloseAll');

%% Saving in Answers folder
% Local time 
Local_Experiment_time=[Local,T_Exp_start_local_time,T_Exp_end_local_time];
Trigger_times = [T_start_triggers_local_time_Movies, T_stop_triggers_local_time_Movies]

if length(Subj_run) == 1
    mov_idx = strcat('0',Subj_run);
else
    mov_idx = Subj_run;
end

% Merging all information for saving
Subj_ID_session_run=['sub-',Subj_ID,'_ses-',Subj_session,'_task-mov',mov_idx,'_',current_name,'_psy'];
save(sprintf('Answers/%s.mat',Subj_ID_session_run),'VAS_score','Local_Experiment_time','Trigger_times'); % path to save answers

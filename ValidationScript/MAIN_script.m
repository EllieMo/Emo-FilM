%% Charting emotion using cinema
% Main script for emotion annotation validation for movie study 
% Authors: Michal Muszynski and Ellie Morgenroth 
% Written for Matlab 2016

% Clean workspace
clc
clear all
close all

% configurations 
% config_keyboard;
% config_sound;
% config_mouse;
% cgloadlib;

currentPath = pwd;
    
%% Set which Screen to use:
%screenid = 1; 
screenid = max(Screen('Screens'));

%% Collect inputs before starting psychtoolbox
prompt_subj_ID = '\n What is the Subject ID? \n'; %Style of this should be S01 or T01
Subj_ID = input(prompt_subj_ID,'s');
S_num = str2num(Subj_ID(2:end));

prompt_subj_session = '\n Choose Session Number: 1,2,3, or 4? \n';
Subj_session = input(prompt_subj_session,'s');
        
prompt_subj_mov = '\n Choose Movie Number (1-14)? \n';
Subj_run = input(prompt_subj_mov,'s'); 

%% load data
load('MP4_Excerpt_File_List_14_movies.mat'); % playlist of movie excerpts
load('Movies_subjs.mat')

pause(2); %- wait for 2000 milliseconds

%% Specification for Questions
% accepted response keys
moveKey = [{'1'}, {'2'}, {'3'}]; %key for down and up the scale
% ValidationQ =  ValItems(S_num,:);

%% Select Movies
movies = MP4_Excerpt_File_List_14_movies{:,moves_subjs(S_num,str2num(Subj_run))};

%%  iterating over movie excerpts
Merged_Answers = [];
%% Set up Screen
% Comment the following line for normal settings
%Screen('Preference','SyncTestSettings', 0.018, 50, 0.1, 5);
Screen('Preference','VisualDebugLevel', 1);      
% Open 'windowrect' sized window on screen, with black [0] background color:
win = Screen('OpenWindow', screenid, 0);

for mov_i = 1:length(movies)
    fprintf('\n Starting movie \n')
    moviename=sprintf('%s/Validation_Clips/%s',currentPath,movies{mov_i});
    Screen('FillRect', win, 0);
    % Wait until user releases keys on keyboard:
    KbReleaseWait;

    try
        movie = Screen('OpenMovie', win, moviename);

        % Start playback engine:
        Screen('PlayMovie', movie, 1);

        % Playback loop: Runs until end of movie or keypress:
        while true

            % Wait for next movie frame, retrieve texture handle to it
            tex = Screen('GetMovieImage', win, movie);

            % Valid texture returned? A negative value means end of movie reached:
            if tex<=0
                % We're done, break out of loop:
                break;
            end

            % Draw the new texture immediately to screen:
            Screen('DrawTexture', win, tex);

            % Update display:
            Screen('Flip', win);

            % Release texture:
            Screen('Close', tex);

            %count=count+1;
        end
        
        % Stop playback:
        Screen('PlayMovie', movie, 0);
        % Close movie:
        Screen('CloseMovie', movie);


    catch %#ok<CTCH>
        sca;
        psychrethrow(psychlasterror);
    end

    [scores] =  PlayMov_get_inputs_val(win, moveKey, S_num);
    Merged_Answers = [Merged_Answers; scores];
    
end

current_name = MP4_Excerpt_File_List_14_movies{:,moves_subjs(S_num,str2num(Subj_run))}{1}(1:end-10)
current_name = strrep(movies{current_name},'_','');

if length(Subj_run) == 1
    mov_idx = strcat('0',Subj_run);
else
    mov_idx = Subj_run;
end

%% Select the path to a folder where you can save results

Subj_ID_session_run=['sub-',Subj_ID,'_ses-',Subj_session,'_task-mov',mov_idx,'-',current_name,'_val'];

%save(sprintf('C:/Users/Michal/Desktop/fMRI_BBL_experiment/fMRI_BBL_experiment_matlab/testing1/Psychtoolbox_scripts/Psychtoolbox_scripts/Button_responses/Post_experiement_annot_Laura_Michal_scripts/Answers/%s_%s.mat',mov_n{1},Subj_ID_session_run),'Merged_Answers'); % path to save answers
save(sprintf('Answers/%s.mat',Subj_ID_session_run),'Merged_Answers'); % path to save answers
      
Screen('CloseAll');
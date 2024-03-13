% Image
clear;
clc;
% images{1}(100,85:145,:)  = 255;
% images{1}(150,85:145,:)  = 255;
 %images{1}(100:150,85,:)  = 255;
 %images{1}(100:150,145,:) = 255;
 backg = uint8(zeros(101,101,3));
 backg(31:71,51,:) = 255;
 backg(51,31:71,:) = 255;
 %image(backg);
 
 images{1}= backg;
 
 
  % create the video writer with 1 fps
 %  writerObj = VideoWriter('myVideo.avi');  
 writerObj = VideoWriter('washout_period','MPEG-4');
 writerObj.FrameRate = 24;
 % set the seconds per image
 duration=90;% in sec
 secsPerImage = 24*duration;
 % open the video writer
 open(writerObj);
 % write the frames to the video
 %for u=1:length(images)
     % convert the image to a frame
 frame = im2frame(images{1});
 for v=1:secsPerImage 
     writeVideo(writerObj, frame);
 end
 %end
 % close the writer object
 close(writerObj);
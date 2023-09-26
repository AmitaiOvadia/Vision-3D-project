path_to_dir = 'C:\Users\amita\OneDrive\Desktop\master\vision 3D\BallSpeed\validation throw\throw 2';
vid_path = "C:\Users\amita\OneDrive\Desktop\master\vision 3D\BallSpeed\validation throw\throw 2.mp4";
% load('C:\Users\amita\OneDrive\Desktop\master\vision 3D\BallSpeed\ball videos 4\calibration video\cameraPrams4.mat')
vidObj = VideoReader(vid_path);
nFrames = vidObj.NumberOfFrames;
vidHeight = vidObj.Height;
vidWidth = vidObj.Width;
%%
for i = 1:1:nFrames
    i
    frame = read(vidObj,i);
    undistortedImage = undistortImage(frame, cameraParamsVal);
    start = 'frame_';
    if i < 100
        start = [start '0'];
    end
    path = [path_to_dir '\' start num2str(i) '.png'];
    imwrite(undistortedImage,path);
end


% Auto-generated by cameraCalibrator app on 01-Sep-2023
%-------------------------------------------------------


% Define images to process
imageFileNames = {'C:\Users\amita\OneDrive\Desktop\master\vision 3D\BallSpeed\Ball videos 3\calibration frames\frame_200.png',...
    'C:\Users\amita\OneDrive\Desktop\master\vision 3D\BallSpeed\Ball videos 3\calibration frames\frame_400.png',...
    'C:\Users\amita\OneDrive\Desktop\master\vision 3D\BallSpeed\Ball videos 3\calibration frames\frame_600.png',...
    'C:\Users\amita\OneDrive\Desktop\master\vision 3D\BallSpeed\Ball videos 3\calibration frames\frame_1000.png',...
    'C:\Users\amita\OneDrive\Desktop\master\vision 3D\BallSpeed\Ball videos 3\calibration frames\frame_1200.png',...
    'C:\Users\amita\OneDrive\Desktop\master\vision 3D\BallSpeed\Ball videos 3\calibration frames\frame_1400.png',...
    'C:\Users\amita\OneDrive\Desktop\master\vision 3D\BallSpeed\Ball videos 3\calibration frames\frame_1600.png',...
    'C:\Users\amita\OneDrive\Desktop\master\vision 3D\BallSpeed\Ball videos 3\calibration frames\frame_1800.png',...
    'C:\Users\amita\OneDrive\Desktop\master\vision 3D\BallSpeed\Ball videos 3\calibration frames\frame_2000.png',...
    'C:\Users\amita\OneDrive\Desktop\master\vision 3D\BallSpeed\Ball videos 3\calibration frames\frame_2200.png',...
    'C:\Users\amita\OneDrive\Desktop\master\vision 3D\BallSpeed\Ball videos 3\calibration frames\frame_2400.png',...
    'C:\Users\amita\OneDrive\Desktop\master\vision 3D\BallSpeed\Ball videos 3\calibration frames\frame_2800.png',...
    'C:\Users\amita\OneDrive\Desktop\master\vision 3D\BallSpeed\Ball videos 3\calibration frames\frame_3000.png',...
    'C:\Users\amita\OneDrive\Desktop\master\vision 3D\BallSpeed\Ball videos 3\calibration frames\frame_3200.png',...
    };
% Detect calibration pattern in images
detector = vision.calibration.monocular.CheckerboardDetector();
[imagePoints, imagesUsed] = detectPatternPoints(detector, imageFileNames);
imageFileNames = imageFileNames(imagesUsed);

% Read the first image to obtain image size
originalImage = imread(imageFileNames{1});
[mrows, ncols, ~] = size(originalImage);

% Generate world coordinates for the planar pattern keypoints
squareSize = 15;  % in units of 'millimeters'
worldPoints = generateWorldPoints(detector, 'SquareSize', squareSize);

% Calibrate the camera
[cameraParams, imagesUsed, estimationErrors] = estimateCameraParameters(imagePoints, worldPoints, ...
    'EstimateSkew', false, 'EstimateTangentialDistortion', false, ...
    'NumRadialDistortionCoefficients', 2, 'WorldUnits', 'millimeters', ...
    'InitialIntrinsicMatrix', [], 'InitialRadialDistortion', [], ...
    'ImageSize', [mrows, ncols]);

% View reprojection errors
h1=figure; showReprojectionErrors(cameraParams);

% Visualize pattern locations
h2=figure; showExtrinsics(cameraParams, 'CameraCentric');

% Display parameter estimation errors
displayErrors(estimationErrors, cameraParams);

% For example, you can use the calibration data to remove effects of lens distortion.
undistortedImage = undistortImage(originalImage, cameraParams);

%% create a 3x4 camera matrix from intrinsics

K = cameraParams.Intrinsics.K;
P = K * [eye(3), zeros(3,1)];
Pinv = pinv(P);





%% find 3x4 camera matrix
% [N, ~, m] = size(imagePoints);
% imagePoints_N = reshape(imagePoints, [N*m, 2]);

imagePoints_1 = squeeze(imagePoints(:, :, 1)); 
camExtrinsics = estimateExtrinsics(imagePoints_1,worldPoints,cameraParams.Intrinsics);
P_new = cameraProjection(cameraParams.Intrinsics,camExtrinsics);
P_inv_new = pinv(P_new);
save('P.mat', 'P_new');
save('P_inv.mat', 'P_inv_new');

% %% find 3x4 camera matrix
% K = cameraParams.IntrinsicMatrix';
% R = cameraParams.RotationMatrices(:,:,1);
% t = cameraParams.TranslationVectors(1,:)';
% P = K * [R t];
% % save('P.mat', 'P');
% %% get the inverse
% P_plus = pinv(P);
% save('P_plus.mat', 'P_plus');
% Q = [P_plus, zeros(4,1)]';

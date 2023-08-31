# Import the necessary modules
import cv2  # For video processing
import os  # For creating directories and paths

# Define the video path, the output directory and the frame interval
video_path = "ball frames video 2/Ball video 2 SM.mp4"  # The path of the mp4 video file
output_dir = "ball frames video 2"  # The name of the output directory
d = 2  # The number of frames to skip between each saved frame

# Create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the video using cv2.VideoCapture
video = cv2.VideoCapture(video_path)

# Check if the video is opened successfully
if not video.isOpened():
    print("Error: Could not open the video file")
    exit()

# Initialize a counter for the frame number
frame_count = 0

# Loop through the frames of the video
while True:
    # Read a frame from the video
    success, frame = video.read()

    # If the frame is not read successfully, break the loop
    if not success:
        break

    # If the frame number is divisible by d, save the frame as a png image
    if frame_count % d == 0:
        # Construct the output file name using the frame number
        start = "frame_"
        # if frame_count < 100:
        #     start = "frame_0"
        #     if frame_count < 10:
        #         start = "frame_00"
        output_file = os.path.join(output_dir, f"{start}{frame_count}.png")
        # Save the frame as a png image using cv2.imwrite
        cv2.imwrite(output_file, frame)

    # Increment the frame number
    frame_count += 1

# Release the video object
video.release()
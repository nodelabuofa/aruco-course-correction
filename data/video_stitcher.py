#!/usr/bin/env python3

import rosbag
from sensor_msgs.msg import Image
import cv2
import os
from cv_bridge import CvBridge
import glob

# finds absolute filepath to this python script
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- Find the most recent directory two levels deep ---
# Get a list of all folders alongside this python script
subdirectories_level1 = [d for d in os.listdir(script_dir) if os.path.isdir(os.path.join(script_dir, d))]

if not subdirectories_level1:
    raise FileNotFoundError(f"No subdirectories found in the script directory: {script_dir}")

# Find the most recently made folder in the first level
most_recent_dir_level1 = max([os.path.join(script_dir, d) for d in subdirectories_level1], key=os.path.getctime)

# Get a list of all folders inside the most recent first-level directory
subdirectories_level2 = [d for d in os.listdir(most_recent_dir_level1) if os.path.isdir(os.path.join(most_recent_dir_level1, d))]

if not subdirectories_level2:
    raise FileNotFoundError(f"No subdirectories found in the most recent first-level directory: {most_recent_dir_level1}")

# Find the most recently made folder in the second level
most_recent_dir_level2 = max([os.path.join(most_recent_dir_level1, d) for d in subdirectories_level2], key=os.path.getctime)


# finds the .bag file inside the second-level directory
bag_files = glob.glob(os.path.join(most_recent_dir_level2, '*.bag'))

if not bag_files:
    raise FileNotFoundError(f"No .bag files found in the most recent directory (2 levels deep): {most_recent_dir_level2}")

# Use the first .bag file found
bag_file = bag_files[0]
print(f"Processing ROSbag file: {bag_file}")

image_topic = '/annotated_image_topic'  # <-- CHANGE THIS if needed

# --- Automatically determine the output video file name ---
# Get the base name of the bag file (without extension)
base_name = os.path.splitext(os.path.basename(bag_file))[0]
# Create the output video file name with .mp4 extension in the same directory as the bag file
output_video_file = os.path.join(os.path.dirname(bag_file), f"{base_name}.mp4")
print(f"Output video file will be: {output_video_file}")


# Set the desired frame rate for the output video
FRAME_RATE = 30.0
# -------------------

# Instantiate CvBridge
bridge = CvBridge()
video = None

try:
    # Open the bag file for reading
    with rosbag.Bag(bag_file, 'r') as bag:
        print(f"Reading messages from topic '{image_topic}' in bag file '{bag_file}'...")
        
        # Iterate through messages on the specified topic
        for topic, msg, t in bag.read_messages(topics=[image_topic]):
            try:
                # Convert the ROS Image message to an OpenCV image (NumPy array)
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            except Exception as e:
                print(f"Warning: Could not convert image message. Error: {e}")
                continue

            # --- Initialize the VideoWriter on the first valid frame ---
            if video is None:
                # Get the frame dimensions from the first image
                height, width, _ = cv_image.shape
                
                # Define the codec and create VideoWriter object
                # 'mp4v' is a widely compatible codec for .mp4 files
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter(output_video_file, fourcc, FRAME_RATE, (width, height))
                print(f"Video dimensions set to {width}x{height} at {FRAME_RATE} fps.")
                print(f"Writing to '{output_video_file}'...")

            # Write the current frame to the video file
            video.write(cv_image)

    if video is not None:
        print("Video creation complete.")
    else:
        print(f"No images found on topic '{image_topic}'. No video was created.")

finally:
    # --- Release the VideoWriter object ---
    if video is not None:
        video.release()
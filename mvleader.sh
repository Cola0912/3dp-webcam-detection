#!/bin/bash

# Define paths
BASE_PATH=~/3dp-webcam-detection
OBSERVER_SCRIPT=$BASE_PATH/mvobserver.py
DETECTOR_SCRIPT=$BASE_PATH/detector.py
REPORTER_SCRIPT=$BASE_PATH/mvreporter.py
INPUT_VIDEO_PATH=$BASE_PATH/input_video.mp4
OUTPUT_VIDEO_PATH=$BASE_PATH/output_video.mp4

# Run observer.py to break down the video into frames
python3 $OBSERVER_SCRIPT $INPUT_VIDEO_PATH

# Assuming observer.py saves frames as temp_frame_*.jpg
# And assuming it saves the number of frames in a file named frame_count.txt
FRAME_COUNT=$(cat $BASE_PATH/frame_count.txt)

# Process each frame using detector.py
for (( i=0; i<FRAME_COUNT; i++ ))
do
    INPUT_FRAME=$BASE_PATH/temp_frame_${i}.jpg
    OUTPUT_FRAME=$BASE_PATH/processed_frame_${i}.jpg
    python3 $DETECTOR_SCRIPT $INPUT_FRAME $OUTPUT_FRAME
done

# Reconstruct the video from processed frames using reporter.py
python3 $REPORTER_SCRIPT $OUTPUT_VIDEO_PATH

# Optional: Clean up temporary frames
rm $BASE_PATH/temp_frame_*.jpg
rm $BASE_PATH/processed_frame_*.jpg

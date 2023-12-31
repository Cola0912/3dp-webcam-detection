#!/bin/bash

# Define paths
BASE_PATH=~/3dp-webcam-detection
OBSERVER_SCRIPT=$BASE_PATH/camobserver.py
DETECTOR_SCRIPT=$BASE_PATH/detector.py
BAIT_IMAGE=$BASE_PATH/detector_bait.jpg
POOP_IMAGE=$BASE_PATH/detector_poop.jpg

# Run observer in the background

# Infinite loop to process images continuously
while true
do
    # Run observer to capture an image from camera
    python3 $OBSERVER_SCRIPT

    # Check if bait image exists
    if [ -f "$BAIT_IMAGE" ]; then
        # Run detector with the bait image and output to poop image
        python3 $DETECTOR_SCRIPT $BAIT_IMAGE $POOP_IMAGE
    fi
    
    # Wait for 10 seconds before capturing the next image
    sleep 5
done
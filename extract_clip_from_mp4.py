

import json
import os
import cv2
import re
from pathlib import Path
import subprocess

def extract_clip_with_ffmpeg(video_path, start_time, output_path, stop_time):
    """
    Extract a clip with audio using ffmpeg.
    """
    try:
        # Convert to seconds if needed
        if isinstance(start_time, str) and ':' in start_time:
            start_time = time_to_seconds(start_time)
        if isinstance(stop_time, str) and ':' in stop_time:
            stop_time = time_to_seconds(stop_time)
        
        # Calculate duration
        duration = stop_time - start_time
        
        print(f"Extracting from {start_time}s to {stop_time}s (duration: {duration}s)")
        
        # Build ffmpeg command - KEY CHANGE: -ss AFTER -i
        cmd = [
            'ffmpeg',
            '-i', video_path,        # Input file FIRST
            '-ss', str(start_time),  # Start time (now works correctly)
            '-t', str(duration),     # Use -t (duration) instead of -to
            '-c:v', 'libx264',       # Video codec
            '-c:a', 'aac',           # Audio codec
            '-preset', 'medium',     # Encoding speed
            '-crf', '23',            # Quality
            '-movflags', '+faststart',  # For web streaming
            '-y',                    # Overwrite output
            output_path
        ]
        
        print(f"Running: {' '.join(cmd)}")
        
        # Run ffmpeg
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ Extracted clip with audio to {output_path}")
            return True
        else:
            print(f"❌ FFmpeg error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error extracting clip: {e}")
        return False

# Helper function
def time_to_seconds(time_str):
    """Convert HH:MM:SS to seconds."""
    if isinstance(time_str, (int, float)):
        return time_str
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + int(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + int(s)
    else:
        return int(parts[0])

# Now this will work
extract_clip_with_ffmpeg(
    "MARGENTO_GraphPoem_DHSI_2023.mp4", 
    3962,  # 01:06:02
    "comm_singularity_graphpoem_dhsi_uvic.mp4",
    4422   # 01:13:42
)


# extract_clip_with_ffmpeg(
#   "MARGENTO_GraphPoem_DHSI_2020.mp4", 
#    "MARGENTO_GraphPoem_DHSI_2023.mp4",
#    "graphpoem_dhsi_uvic.mp4",
#    "00:00:10", 
#    "01:06:02",
#    "comm_graphpoem_dhsi_uvic.mp4", 
#    "00:02:37"
#    "01:13:42"
#)

# extract_clip_with_ffmpeg(
#    "MARGENTO_GraphPoem_DHSI_2023.mp4", 
#    "01:06:02", 
#    "comm_graphpoem_dhsi_uvic.mp4",
#    "01:13:42"
# )
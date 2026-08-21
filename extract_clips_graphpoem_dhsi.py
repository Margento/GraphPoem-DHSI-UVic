import json
import os
import cv2
import re
from pathlib import Path

def find_mp4_file(dhsi_year, current_folder):
    """
    Find an MP4 file in the current folder that matches the DHSI year.
    """
    # Map year to folder name pattern
    year_patterns = {
        '2021': 'DHSI_2021',
        '2022': 'DHSI_2022',
        '2023': 'DHSI_2023',
        '2024': 'DHSI_2024',
    }
    
    pattern = year_patterns.get(dhsi_year)
    if not pattern:
        print(f"⚠ No pattern found for year: {dhsi_year}")
        return None
    
    # Search for MP4 files in current directory
    for file in os.listdir(current_folder):
        if file.endswith('.mp4') and pattern in file:
            return os.path.join(current_folder, file)
    
    print(f"⚠ No MP4 file found for {pattern}")
    return None

def extract_dhsi_year_from_path(image_path):
    """
    Extract DHSI year from the image path.
    Example: "graphpoem_dhsi22_extracted_shots/shot_1168_time_7596s.jpg" -> "2022"
    """
    # Look for dhsi22, dhsi21, dhsi23, etc.
    match = re.search(r'dhsi(\d{2})', image_path, re.IGNORECASE)
    if match:
        year_code = match.group(1)
        # Convert "22" to "2022"
        return f"20{year_code}"
    return None

def extract_time_from_path(image_path):
    """
    Extract time in seconds from the image path.
    Example: "shot_1168_time_7596s.jpg" -> 7596
    """
    match = re.search(r'time_(\d+)s', image_path)
    if match:
        return int(match.group(1))
    return None

import json
import os
import re
import subprocess
from pathlib import Path

def extract_clip_with_ffmpeg(video_path, start_time, output_path, duration=10):
    """
    Extract a clip with audio using ffmpeg.
    """
    try:
        # Build ffmpeg command
        cmd = [
            'ffmpeg',
            '-ss', str(start_time),  # Start time
            '-i', video_path,        # Input file
            '-t', str(duration),     # Duration
            '-c:v', 'libx264',       # Video codec
            '-c:a', 'aac',           # Audio codec
            '-preset', 'medium',     # Encoding speed
            '-crf', '23',            # Quality
            '-movflags', '+faststart',  # For web streaming
            '-y',                    # Overwrite output
            output_path
        ]
        
        # Run ffmpeg
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ Extracted {duration}s with audio to {output_path}")
            return True
        else:
            print(f"❌ FFmpeg error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error extracting clip: {e}")
        return False


def process_top_json_entries(json_file_path, top_n=10):
    """
    Process the first N entries from a JSON file and extract video clips.
    """
    # Load JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Create output directory
    # output_dir = "after_that_graphpoem_dhsi_extracted_clips"
    # output_dir = "place_flagey_graphpoem_dhsi_extracted_clips"
    output_dir = "dhsi_collage_graphpoem_dhsi_extracted_clips"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get current folder (where the script is running)
    current_folder = os.getcwd()
    
    # Process first N entries
    successful = 0
    failed = 0
    
    for i, entry in enumerate(data[:top_n]):
        print("\n" + "=" * 60)
        print(f"Processing entry {i+1}/{min(top_n, len(data))}")
        print("=" * 60)
        
        # Get image path
        image_path = entry.get('image_path')
        if not image_path:
            print(f"⚠ No image_path found for entry {i}")
            failed += 1
            continue
        
        # Extract DHSI year
        year = extract_dhsi_year_from_path(image_path)
        if not year:
            print(f"⚠ Could not extract year from: {image_path}")
            failed += 1
            continue
        
        # Extract time
        start_time = extract_time_from_path(image_path)
        if not start_time:
            print(f"⚠ Could not extract time from: {image_path}")
            failed += 1
            continue
        
        # Get image name for output
        image_name = entry.get('image_name', '')
        if not image_name:
            # Fallback: extract from image_path
            image_name = os.path.basename(image_path)
        
        # Remove .jpg extension for output name
        output_base_name = os.path.splitext(image_name)[0]
        output_path = os.path.join(output_dir, f"{output_base_name}.mp4")
        
        print(f"Image: {image_path}")
        print(f"Year: {year}")
        print(f"Start time: {start_time}s")
        print(f"Output: {output_path}")
        
        # Find the MP4 file
        mp4_path = find_mp4_file(year, current_folder)
        if not mp4_path:
            print(f"❌ No MP4 file found for year {year}")
            failed += 1
            continue
        
        print(f"Found MP4: {mp4_path}")
        
        # Extract the clip
        success = extract_clip_with_ffmpeg(mp4_path, start_time, output_path, duration=10)
        if success:
            successful += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully extracted: {successful} clips")
    print(f"❌ Failed: {failed} clips")
    print(f"📁 Output directory: {output_dir}")

# ============================================================
# RUN THE SCRIPT
# ============================================================

if __name__ == "__main__":
    # Path to your JSON file
    # JSON_FILE = "after_that_sorted_images_by_affect.json"
    # JSON_FILE = "place_flagey_sorted_images_by_affect.json"
    JSON_FILE = "dhsi_collage_sorted_images_by_affect.json"
    # Process the first 10 entries
    process_top_json_entries(JSON_FILE, top_n=10)

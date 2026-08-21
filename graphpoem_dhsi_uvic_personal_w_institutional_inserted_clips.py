import os
import random
import subprocess
from pathlib import Path
import numpy as np

def get_video_duration(video_path):
    """Get duration of video in seconds."""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except:
        pass
    return None

def parse_timestamp_from_name(shot_name):
    """Extract timestamp from shot name."""
    import re
    match = re.search(r'time_(\d+)s', shot_name)
    if match:
        return int(match.group(1))
    return None

def generate_well_spaced_timestamps(n, video_duration, clip_duration=10, min_gap=5, seed=None):
    """Generate well-spaced timestamps throughout the video."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Leave margin at start and end
    margin = 5
    available = video_duration - margin * 2 - (n * clip_duration)
    
    if available < 0:
        clip_duration = (video_duration - margin * 2) / n
        if clip_duration < 2:
            print(f"❌ Video too short ({video_duration}s) for {n} clips")
            return [], 0
        print(f"⚠ Adjusted clip duration to {clip_duration:.2f}s")
        available = video_duration - margin * 2 - (n * clip_duration)
    
    # Generate gaps with Pareto distribution
    gaps = []
    for i in range(n - 1):
        gap = np.random.pareto(1.5) * 5 + min_gap
        gaps.append(gap)
    
    total_gaps = sum(gaps) if gaps else 0
    if total_gaps > 0 and available > 0:
        scale_factor = available / total_gaps
        gaps = [g * scale_factor for g in gaps]
    
    # Build timestamps
    timestamps = []
    current_time = margin
    
    for i in range(n):
        timestamps.append(current_time)
        if i < n - 1:
            current_time += clip_duration + gaps[i]
    
    # Add small random jitter
    for i in range(len(timestamps)):
        if i > 0 and i < len(timestamps) - 1:
            jitter = random.uniform(-1, 1)
            min_allowed = timestamps[i-1] + clip_duration + 1
            max_allowed = timestamps[i+1] - clip_duration - 1 if i < len(timestamps) - 1 else video_duration - clip_duration
            if min_allowed < max_allowed:
                timestamps[i] = min(max(timestamps[i] + jitter, min_allowed), max_allowed)
    
    return timestamps, clip_duration

def insert_clips_with_moviepy_debug(video_path, clip_folder, clip_order, timestamps, output_path):
    """
    Insert clips using moviepy with debugging steps.
    """
    try:
        from moviepy.editor import VideoFileClip, CompositeVideoClip
        
        print("="*60)
        print("Step 1: Loading original video...")
        print("="*60)
        original = VideoFileClip(video_path)
        video_duration = original.duration
        print(f"Original video: {video_duration:.2f}s, size: {original.size}")
        
        # Load clips
        print("\n" + "="*60)
        print("Step 2: Loading clips...")
        print("="*60)
        clips_to_overlay = []
        
        for i, (clip_name, timestamp) in enumerate(zip(clip_order, timestamps)):
            clip_path = os.path.join(clip_folder, clip_name)
            if not os.path.exists(clip_path):
                print(f"⚠ Clip not found: {clip_path}")
                continue
            
            print(f"\nClip {i+1}: {clip_name}")
            print(f"  Timestamp: {timestamp:.2f}s")
            
            clip = VideoFileClip(clip_path)
            print(f"  Duration: {clip.duration:.2f}s")
            print(f"  Size: {clip.size}")
            
            # Check if clip fits in video
            if timestamp + clip.duration > video_duration:
                new_duration = video_duration - timestamp - 0.5
                if new_duration < 1:
                    print(f"  ⚠ Clip too long for remaining video, skipping...")
                    clip.close()
                    continue
                print(f"  Truncating to {new_duration:.2f}s")
                clip = clip.subclip(0, new_duration)
            
            # Set the clip to start at the correct time
            clip = clip.set_start(timestamp)
            
            # Optional: Resize clip to fit if needed
            # if clip.size[0] > original.size[0] or clip.size[1] > original.size[1]:
            #     clip = clip.resize(width=original.size[0]//2)
            
            clips_to_overlay.append(clip)
            print(f"  ✅ Loaded and positioned at {timestamp:.2f}s")
        
        if not clips_to_overlay:
            print("❌ No clips to overlay")
            original.close()
            return False
        
        print("\n" + "="*60)
        print(f"Step 3: Creating composite with {len(clips_to_overlay)} clips...")
        print("="*60)
        
        # Create composite video (original + overlays)
        final = CompositeVideoClip([original] + clips_to_overlay)
        
        print(f"Composite duration: {final.duration:.2f}s")
        
        print("\n" + "="*60)
        print("Step 4: Writing output video...")
        print("="*60)
        print(f"Output: {output_path}")
        print("This may take a few minutes...")
        
        # Write output with good quality
        final.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            fps=24,
            threads=4,
            preset='medium',
            ffmpeg_params=['-crf', '23'],
            verbose=False,
            logger=None
        )
        
        # Clean up
        original.close()
        for clip in clips_to_overlay:
            clip.close()
        final.close()
        
        print(f"\n✅ Successfully inserted clips into {output_path}")
        
        # Verify output
        output_duration = get_video_duration(output_path)
        print(f"Output video duration: {output_duration:.2f}s")
        if output_duration < video_duration - 1:
            print(f"⚠ Warning: Output is shorter than original ({video_duration:.2f}s)")
        
        return True
        
    except ImportError:
        print("❌ moviepy not installed. Run: pip install moviepy")
        return False
    except Exception as e:
        print(f"❌ Error with moviepy: {e}")
        import traceback
        traceback.print_exc()
        return False

def insert_clips_with_ffmpeg_overlay(video_path, clip_folder, clip_order, timestamps, output_path):
    """
    Alternative: Use ffmpeg with overlay filter (simpler, one clip at a time).
    """
    import tempfile
    
    print("\n" + "="*60)
    print("Using FFmpeg overlay approach...")
    print("="*60)
    
    # Create a copy of the original video
    temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    temp_video.close()
    
    # Copy original to temp
    copy_cmd = [
        'ffmpeg',
        '-i', video_path,
        '-c', 'copy',
        '-y',
        temp_video.name
    ]
    subprocess.run(copy_cmd, capture_output=True)
    
    current_video = temp_video.name
    
    # Process each clip
    for i, (clip_name, timestamp) in enumerate(zip(clip_order, timestamps)):
        clip_path = os.path.join(clip_folder, clip_name)
        if not os.path.exists(clip_path):
            print(f"⚠ Clip not found: {clip_path}")
            continue
        
        print(f"\nProcessing clip {i+1}/{len(clip_order)}: {clip_name} at {timestamp:.2f}s")
        
        # Get clip duration
        duration = get_video_duration(clip_path)
        if duration is None:
            continue
        
        # Create a new temp file for the overlay
        output_temp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        output_temp.close()
        
        # Overlay clip onto current video
        overlay_cmd = [
            'ffmpeg',
            '-i', current_video,
            '-i', clip_path,
            '-filter_complex',
            f"[0:v][1:v]overlay=0:0:enable='between(t,{timestamp},{timestamp+duration})'[outv];[0:a][1:a]amix=inputs=2:duration=longest[aout]",
            '-map', '[outv]',
            '-map', '[aout]',
            '-c:v', 'libx264',
            '-c:a', 'aac',
            '-preset', 'medium',
            '-crf', '23',
            '-movflags', '+faststart',
            '-y',
            output_temp.name
        ]
        
        result = subprocess.run(overlay_cmd, capture_output=True)
        
        if result.returncode != 0:
            print(f"❌ FFmpeg error: {result.stderr}")
            continue
        
        # Clean up previous video and set current to new video
        os.unlink(current_video)
        current_video = output_temp.name
        
        print(f"  ✅ Overlaid at {timestamp:.2f}s")
    
    # Move final video to output path
    os.rename(current_video, output_path)
    print(f"\n✅ Successfully inserted clips into {output_path}")
    return True

# ============================================================
# MAIN SCRIPT
# ============================================================

def main():
    # Your data
    clip_order = [
        "shot_0502_time_3809s.mp4",
        "shot_0665_time_5628s.mp4",
        "shot_0441_time_4099s.mp4",
        "shot_0222_time_1590s.mp4",
        "shot_0788_time_6806s.mp4",
        "shot_0568_time_4339s.mp4",
        "shot_0371_time_3458s.mp4",
        "shot_0635_time_4843s.mp4",
        "shot_1168_time_7596s.mp4",
        "shot_0682_time_5321s.mp4"
    ]
    
    # Video paths
    video_path = "costin_graphpoem_dhsi.mp4"
    clip_folder = "place_flagey_graphpoem_dhsi_extracted_clips"  # Folder containing clips
    output_path = "costin_graphpoem_dhsi_with_insertions.mp4"
    
    
    # Get video duration
    video_duration = get_video_duration(video_path)
    if video_duration is None:
        print("❌ Could not get video duration")
        return
    
    print(f"Video duration: {video_duration:.2f}s ({video_duration/60:.2f} minutes)")
    print(f"Number of clips: {len(clip_order)}")
    
    # Get clip duration
    first_clip = os.path.join(clip_folder, clip_order[0])
    if os.path.exists(first_clip):
        clip_duration = get_video_duration(first_clip) or 10
    else:
        clip_duration = 10
    print(f"Clip duration: {clip_duration:.2f}s")
    
    # Generate well-spaced timestamps
    timestamps, clip_duration = generate_well_spaced_timestamps(
        n=len(clip_order),
        video_duration=video_duration,
        clip_duration=clip_duration,
        min_gap=3,
        seed=42
    )
    
    if not timestamps:
        return
    
    # Show timestamps
    print("\nGenerated timestamps:")
    for i, (clip, ts) in enumerate(zip(clip_order, timestamps)):
        original = parse_timestamp_from_name(clip)
        print(f"  {i+1}. {clip} → Original: {original}s, New: {ts:.2f}s")
    
    print(f"\nTimestamps range: {min(timestamps):.2f}s to {max(timestamps):.2f}s")
    print(f"Total clips duration: {len(timestamps) * clip_duration:.2f}s")
    
    # Try moviepy with debugging first
    print("\n" + "="*60)
    print("Attempting moviepy approach with debugging...")
    print("="*60)
    
    success = insert_clips_with_moviepy_debug(video_path, clip_folder, clip_order, timestamps, output_path)
    
    # If moviepy fails or produces bad output, try FFmpeg overlay
    if not success:
        print("\n" + "="*60)
        print("Falling back to FFmpeg overlay approach...")
        print("="*60)
        success = insert_clips_with_ffmpeg_overlay(video_path, clip_folder, clip_order, timestamps, output_path)
    
    if success:
        # Verify the output
        output_duration = get_video_duration(output_path)
        print(f"\n✅ Final video saved as: {output_path}")
        print(f"📹 Output duration: {output_duration:.2f}s")
        if output_duration < video_duration * 0.8:
            print(f"⚠ Warning: Output is significantly shorter than original ({video_duration:.2f}s)")
            print("This might indicate a problem with the video encoding.")
    else:
        print("\n❌ Insertion failed. Check your video and clip paths.")

if __name__ == "__main__":
    main()




# 03:07, 04:57, 05:50, 06:04
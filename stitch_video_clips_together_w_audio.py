
import subprocess
import os

def stitch_clips_ffmpeg(clip_paths, output_path):
    """
    Stitch multiple video clips together using ffmpeg.
    """
    # Create a temporary file with the list of clips
    list_file = "clip_list.txt"
    with open(list_file, 'w') as f:
        for clip_path in clip_paths:
            # ffmpeg needs this format for the concat demuxer
            f.write(f"file '{os.path.abspath(clip_path)}'\n")
    
    try:
        # Build ffmpeg command
        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', list_file,
            '-c', 'copy',  # This preserves quality
            '-y',  # Overwrite output
            output_path
        ]
        
        # Run ffmpeg
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Clean up temp file
        os.remove(list_file)
        
        if result.returncode == 0:
            print(f"✅ Stitched {len(clip_paths)} clips into {output_path}")
            return True
        else:
            print(f"❌ FFmpeg error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error stitching clips: {e}")
        if os.path.exists(list_file):
            os.remove(list_file)
        return False


clip_paths = [
    "graphpoem_dhsi_uvic_0.mp4",
    "graphpoem_dhsi_uvic_01.mp4"
]

output_path = "graphpoem_dhsi_uvic_1.mp4"

stitch_clips_ffmpeg(clip_paths, output_path)

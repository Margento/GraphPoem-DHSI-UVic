import os
import random
import subprocess
from pathlib import Path

import numpy as np


# ============================================================
# VIDEO / AUDIO HELPERS
# ============================================================

def get_video_duration(video_path):
    """Get the duration of a video/audio file in seconds."""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())

    except Exception as e:
        print(f"⚠ Could not get duration for {video_path}: {e}")

    return None


def has_audio_stream(video_path):
    """Check whether a video file contains an audio stream."""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=codec_type",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )

        return (
            result.returncode == 0
            and "audio" in result.stdout.lower()
        )

    except Exception:
        return False


def parse_timestamp_from_name(shot_name):
    """Extract timestamp from a filename such as time_3809s."""
    import re

    match = re.search(r"time_(\d+)s", shot_name)

    if match:
        return int(match.group(1))

    return None


# ============================================================
# TIMESTAMP GENERATION
# ============================================================

def generate_well_spaced_timestamps(
    n,
    video_duration,
    clip_duration=10,
    min_gap=3,
    seed=None
):
    """Generate well-spaced timestamps throughout the video."""

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Leave some room at the beginning and end
    margin = 5

    available = (
        video_duration
        - margin * 2
        - (n * clip_duration)
    )

    # If the requested clips do not fit,
    # reduce the assumed clip duration.
    if available < 0:

        clip_duration = (
            video_duration - margin * 2
        ) / n

        if clip_duration < 2:
            print(
                f"❌ Video too short "
                f"({video_duration:.2f}s) "
                f"for {n} clips"
            )
            return [], 0

        print(
            f"⚠ Adjusted assumed clip duration "
            f"to {clip_duration:.2f}s"
        )

        available = (
            video_duration
            - margin * 2
            - (n * clip_duration)
        )

    # Generate variable gaps
    gaps = []

    for _ in range(n - 1):

        gap = (
            np.random.pareto(1.5) * 3
            + min_gap
        )

        gaps.append(gap)

    total_gaps = sum(gaps) if gaps else 0

    # Scale gaps so everything fits
    if total_gaps > 0 and available > 0:

        scale_factor = available / total_gaps

        gaps = [
            gap * scale_factor
            for gap in gaps
        ]

    # Build timestamps
    timestamps = []

    current_time = margin

    for i in range(n):

        timestamps.append(current_time)

        if i < n - 1:
            current_time += (
                clip_duration
                + gaps[i]
            )

    # Add a small amount of jitter
    # while preventing overlap
    for i in range(len(timestamps)):

        if i > 0 and i < len(timestamps) - 1:

            jitter = random.uniform(-1, 1)

            min_allowed = (
                timestamps[i - 1]
                + clip_duration
                + 0.5
            )

            max_allowed = (
                timestamps[i + 1]
                - clip_duration
                - 0.5
            )

            if min_allowed < max_allowed:

                timestamps[i] = min(
                    max(
                        timestamps[i] + jitter,
                        min_allowed
                    ),
                    max_allowed
                )

    return timestamps, clip_duration


# ============================================================
# MAIN INSERTION FUNCTION
# ============================================================

def insert_clips_with_audio_correct(
    video_path,
    clip_folder,
    clip_order,
    timestamps,
    output_path
):
    """
    Insert clips into the original video.

    Each inserted clip:
      - appears at its assigned timestamp
      - plays normally for its full duration
      - replaces the original audio during that duration

    The original video and audio resume after each clip.
    """

    print("\n" + "=" * 60)
    print("Using FFmpeg overlay with synchronized video/audio...")
    print("=" * 60)

    video_duration = get_video_duration(video_path)

    if video_duration is None:
        print("❌ Could not get original video duration")
        return False

    if not has_audio_stream(video_path):
        print("❌ Original video has no audio stream")
        return False

    # --------------------------------------------------------
    # Gather valid clips
    # --------------------------------------------------------

    clip_info = []

    for clip_name, timestamp in zip(clip_order, timestamps):

        clip_path = os.path.join(
            clip_folder,
            clip_name
        )

        if not os.path.exists(clip_path):
            print(f"⚠ Clip not found: {clip_path}")
            continue

        duration = get_video_duration(clip_path)

        if duration is None:
            print(f"⚠ Could not determine duration: {clip_name}")
            continue

        if not has_audio_stream(clip_path):
            print(f"⚠ Clip has no audio: {clip_name}")
            continue

        # Ensure clip fits inside original video
        if timestamp + duration > video_duration:
            duration = video_duration - timestamp

        if duration <= 0:
            print(f"⚠ Clip does not fit: {clip_name}")
            continue

        clip_info.append({
            "path": clip_path,
            "name": clip_name,
            "timestamp": float(timestamp),
            "duration": float(duration)
        })

    if not clip_info:
        print("❌ No valid clips to insert")
        return False

    print(f"\nInserting {len(clip_info)} clips")

    # --------------------------------------------------------
    # Build FFmpeg command
    # --------------------------------------------------------

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path
    ]

    for info in clip_info:
        cmd.extend([
            "-i",
            info["path"]
        ])

    filter_parts = []

    # ========================================================
    # VIDEO
    # ========================================================

    filter_parts.append(
        "[0:v]setpts=PTS-STARTPTS[base_v]"
    )

    current_video = "[base_v]"

    for i, info in enumerate(clip_info):

        input_index = i + 1
        timestamp = info["timestamp"]
        duration = info["duration"]

        clip_video_label = f"clip_v_{i}"
        output_video_label = f"v_{i}"

        # IMPORTANT:
        # Reset the clip timeline, trim it, THEN move it forward
        # to the exact insertion timestamp.
        filter_parts.append(
            f"[{input_index}:v]"
            f"setpts=PTS-STARTPTS,"
            f"trim=duration={duration:.6f},"
            f"setpts=PTS-STARTPTS+{timestamp:.6f}/TB"
            f"[{clip_video_label}]"
        )

        # eof_action=pass makes sure the base video continues
        # normally after the inserted clip ends.
        filter_parts.append(
            f"{current_video}"
            f"[{clip_video_label}]"
            f"overlay="
            f"x=0:y=0:"
            f"enable='between("
            f"t,"
            f"{timestamp:.6f},"
            f"{timestamp + duration:.6f}"
            f")':"
            f"eof_action=pass"
            f"[{output_video_label}]"
        )

        current_video = f"[{output_video_label}]"

    filter_parts.append(
        f"{current_video}null[outv]"
    )

    # ========================================================
    # AUDIO
    # ========================================================

    filter_parts.append(
        "[0:a]asetpts=PTS-STARTPTS[base_a]"
    )

    current_audio = "[base_a]"

    for i, info in enumerate(clip_info):

        input_index = i + 1
        timestamp = info["timestamp"]
        duration = info["duration"]

        delay_ms = int(
            round(timestamp * 1000)
        )

        clip_audio_label = f"clip_a_{i}"
        muted_audio_label = f"muted_a_{i}"
        output_audio_label = f"a_{i}"

        # Trim clip audio, then delay it to its insertion point
        filter_parts.append(
            f"[{input_index}:a]"
            f"asetpts=PTS-STARTPTS,"
            f"atrim=duration={duration:.6f},"
            f"asetpts=PTS-STARTPTS,"
            f"adelay={delay_ms}:all=1"
            f"[{clip_audio_label}]"
        )

        # Mute original audio while inserted clip plays
        filter_parts.append(
            f"{current_audio}"
            f"volume="
            f"enable='between("
            f"t,"
            f"{timestamp:.6f},"
            f"{timestamp + duration:.6f}"
            f")':"
            f"volume=0"
            f"[{muted_audio_label}]"
        )

        # Add clip audio
        filter_parts.append(
            f"[{muted_audio_label}]"
            f"[{clip_audio_label}]"
            f"amix="
            f"inputs=2:"
            f"duration=first:"
            f"dropout_transition=0:"
            f"normalize=0"
            f"[{output_audio_label}]"
        )

        current_audio = f"[{output_audio_label}]"

    filter_parts.append(
        f"{current_audio}anull[aout]"
    )

    # ========================================================
    # FINAL FILTER
    # ========================================================

    final_filter = ";".join(filter_parts)

    # Uncomment for debugging if needed:
    #
    # print("\nFILTER GRAPH:")
    # print(final_filter)

    # ========================================================
    # OUTPUT SETTINGS
    # ========================================================

    cmd.extend([
        "-filter_complex",
        final_filter,

        "-map",
        "[outv]",
        "-map",
        "[aout]",

        "-c:v",
        "libx264",

        "-preset",
        "medium",

        "-crf",
        "23",

        "-c:a",
        "aac",

        "-b:a",
        "192k",

        "-movflags",
        "+faststart",

        output_path
    ])

    # ========================================================
    # RUN FFMPEG
    # ========================================================

    print("\nRunning FFmpeg...")
    print("Synchronizing inserted video and audio...")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:

        print("\n❌ FFmpeg error:\n")
        print(result.stderr)

        return False

    print(f"\n✅ Successfully created:\n{output_path}")

    return True

# ============================================================
# MAIN SCRIPT
# ============================================================

def main():

    # --------------------------------------------------------
    # Clip order
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    video_path = (
        # "costin_graphpoem_dhsi.mp4"
        "
    )

    clip_folder = (
        # "place_flagey_graphpoem_dhsi_extracted_clips"
        "after_that_graphpoem_dhsi_extracted_clips"
    )

    output_path = (
        "costin_graphpoem_dhsi_with_insertions_2.mp4"
    )

    # --------------------------------------------------------
    # Check original video
    # --------------------------------------------------------

    if not os.path.exists(video_path):

        print(
            f"❌ Original video not found:\n"
            f"{video_path}"
        )

        return

    if not os.path.exists(clip_folder):

        print(
            f"❌ Clip folder not found:\n"
            f"{clip_folder}"
        )

        return

    # --------------------------------------------------------
    # Get original video duration
    # --------------------------------------------------------

    video_duration = get_video_duration(
        video_path
    )

    if video_duration is None:

        print(
            "❌ Could not get video duration"
        )

        return

    print(
        f"\nVideo duration: "
        f"{video_duration:.2f}s "
        f"({video_duration / 60:.2f} minutes)"
    )

    print(
        f"Number of clips: "
        f"{len(clip_order)}"
    )

    # --------------------------------------------------------
    # Determine an initial clip duration
    # --------------------------------------------------------

    first_clip = os.path.join(
        clip_folder,
        clip_order[0]
    )

    if os.path.exists(first_clip):

        clip_duration = (
            get_video_duration(first_clip)
            or 10
        )

    else:

        print(
            f"⚠ First clip not found: "
            f"{first_clip}"
        )

        clip_duration = 10

    print(
        f"Initial clip duration: "
        f"{clip_duration:.2f}s"
    )

    # --------------------------------------------------------
    # Generate timestamps
    # --------------------------------------------------------

    timestamps, assumed_clip_duration = (
        generate_well_spaced_timestamps(
            n=len(clip_order),
            video_duration=video_duration,
            clip_duration=clip_duration,
            min_gap=3,
            seed=42
        )
    )

    if not timestamps:

        print(
            "❌ Could not generate timestamps"
        )

        return

    # --------------------------------------------------------
    # Display timestamps
    # --------------------------------------------------------

    print("\nGenerated timestamps:")

    for i, (clip, timestamp) in enumerate(
        zip(clip_order, timestamps),
        start=1
    ):

        original_timestamp = (
            parse_timestamp_from_name(clip)
        )

        print(
            f"\n  {i}. {clip}"
        )

        print(
            f"     Original timestamp: "
            f"{original_timestamp}s"
        )

        print(
            f"     New timestamp: "
            f"{timestamp:.2f}s"
        )

    print(
        f"\nTimestamp range: "
        f"{min(timestamps):.2f}s "
        f"to "
        f"{max(timestamps):.2f}s"
    )

    print(
        f"Assumed spacing duration: "
        f"{assumed_clip_duration:.2f}s"
    )

    # --------------------------------------------------------
    # Insert clips
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print(
        "STARTING VIDEO AND AUDIO INSERTION"
    )
    print("=" * 60)

    success = (
        insert_clips_with_audio_correct(
            video_path=video_path,
            clip_folder=clip_folder,
            clip_order=clip_order,
            timestamps=timestamps,
            output_path=output_path
        )
    )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    if success:

        output_duration = (
            get_video_duration(output_path)
        )

        print("\n" + "=" * 60)
        print("FINAL OUTPUT")
        print("=" * 60)

        print(
            f"📁 File: "
            f"{output_path}"
        )

        if output_duration is not None:

            print(
                f"📹 Output duration: "
                f"{output_duration:.2f}s"
            )

            print(
                f"📹 Original duration: "
                f"{video_duration:.2f}s"
            )

            duration_difference = abs(
                output_duration
                - video_duration
            )

            print(
                f"Difference: "
                f"{duration_difference:.2f}s"
            )

            if duration_difference < 1:

                print(
                    "✅ Output duration matches "
                    "the original."
                )

            else:

                print(
                    "⚠ Output duration differs "
                    "from the original."
                )

        if has_audio_stream(output_path):

            print(
                "🔊 Audio stream confirmed "
                "in output."
            )

        else:

            print(
                "⚠ WARNING: No audio stream detected "
                "in output."
            )

    else:

        print("\n" + "=" * 60)
        print("❌ INSERTION FAILED")
        print("=" * 60)

        print(
            "The FFmpeg error above should help "
            "identify the problem."
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
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
    seed=None,
    mode="pareto"
):
    """
    Generate timestamps using different spacing strategies.

    Available modes:
        pareto     - Uneven gaps with occasional large gaps
        chaos      - Logistic-map chaotic distribution
        fibonacci  - Fibonacci/golden-ratio-inspired spacing
        uniform    - Evenly distributed
        random     - Random gap distribution
        clustered  - Groups of clips separated by larger gaps
    """

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    margin = 5

    # Total amount of time occupied by clips
    total_clip_time = n * clip_duration

    # Time available for gaps
    available_gap_time = (
        video_duration
        - margin * 2
        - total_clip_time
    )

    # If clips do not fit, reduce assumed duration
    if available_gap_time < 0:

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
            f"⚠ Adjusted clip duration "
            f"to {clip_duration:.2f}s"
        )

        total_clip_time = n * clip_duration

        available_gap_time = (
            video_duration
            - margin * 2
            - total_clip_time
        )

    # Number of gaps between clips
    num_gaps = max(0, n - 1)

    if num_gaps == 0:

        return [margin], clip_duration

    # --------------------------------------------------------
    # Generate raw gap weights
    # --------------------------------------------------------

    mode = mode.lower()

    if mode == "pareto":

        # Heavy-tailed distribution:
        # mostly smaller gaps, occasionally large ones
        gaps = (
            np.random.pareto(1.5, num_gaps)
            + 0.2
        )

    elif mode == "chaos":

        # Logistic map.
        # A classic deterministic chaotic system.
        r = 3.99

        # Random starting value unless seed is fixed
        x = random.uniform(0.1, 0.9)

        values = []

        # Warm-up iterations
        for _ in range(20):
            x = r * x * (1 - x)

        # Generate chaotic values
        for _ in range(num_gaps):

            x = r * x * (1 - x)

            values.append(x)

        gaps = np.array(values)

        # Prevent extremely tiny gaps
        gaps = gaps + 0.1

    elif mode == "fibonacci":

        # Fibonacci sequence as gap weights
        fib = [1, 1]

        while len(fib) < num_gaps:

            fib.append(
                fib[-1] + fib[-2]
            )

        gaps = np.array(
            fib[:num_gaps],
            dtype=float
        )

        # Reverse some of the sequence randomly so that
        # the largest gap is not always at the end.
        if random.choice([True, False]):

            gaps = gaps[::-1]

    elif mode == "uniform":

        # All gaps equal
        gaps = np.ones(num_gaps)

    elif mode == "random":

        # Completely random gap weights
        gaps = np.random.uniform(
            0.2,
            1.0,
            num_gaps
        )

    elif mode == "clustered":

        # Small gaps inside clusters,
        # larger gaps between clusters

        gaps = []

        for i in range(num_gaps):

            if i % 3 == 2:

                # Larger gap between clusters
                gap = random.uniform(
                    2.0,
                    6.0
                )

            else:

                # Smaller gap within cluster
                gap = random.uniform(
                    0.2,
                    1.0
                )

            gaps.append(gap)

        gaps = np.array(gaps)

    else:

        print(
            f"⚠ Unknown mode '{mode}', "
            f"using Pareto."
        )

        mode = "pareto"

        gaps = (
            np.random.pareto(1.5, num_gaps)
            + 0.2
        )

    # --------------------------------------------------------
    # Scale gaps to fill available space
    # --------------------------------------------------------

    gap_sum = np.sum(gaps)

    if gap_sum > 0:

        gaps = (
            gaps / gap_sum
            * available_gap_time
        )

    # --------------------------------------------------------
    # Build timestamps
    # --------------------------------------------------------

    timestamps = []

    current_time = margin

    for i in range(n):

        timestamps.append(current_time)

        if i < num_gaps:

            current_time += (
                clip_duration
                + gaps[i]
            )

    # --------------------------------------------------------
    # Add small jitter
    #
    # Not applied to uniform because that would defeat
    # perfectly even spacing.
    # --------------------------------------------------------

    if mode not in ["uniform", "fibonacci"]:

        for i in range(1, len(timestamps) - 1):

            jitter = random.uniform(
                -1.0,
                1.0
            )

            previous_end = (
                timestamps[i - 1]
                + clip_duration
                + min_gap
            )

            next_start_limit = (
                timestamps[i + 1]
                - clip_duration
                - min_gap
            )

            if previous_end < next_start_limit:

                timestamps[i] = min(
                    max(
                        timestamps[i] + jitter,
                        previous_end
                    ),
                    next_start_limit
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

        # "shot_0502_time_3809s.mp4",
        # "shot_0665_time_5628s.mp4",
        # "shot_0441_time_4099s.mp4",
        # "shot_0222_time_1590s.mp4",
        # "shot_0788_time_6806s.mp4",
        # "shot_0568_time_4339s.mp4",
        # "shot_0371_time_3458s.mp4",
        # "shot_0635_time_4843s.mp4",
        # "shot_1168_time_7596s.mp4",
        # "shot_0682_time_5321s.mp4"
        # "shot_0668_time_5646s.mp4",
        # "shot_0568_time_4339s.mp4",
        # "shot_0635_time_4843s.mp4",
        # "shot_0682_time_5321s.mp4",
        # "shot_0788_time_6806s.mp4"
        "shot_0665_time_5628s.mp4",
        "shot_0635_time_4843s.mp4",
        "shot_0568_time_4339s.mp4",
        "shot_0371_time_3458s.mp4",
        "shot_0682_time_5321s.mp4",
        "shot_0788_time_6806s.mp4",
        "shot_0502_time_3809s.mp4",
        "shot_1168_time_7596s.mp4",
        "shot_0441_time_4099s.mp4",
        "shot_0222_time_1590s.mp4"
    ]
    # r0 → r1 → r2 → r8 → r3 → r4 → r5 → r7 → r9 → r6

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    video_path = (
        # "costin_graphpoem_dhsi.mp4"
        # "comm_singularity_graphpoem_dhsi_uvic.mp4"
        "graphpoem_dhsi_uvic_1.mp4"
    )

    clip_folder = (
        # "place_flagey_graphpoem_dhsi_extracted_clips"
        # "after_that_graphpoem_dhsi_extracted_clips"
        "place_flagey_graphpoem_dhsi_extracted_clips"
    )

    output_path = (
        # "costin_graphpoem_dhsi_with_insertions_2.mp4"
        # "community_for_community_graphpoem_dhsi_w_insertions.mp4"
        "institutional_graphpoem_dhsi_w_personal_insertions.mp4"
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

    timestamp_mode = "chaos"

    # Set to None for a new arrangement every run.
    # Set to an integer, e.g. 42, to reproduce the same arrangement.
    timestamp_seed = None
    
    timestamps, assumed_clip_duration = (
        generate_well_spaced_timestamps(
            n=len(clip_order),
            video_duration=video_duration,
            clip_duration=clip_duration,
            min_gap=3,
            seed=timestamp_seed,
            mode=timestamp_mode
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

#!/usr/bin/env python3
"""
Analyze the audio track of an MP4 and write:

{
  "audio_features": {...},
  "affect_vector": {...},
  "temporal_features": {...}
}

Source-derived pipeline:
1. Extract/load audio from the MP4.
2. Compute the seven audio features from the notebooks.
3. Compute OpenL3 affect embeddings.
4. Compute temporal-flow features:
   - MFCC + chroma + RMS windows
   - feature variation index
   - change scores and ruptures
   - motif clustering
   - linear / cyclical / recursive / hybrid scores
   - recursive drift and event types

Usage:
    python analyze_mp4_audio.py input.mp4
    python analyze_mp4_audio.py input.mp4 -o result.json

You're very welcome! If you've saved the script as, for example:

```text
analyze_mp4_audio.py
```

and your video is:

```text
my_video.mp4
```

open Terminal, navigate to the folder containing both files, and run:

```bash
python analyze_mp4_audio.py my_video.mp4
```

That will automatically create:

```text
my_video_audio_analysis.json
```

### Specify an output filename

```bash
python analyze_mp4_audio.py my_video.mp4 -o results.json
```

### Use the other OpenL3 content type

The script defaults to `music`, as in the notebook. You can instead use `env`:

```bash
python analyze_mp4_audio.py my_video.mp4 --content-type env
```

### Change the affect chunk size

For example, 3-second chunks:

```bash
python analyze_mp4_audio.py my_video.mp4 --affect-chunk 3
```

### Change the temporal analysis window and hop

For 2-second windows with 1-second hops:

```bash
python analyze_mp4_audio.py my_video.mp4 \
  --window-duration 2 \
  --hop-duration 1
```

### Combine everything

```bash
python analyze_mp4_audio.py my_video.mp4 \
  -o analysis.json \
  --content-type music \
  --affect-chunk 2 \
  --window-duration 1 \
  --hop-duration 0.5
```

On macOS/Linux, if `python` doesn't work, try:

```bash
python3 analyze_mp4_audio.py my_video.mp4
```

The **first positional argument is always the MP4 file**; everything beginning with `--` or `-o` is optional.

"""

import argparse
import json
import os
import tempfile

import librosa
import numpy as np
import soundfile as sf
import openl3

from moviepy import VideoFileClip
from scipy.spatial.distance import euclidean
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------
# JSON UTILITIES
# ---------------------------------------------------------------------

def to_jsonable(value):
    """Convert NumPy values recursively into ordinary Python JSON values."""
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]

    return value


# ---------------------------------------------------------------------
# AUDIO EXTRACTION
# ---------------------------------------------------------------------

def extract_audio_from_mp4(mp4_path, wav_out_path, fps=48000):
    """
    Extract the MP4 audio track as WAV.

    Based on the notebook's:
        VideoFileClip(...).audio.write_audiofile(...)
    """
    clip = VideoFileClip(mp4_path)

    try:
        if clip.audio is None:
            raise ValueError(f"No audio track found in: {mp4_path}")

        clip.audio.write_audiofile(
            wav_out_path,
            fps=fps,
            logger=None
        )
    finally:
        clip.close()


# ---------------------------------------------------------------------
# BASIC AUDIO FEATURES
# ---------------------------------------------------------------------

def calculate_pacing_variance(
    audio,
    sr,
    frame_length=2048,
    hop_length=512
):
    """
    Notebook definition:
        std(RMS) + sqrt(sum(diff(RMS)^2))
    """
    if len(audio) == 0:
        return 0.0

    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    if len(rms) < 2:
        return float(np.std(rms)) if len(rms) else 0.0

    spectral_flux = np.sqrt(np.sum(np.diff(rms) ** 2))

    return float(np.std(rms) + spectral_flux)


def calculate_fricative_density(
    audio,
    sr,
    frame_length=2048,
    hop_length=512,
    fricative_range=(4000, 8000)
):
    """
    Notebook definition:
    Count high-frequency energy above the median in the fricative range.
    """
    if len(audio) == 0:
        return 0.0

    n_fft = min(frame_length, max(2, len(audio)))

    stft = np.abs(
        librosa.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length
        )
    )

    freqs = librosa.fft_frequencies(
        sr=sr,
        n_fft=n_fft
    )

    idx = np.where(
        (freqs >= fricative_range[0])
        & (freqs <= fricative_range[1])
    )[0]

    if len(idx) == 0:
        return 0.0

    fricative_energy = stft[idx, :]

    if fricative_energy.size == 0:
        return 0.0

    fricative_frames = np.sum(
        fricative_energy > np.median(fricative_energy)
    )

    total_frames = fricative_energy.shape[1]

    return float(
        fricative_frames / total_frames
        if total_frames > 0
        else 0.0
    )


def calculate_plosive_density(
    audio,
    sr,
    frame_length=2048,
    hop_length=512,
    plosive_range=(0, 500)
):
    """
    Notebook definition:
    Detect bursts in the low-frequency STFT region.
    """
    if len(audio) == 0:
        return 0.0

    n_fft = min(frame_length, max(2, len(audio)))

    stft = np.abs(
        librosa.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length
        )
    )

    freqs = librosa.fft_frequencies(
        sr=sr,
        n_fft=n_fft
    )

    idx = np.where(
        (freqs >= plosive_range[0])
        & (freqs <= plosive_range[1])
    )[0]

    if len(idx) == 0:
        return 0.0

    plosive_energy = stft[idx, :]

    if (
        plosive_energy.size == 0
        or plosive_energy.shape[1] < 2
    ):
        return 0.0

    burst_frames = np.sum(
        np.diff(plosive_energy, axis=1)
        > np.median(plosive_energy)
    )

    total_frames = plosive_energy.shape[1]

    return float(
        burst_frames / total_frames
        if total_frames > 0
        else 0.0
    )


def calculate_vocal_smoothness(
    audio,
    sr,
    frame_length=2048,
    hop_length=512
):
    """
    Uses the later notebook version based on onset-strength changes.

    smoothness = 1 - mean(abs(diff(onset_strength))) / max(...)
    """
    if len(audio) == 0:
        return 0.0

    spectral_flux = librosa.onset.onset_strength(
        y=audio,
        sr=sr
    )

    if spectral_flux.size < 2:
        return 0.0

    flux = np.abs(np.diff(spectral_flux))

    if flux.size == 0 or np.max(flux) == 0:
        return 0.0

    return float(
        1 - (np.mean(flux) / np.max(flux))
    )


def analyze_audio_features(
    audio,
    sr,
    chunk_duration=1.5
):
    """
    Compute the seven audio features from the notebook.
    """
    if len(audio) == 0:
        raise ValueError("Audio is empty.")

    pacing_variance = calculate_pacing_variance(
        audio,
        sr
    )

    fricative_density = calculate_fricative_density(
        audio,
        sr
    )

    plosive_density = calculate_plosive_density(
        audio,
        sr
    )

    vocal_smoothness = calculate_vocal_smoothness(
        audio,
        sr
    )

    onset_env = librosa.onset.onset_strength(
        y=audio,
        sr=sr
    )

    chunk_samples = max(
        1,
        int(chunk_duration * sr)
    )

    rms_values = []

    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples]

        if len(chunk) > 0:
            rms_values.append(
                np.sqrt(np.mean(chunk ** 2))
            )

    if rms_values and max(rms_values) > 0:
        silence_threshold = 0.2 * max(rms_values)

        silence_chunks = sum(
            1
            for rms in rms_values
            if rms < silence_threshold
        )

        silence_ratio = silence_chunks / len(rms_values)
    else:
        silence_ratio = 1.0

    onsets = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr
    )

    tempo, _ = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr
    )

    duration = librosa.get_duration(
        y=audio,
        sr=sr
    )

    syllable_density = (
        len(onsets) / duration
        if duration > 0
        else 0.0
    )

    return {
        "syllable_density": float(syllable_density),
        "tempo": float(np.asarray(tempo).squeeze()),
        "pacing_variance": float(pacing_variance),
        "fricative_density": float(fricative_density),
        "plosive_density": float(plosive_density),
        "vocal_smoothness": float(vocal_smoothness),
        "silence_ratio": float(silence_ratio),
    }


# ---------------------------------------------------------------------
# AFFECT VECTOR: OPENL3
# ---------------------------------------------------------------------

def make_stereo(audio):
    """
    OpenL3 notebook path expects audio in shape:
        (samples, channels)
    """
    if audio.ndim == 1:
        return np.stack(
            [audio, audio],
            axis=-1
        )

    if audio.ndim == 2:
        if audio.shape[1] == 1:
            return np.repeat(audio, 2, axis=1)

        return audio[:, :2]

    raise ValueError(
        f"Unexpected audio shape: {audio.shape}"
    )


def chunk_audio_array(
    audio,
    sr,
    chunk_duration=2.0,
    keep_min_fraction=0.8
):
    """
    Based on the notebook's audio chunker.

    Keeps a final chunk only if it is at least
    80% of the requested chunk duration.
    """
    chunk_length = max(
        1,
        int(chunk_duration * sr)
    )

    chunks = []

    for i in range(
        0,
        len(audio),
        chunk_length
    ):
        chunk = audio[i:i + chunk_length]

        if len(chunk) >= chunk_length * keep_min_fraction:
            chunks.append(chunk)

    return chunks


def extract_audio_affect_vector(
    audio_chunk,
    sr,
    embedding_size=512,
    content_type="music"
):
    """
    Extract one OpenL3 embedding per chunk,
    then average the embedding over time.
    """
    stereo_chunk = make_stereo(audio_chunk)

    embedding, _ = openl3.get_audio_embedding(
        audio=stereo_chunk,
        sr=sr,
        input_repr="mel256",
        content_type=content_type,
        embedding_size=embedding_size
    )

    return np.mean(
        embedding,
        axis=0
    )


def analyze_affect(
    audio,
    sr,
    chunk_duration=2.0,
    embedding_size=512,
    content_type="music"
):
    """
    Produce:
      - one embedding per audio chunk
      - one global embedding, averaged across chunks
    """
    chunks = chunk_audio_array(
        audio,
        sr,
        chunk_duration=chunk_duration
    )

    if not chunks:
        chunks = [audio]

    vectors = []

    for chunk in chunks:
        vector = extract_audio_affect_vector(
            chunk,
            sr,
            embedding_size=embedding_size,
            content_type=content_type
        )

        vectors.append(vector)

    vectors = np.asarray(vectors)

    global_vector = np.mean(
        vectors,
        axis=0
    )

    return {
        "embedding_size": int(embedding_size),
        "content_type": content_type,
        "chunk_duration": float(chunk_duration),
        "n_chunks": int(len(vectors)),
        "vector": global_vector.tolist(),

        # Retained because the notebooks work with
        # per-chunk affect vectors.
        "chunk_vectors": vectors.tolist()
    }


# ---------------------------------------------------------------------
# TEMPORAL FLOW
# ---------------------------------------------------------------------

def compute_recursive_drift(
    segments,
    labels,
    hop_duration
):
    """
    Direct consolidation of the notebook's recursive-drift logic.
    """
    motif_instances = {}
    event_types = []

    if not segments:
        return 0.0, event_types

    avg_segment_duration = np.mean([
        (end - start) * hop_duration
        for start, end, _ in segments
    ])

    time_threshold = max(
        2.0,
        avg_segment_duration * 2
    )

    for idx, (start, end, features) in enumerate(segments):
        label = int(labels[idx])

        if label == -1:
            continue

        motif_instances.setdefault(
            label,
            []
        ).append(
            (start, end, features)
        )

    drift_scores = []

    for label, instances in motif_instances.items():
        instances.sort(key=lambda x: x[0])

        for i in range(1, len(instances)):
            prev = instances[i - 1]
            curr = instances[i]

            feature_drift = euclidean(
                curr[2],
                prev[2]
            )

            time_drift = (
                curr[0] - prev[0]
            ) * hop_duration

            drift = (
                feature_drift * time_drift
            )

            drift_scores.append(drift)

            if (
                feature_drift < 0.3
                and time_drift < time_threshold
            ):
                event_type = "loop"

            elif (
                feature_drift < 0.3
                and time_drift >= time_threshold
            ):
                event_type = "return"

            elif (
                feature_drift < 0.7
                and time_drift >= time_threshold
            ):
                event_type = "recursive echo"

            elif (
                feature_drift >= 0.7
                and time_drift < time_threshold
            ):
                event_type = "mutation"

            else:
                event_type = "ghost return"

            event_types.append({
                "label": int(label),
                "from": round(
                    prev[0] * hop_duration,
                    3
                ),
                "to": round(
                    curr[0] * hop_duration,
                    3
                ),
                "feature_drift": round(
                    float(feature_drift),
                    3
                ),
                "drift": round(
                    float(drift),
                    3
                ),
                "time_drift": round(
                    float(time_drift),
                    3
                ),
                "duration_prev": round(
                    (prev[1] - prev[0])
                    * hop_duration,
                    3
                ),
                "duration_curr": round(
                    (curr[1] - curr[0])
                    * hop_duration,
                    3
                ),
                "type": event_type
            })

    recursive_score = (
        float(np.mean(drift_scores))
        if drift_scores
        else 0.0
    )

    return recursive_score, event_types


def analyze_temporal_features(
    audio,
    sr,
    window_duration=1.0,
    hop_duration=0.5,
    eps=0.25,
    min_samples=2
):
    """
    Consolidated temporal-flow analysis.

    Window feature vector:
        mean MFCC
        mean chroma
        mean RMS

    Then:
        FVI weighting
        change scores
        rupture detection
        segmentation
        cosine/DBSCAN motif clustering
        linear/cyclical/recursive/hybrid scores
        recursive drift/events
    """
    window_size = max(
        1,
        int(sr * window_duration)
    )

    hop_size = max(
        1,
        int(sr * hop_duration)
    )

    feature_vectors = []

    # Preserve the notebook's sliding-window idea,
    # while also allowing a short file to yield one window.
    starts = list(
        range(
            0,
            max(1, len(audio) - window_size + 1),
            hop_size
        )
    )

    if not starts:
        starts = [0]

    for start in starts:
        window = audio[start:start + window_size]

        if len(window) < 2:
            continue

        n_fft = min(
            2048,
            len(window)
        )

        mfcc = np.mean(
            librosa.feature.mfcc(
                y=window,
                sr=sr,
                n_fft=n_fft
            ),
            axis=1
        )

        chroma = np.mean(
            librosa.feature.chroma_stft(
                y=window,
                sr=sr,
                n_fft=n_fft
            ),
            axis=1
        )

        rms = np.mean(
            librosa.feature.rms(
                y=window,
                frame_length=min(
                    2048,
                    len(window)
                )
            )
        )

        feature_vector = np.concatenate([
            mfcc,
            chroma,
            [rms]
        ])

        feature_vectors.append(
            feature_vector
        )

    if not feature_vectors:
        return {
            "window_duration": float(window_duration),
            "hop_duration": float(hop_duration),
            "segments": 0,
            "motifs": 0,
            "uniques": 0,
            "ruptures": 0,
            "score_linear": 0.0,
            "score_cyclical": 0.0,
            "score_recursive": 0.0,
            "score_recursive_drift": 0.0,
            "score_hybrid": 0.0,
            "mean_change_score": 0.0,
            "max_change_score": 0.0,
            "segment_annotations": [],
            "recursive_events": []
        }

    fv_array = np.asarray(
        feature_vectors
    )

    # -------------------------------------------------------------
    # Feature Variation Index (FVI)
    # -------------------------------------------------------------

    if len(fv_array) > 1:
        fvi = (
            np.mean(
                np.abs(
                    np.diff(
                        fv_array,
                        axis=0
                    )
                ),
                axis=0
            )
            /
            (
                np.std(
                    fv_array,
                    axis=0
                )
                + 1e-6
            )
        )
    else:
        fvi = np.ones(
            fv_array.shape[1]
        )

    fvi_sum = np.sum(fvi)

    if fvi_sum > 0:
        fvi_normalized = fvi / fvi_sum
    else:
        fvi_normalized = np.ones_like(fvi) / len(fvi)

    weighted_features = (
        fv_array * fvi_normalized
    )

    # -------------------------------------------------------------
    # Change scores and ruptures
    # -------------------------------------------------------------

    change_scores = [
        float(
            euclidean(
                weighted_features[i + 1],
                weighted_features[i]
            )
        )
        for i in range(
            len(weighted_features) - 1
        )
    ]

    if change_scores:
        threshold = (
            np.mean(change_scores)
            + np.std(change_scores) * 0.5
        )

        ruptures = [
            i
            for i, score in enumerate(change_scores)
            if score > threshold
        ]
    else:
        threshold = 0.0
        ruptures = []

    # -------------------------------------------------------------
    # Segmentation
    # -------------------------------------------------------------

    segments = []
    start_idx = 0

    boundary_indices = (
        ruptures
        + [len(feature_vectors) - 1]
    )

    for rupture_idx in boundary_indices:
        segment = weighted_features[
            start_idx:rupture_idx + 1
        ]

        if len(segment) == 0:
            continue

        avg_feature = np.mean(
            segment,
            axis=0
        )

        segments.append(
            (
                start_idx,
                rupture_idx + 1,
                avg_feature
            )
        )

        start_idx = rupture_idx + 1

    if not segments:
        segments = [
            (
                0,
                len(feature_vectors),
                np.mean(
                    weighted_features,
                    axis=0
                )
            )
        ]

    segment_features = np.asarray(
        [seg[2] for seg in segments]
    )

    # -------------------------------------------------------------
    # Motif clustering
    # -------------------------------------------------------------

    if len(segment_features) >= min_samples:
        clustering = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric="cosine"
        ).fit(
            segment_features
        )

        labels = clustering.labels_
    else:
        labels = np.full(
            len(segment_features),
            -1,
            dtype=int
        )

    avg_segment_duration = np.mean([
        (end - start) * hop_duration
        for start, end, _ in segments
    ])

    rupture_threshold = max(
        1.5,
        avg_segment_duration * 0.75
    )

    segment_annotations = []

    for idx, (start, end, _) in enumerate(segments):
        duration = (
            end - start
        ) * hop_duration

        label = int(labels[idx])

        if (
            duration < rupture_threshold
            and (
                idx == 0
                or idx in ruptures
            )
        ):
            seg_type = "rupture"

        elif label == -1:
            seg_type = "unique"

        else:
            seg_type = "motif"

        segment_annotations.append({
            "start": round(
                start * hop_duration,
                3
            ),
            "end": round(
                end * hop_duration,
                3
            ),
            "type": seg_type,
            "label": label
        })

    # -------------------------------------------------------------
    # Temporal-flow scores
    # -------------------------------------------------------------

    n_segments = len(
        segment_annotations
    )

    n_motifs = sum(
        1
        for segment in segment_annotations
        if segment["type"] == "motif"
    )

    n_uniques = sum(
        1
        for segment in segment_annotations
        if segment["type"] == "unique"
    )

    n_ruptures = sum(
        1
        for segment in segment_annotations
        if segment["type"] == "rupture"
    )

    score_linear = (
        1
        - (
            n_ruptures + n_uniques
        ) / n_segments
        if n_segments > 0
        else 0.0
    )

    score_cyclical = (
        n_motifs / n_segments
        if n_segments > 0
        else 0.0
    )

    non_unique_labels = set(
        segment["label"]
        for segment in segment_annotations
        if segment["label"] != -1
    )

    score_recursive = (
        len(non_unique_labels)
        / n_segments
        if n_segments > 0
        else 0.0
    )

    score_hybrid = (
        1.0
        - min(
            score_linear,
            score_cyclical
        )
    )

    recursive_drift_score, event_types = (
        compute_recursive_drift(
            segments,
            labels,
            hop_duration
        )
    )

    return {
        "window_duration": float(window_duration),
        "hop_duration": float(hop_duration),

        "segments": int(n_segments),
        "motifs": int(n_motifs),
        "uniques": int(n_uniques),
        "ruptures": int(n_ruptures),

        "score_linear": round(
            float(score_linear),
            3
        ),

        "score_cyclical": round(
            float(score_cyclical),
            3
        ),

        "score_recursive": round(
            float(score_recursive),
            3
        ),

        "score_recursive_drift": round(
            float(recursive_drift_score),
            3
        ),

        "score_hybrid": round(
            float(score_hybrid),
            3
        ),

        "mean_change_score": round(
            float(np.mean(change_scores))
            if change_scores
            else 0.0,
            6
        ),

        "max_change_score": round(
            float(np.max(change_scores))
            if change_scores
            else 0.0,
            6
        ),

        "change_threshold": round(
            float(threshold),
            6
        ),

        "segment_annotations":
            segment_annotations,

        "recursive_events":
            event_types
    }


# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------

def analyze_mp4_audio(
    mp4_path,
    feature_chunk_duration=1.5,
    affect_chunk_duration=2.0,
    embedding_size=512,
    content_type="music",
    window_duration=1.0,
    hop_duration=0.5,
    eps=0.25,
    min_samples=2
):
    """
    Main entry point.

    Returns exactly the three requested top-level groups:
        audio_features
        affect_vector
        temporal_features
    """
    if not os.path.isfile(mp4_path):
        raise FileNotFoundError(
            f"File not found: {mp4_path}"
        )

    with tempfile.TemporaryDirectory() as temp_dir:
        wav_path = os.path.join(
            temp_dir,
            "audio.wav"
        )

        # 1. Extract audio from MP4.
        extract_audio_from_mp4(
            mp4_path,
            wav_path
        )

        # 2. Load audio.
        audio, sr = sf.read(
            wav_path,
            always_2d=False
        )

        # Convert stereo/multichannel audio to mono for
        # the conventional Librosa feature analysis.
        if audio.ndim > 1:
            mono_audio = np.mean(
                audio,
                axis=1
            )
        else:
            mono_audio = audio

        mono_audio = np.asarray(
            mono_audio,
            dtype=np.float32
        )

        # ---------------------------------------------------------
        # Conventional audio features
        # ---------------------------------------------------------

        audio_features = analyze_audio_features(
            mono_audio,
            sr,
            chunk_duration=feature_chunk_duration
        )

        # ---------------------------------------------------------
        # OpenL3 affect representation
        # ---------------------------------------------------------

        affect_vector = analyze_affect(
            mono_audio,
            sr,
            chunk_duration=affect_chunk_duration,
            embedding_size=embedding_size,
            content_type=content_type
        )

        # ---------------------------------------------------------
        # Temporal-flow analysis
        # ---------------------------------------------------------

        temporal_features = analyze_temporal_features(
            mono_audio,
            sr,
            window_duration=window_duration,
            hop_duration=hop_duration,
            eps=eps,
            min_samples=min_samples
        )

    return {
        "audio_features": audio_features,
        "affect_vector": affect_vector,
        "temporal_features": temporal_features
    }


# ---------------------------------------------------------------------
# COMMAND-LINE INTERFACE
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the audio track of an MP4 and "
            "write audio, affect, and temporal features as JSON."
        )
    )

    parser.add_argument(
        "mp4",
        help="Input MP4 file"
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Output JSON path. "
            "Default: <input-name>_audio_analysis.json"
        )
    )

    parser.add_argument(
        "--content-type",
        choices=["music", "env"],
        default="music",
        help="OpenL3 content type."
    )

    parser.add_argument(
        "--affect-chunk",
        type=float,
        default=2.0,
        help=(
            "Chunk duration in seconds for "
            "OpenL3 affect embeddings."
        )
    )

    parser.add_argument(
        "--window-duration",
        type=float,
        default=1.0,
        help=(
            "Temporal analysis window duration "
            "in seconds."
        )
    )

    parser.add_argument(
        "--hop-duration",
        type=float,
        default=0.5,
        help=(
            "Temporal analysis hop duration "
            "in seconds."
        )
    )

    args = parser.parse_args()

    if args.output is None:
        stem, _ = os.path.splitext(
            args.mp4
        )

        args.output = (
            stem
            + "_audio_analysis.json"
        )

    result = analyze_mp4_audio(
        args.mp4,
        affect_chunk_duration=args.affect_chunk,
        content_type=args.content_type,
        window_duration=args.window_duration,
        hop_duration=args.hop_duration
    )

    result = to_jsonable(result)

    with open(
        args.output,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"Analysis written to: {args.output}"
    )


if __name__ == "__main__":
    main()

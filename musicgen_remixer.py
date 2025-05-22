# MusicGen Remixer for Python 3.11
# Compatible with Python 3.11 and updated library versions as of May 22, 2025

import replicate
from pathlib import Path
import urllib.request
import json
import numpy as np
import requests
from pydub import AudioSegment
from io import BytesIO
from scipy.signal import resample_poly
import argparse
import logging

def download_audio_and_load_as_numpy(url):
    """
    Downloads an audio file (MP3 or WAV) from a URL and loads it into a NumPy array.
    
    Args:
        url (str): URL of the audio file.
    
    Returns:
        tuple: (audio_data as np.ndarray, sample_rate as int)
    
    Raises:
        ValueError: If the file format is unsupported or download fails.
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        raise ValueError(f"Failed to download audio from {url}: {e}")

    audio_file = BytesIO(response.content)
    file_format = url.split(".")[-1].lower()

    if file_format not in ["mp3", "wav"]:
        raise ValueError("Unsupported file format: only MP3 and WAV are supported.")

    try:
        audio = AudioSegment.from_file(audio_file, format=file_format)
    except Exception as e:
        raise ValueError(f"Failed to load audio file: {e}")

    channel_count = audio.channels
    audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)
    audio_data = audio_data.reshape(-1, channel_count)

    return audio_data, audio.frame_rate

def save_numpy_as_audio(audio_data, sample_rate, output_filename):
    """
    Saves a NumPy array as an audio file (MP3 or WAV).
    
    Args:
        audio_data (np.ndarray): Audio data to save.
        sample_rate (int): Sample rate of the audio.
        output_filename (str): Output file path.
    """
    file_format = output_filename.split(".")[-1].lower()
    channels = audio_data.shape[1] if audio_data.ndim > 1 else 1

    audio_segment = AudioSegment(
        audio_data.tobytes(),
        frame_rate=sample_rate,
        sample_width=audio_data.dtype.itemsize,
        channels=channels,
    )

    try:
        audio_segment.export(output_filename, format=file_format)
    except Exception as e:
        raise ValueError(f"Failed to save audio to {output_filename}: {e}")

def resample_audio(audio_data, original_sample_rate, new_sample_rate):
    """
    Resamples audio data to a new sample rate using resample_poly.
    
    Args:
        audio_data (np.ndarray): Audio data to resample.
        original_sample_rate (int): Original sample rate.
        new_sample_rate (int): Target sample rate.
    
    Returns:
        np.ndarray: Resampled audio data.
    """
    if original_sample_rate == new_sample_rate:
        return audio_data

    num_original_samples = audio_data.shape[0]
    resample_ratio = new_sample_rate / original_sample_rate
    num_new_samples = int(num_original_samples * resample_ratio)

    if audio_data.ndim == 2:
        resampled = np.zeros((num_new_samples, audio_data.shape[1]), dtype=np.float32)
        for ch in range(audio_data.shape[1]):
            resampled[:, ch] = resample_poly(audio_data[:, ch], new_sample_rate, original_sample_rate)
    else:
        resampled = resample_poly(audio_data, new_sample_rate, original_sample_rate)

    return resampled

def normalize_audio(audio_data):
    """
    Normalizes audio data to [-1, 1].
    
    Args:
        audio_data (np.ndarray): Audio data to normalize.
    
    Returns:
        np.ndarray: Normalized audio data.
    """
    max_val = np.max(np.abs(audio_data))
    return audio_data / max_val if max_val != 0 else audio_data

def mix_audio_volumes(audio1, audio2, weight1=0.5, weight2=0.5):
    """
    Mixes two audio arrays with specified weights.
    
    Args:
        audio1 (np.ndarray): First audio data.
        audio2 (np.ndarray): Second audio data.
        weight1 (float): Weight for audio1.
        weight2 (float): Weight for audio2.
    
    Returns:
        np.ndarray: Mixed audio scaled to int16.
    
    Raises:
        ValueError: If audio shapes do not match.
    """
    if audio1.shape != audio2.shape:
        raise ValueError("Audio arrays must have the same shape")

    audio1_normalized = normalize_audio(audio1)
    audio2_normalized = normalize_audio(audio2)
    mixed_audio = (audio1_normalized * weight1) + (audio2_normalized * weight2)

    max_int16 = np.iinfo(np.int16).max
    return np.clip(mixed_audio * max_int16, -max_int16, max_int16).astype(np.int16)

def int16_scale(audio):
    """
    Scales audio to int16 range.
    
    Args:
        audio (np.ndarray): Audio data to scale.
    
    Returns:
        np.ndarray: Scaled audio data.
    """
    max_int16 = np.iinfo(np.int16).max
    return np.clip(audio * max_int16, -max_int16, max_int16).astype(np.int16)

def main(
    prompt,
    audio_path,
    model_version="chord",
    beat_sync_threshold=None,
    upscale=False,
    mix_weight=0.65,
    output_path="output",
):
    """
    Generates a remix of an audio file based on a prompt.
    
    Args:
        prompt (str): Prompt for generating the remix.
        audio_path (str): Path to the input audio file.
        model_version (str): Model version ['chord', 'chord-large', 'stereo-chord', 'stereo-chord-large'].
        beat_sync_threshold (float): Threshold for beat synchronization.
        upscale (bool): Whether to upscale audio to 48 kHz.
        mix_weight (float): Weight for instrumental track in mixing.
        output_path (str): Directory to save output files.
    """
    logger = logging.getLogger("postprocessor")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s:%(message)s"))
    logger.addHandler(handler)

    # Validate inputs
    if not Path(audio_path).is_file():
        raise ValueError(f"Audio file {audio_path} does not exist")
    if model_version not in ["chord", "chord-large", "stereo-chord", "stereo-chord-large"]:
        raise ValueError(f"Invalid model_version: {model_version}")

    # Create output directories
    Path(output_path).mkdir(parents=True, exist_ok=True)
    (Path(output_path) / "inter_process").mkdir(parents=True, exist_ok=True)

    # Determine channels
    channel = 2 if "stereo" in model_version else 1

    # BPM and downbeat analysis
    logger.info("Getting BPM and downbeat analysis of input audio...")
    try:
        time_analysis_url = replicate.run(
            "sakemin/all-in-one-music-structure-analyzer:001b4137be6ac67bdc28cb5cffacf128b874f530258d033de23121e785cb7290",
            input={"music_input": Path(audio_path)},
        )
        with urllib.request.urlopen(time_analysis_url[0]) as url:
            time_analysis = json.load(url)
    except Exception as e:
        raise RuntimeError(f"Failed to analyze BPM and downbeats: {e}")

    bpm = time_analysis.get("bpm")
    input_downbeats = time_analysis.get("downbeats", [])
    logger.info(f"BPM: {bpm}, Downbeats: {input_downbeats}")

    # Set beat_sync_threshold
    if beat_sync_threshold is None or beat_sync_threshold == -1:
        beat_sync_threshold = 1.1 / (bpm / 60) if bpm else 0.75
        logger.info(f"Setting beat_sync_threshold to {beat_sync_threshold}")

    # Separate vocal and instrumental tracks
    logger.info("Separating vocal track from instrumental...")
    try:
        track_urls = replicate.run(
            "cjwbw/demucs:25a173108cff36ef9f80f854c162d01df9e6528be175794b81158fa03836d953",
            input={
                "audio": Path(audio_path),
                "stem": "vocals",
                "shifts": 2,
                "float32": True,
                "output_format": "mp3",
            },
        )
    except Exception as e:
        raise RuntimeError(f"Failed to separate tracks: {e}")

    # Process vocals
    vocal_track, vocal_sr = download_audio_and_load_as_numpy(track_urls["vocals"])
    vocal_path = (
        Path(output_path) / "inter_process" / f"{Path(audio_path).stem}_vocals.mp3"
    )
    save_numpy_as_audio(vocal_track, vocal_sr, str(vocal_path))
    logger.info(f"Saved vocal track to {vocal_path}")

    # Process instrumental
    instrumental_track, instrumental_sr = download_audio_and_load_as_numpy(track_urls["other"])
    instrumental_path = (
        Path(output_path) / "inter_process" / f"{Path(audio_path).stem}_inst.mp3"
    )
    save_numpy_as_audio(instrumental_track, instrumental_sr, str(instrumental_path))
    logger.info(f"Saved instrumental track to {instrumental_path}")

    # Generate new instrumental
    logger.info("Generating new instrumental track...")
    try:
        generated_instrumental_url = replicate.run(
            "sakemin/musicgen-stereo-chord:fbdc5ef7200220ed300015d9b4fd3f8e620f84547e970b23aa2be7f2ff366a5b",
            input={
                "model_version": model_version,
                "prompt": f"{prompt}, bpm: {bpm}",
                "audio_chords": Path(instrumental_path),
                "duration": int(instrumental_track.shape[0] / instrumental_sr),
            },
        )
    except Exception as e:
        raise RuntimeError(f"Failed to generate instrumental: {e}")

    generated_instrumental_track, generated_instrumental_sr = download_audio_and_load_as_numpy(generated_instrumental_url)
    generated_instrumental_path = (
        Path(output_path) / "inter_process" / f"{Path(audio_path).stem}_{prompt}_generated_inst.mp3"
    )
    save_numpy_as_audio(generated_instrumental_track, generated_instrumental_sr, str(generated_instrumental_path))
    logger.info(f"Saved generated instrumental to {generated_instrumental_path}")

    # Sample rate matching
    if not upscale:
        logger.info("Resampling generated instrumental track...")
        resampled_instrumental_track = resample_audio(
            generated_instrumental_track, generated_instrumental_sr, vocal_sr
        )
        resampled_instrumental_track = int16_scale(normalize_audio(resampled_instrumental_track))
        resampled_instrumental_sr = vocal_sr
    else:
        logger.info("Upscaling tracks to 48kHz...")
        try:
            resampled_instrumental_url = replicate.run(
                "sakemin/audiosr-long-audio:44b37256d8d2ade24655f05a0d35128642ca90cbad0f5fa0e9bfa2d345124c8c",
                input={"input_file": Path(generated_instrumental_path)},
            )
            resampled_instrumental_track, resampled_instrumental_sr = download_audio_and_load_as_numpy(resampled_instrumental_url)
            resampled_vocal_url = replicate.run(
                "sakemin/audiosr-long-audio:44b37256d8d2ade24655f05a0d35128642ca90cbad0f5fa0e9bfa2d345124c8c",
                input={"input_file": Path(vocal_path)},
            )
            vocal_track, vocal_sr = download_audio_and_load_as_numpy(resampled_vocal_url)
        except Exception as e:
            raise RuntimeError(f"Failed to upscale audio: {e}")

    resampled_instrumental_path = (
        Path(output_path) / "inter_process" / f"{Path(audio_path).stem}_{prompt}_resampled_inst.mp3"
    )
    save_numpy_as_audio(resampled_instrumental_track, resampled_instrumental_sr, str(resampled_instrumental_path))
    logger.info(f"Saved resampled instrumental to {resampled_instrumental_path}")

    # Beat synchronization
    logger.info("Getting BPM and downbeat analysis of generated audio...")
    try:
        output_time_analysis_url = replicate.run(
            "sakemin/all-in-one-music-structure-analyzer:001b4137be6ac67bdc28cb5cffacf128b874f530258d033de23121e785cb7290",
            input={"music_input": Path(resampled_instrumental_path)},
        )
        with urllib.request.urlopen(output_time_analysis_url[0]) as url:
            output_time_analysis = json.load(url)
    except Exception as e:
        raise RuntimeError(f"Failed to analyze generated audio: {e}")

    generated_downbeats = output_time_analysis.get("downbeats", [])

    # Align downbeats
    logger.info("Aligning downbeats pair-wise...")
    aligned_generated_downbeats = []
    aligned_input_downbeats = []

    for generated_downbeat in generated_downbeats:
        input_beat = min(input_downbeats, key=lambda x: abs(generated_downbeat - x), default=None)
        if input_beat is None:
            continue
        if (
            aligned_input_downbeats
            and int(input_beat * vocal_sr) == aligned_input_downbeats[-1]
        ):
            logger.debug("Dropped duplicate beat")
            continue
        if abs(generated_downbeat - input_beat) > beat_sync_threshold:
            input_beat = generated_downbeat
            logger.debug(f"Replaced beat: {input_beat}")
        aligned_generated_downbeats.append(int(generated_downbeat * vocal_sr))
        aligned_input_downbeats.append(int(input_beat * vocal_sr))

    wav_length = resampled_instrumental_track.shape[0]
    downbeat_offset = aligned_input_downbeats[0] - aligned_generated_downbeats[0] if aligned_input_downbeats else 0
    if downbeat_offset > 0:
        resampled_instrumental_track = np.pad(
            resampled_instrumental_track, ((0, 0), (int(downbeat_offset), 0)), "constant"
        )
        aligned_generated_downbeats = [x + downbeat_offset for x in aligned_generated_downbeats]

    aligned_generated_downbeats = [0] + aligned_generated_downbeats + [wav_length]
    aligned_input_downbeats = [0] + aligned_input_downbeats + [wav_length]

    s_ap = ",".join(f"{g}:{i}" for g, i in zip(aligned_generated_downbeats, aligned_input_downbeats))

    # Dynamic time-stretching
    logger.info("Applying dynamic time-stretching...")
    try:
        time_stretched_instrumental_url = replicate.run(
            "sakemin/pytsmod:41b355721c8a7ed501be7fd89e73631e7c07d75e1c94b1372c1c119b0774cdae",
            input={
                "audio_input": Path(resampled_instrumental_path),
                "s_ap": s_ap,
                "absolute_frame": True,
            },
        )
    except Exception as e:
        raise RuntimeError(f"Failed to apply time-stretching: {e}")

    time_stretched_instrumental_track, time_stretched_instrumental_sr = download_audio_and_load_as_numpy(time_stretched_instrumental_url)
    time_stretched_instrumental_path = (
        Path(output_path) / "inter_process" / f"{Path(audio_path).stem}_{prompt}_time_stretched_inst.mp3"
    )
    save_numpy_as_audio(
        time_stretched_instrumental_track,
        time_stretched_instrumental_sr,
        str(time_stretched_instrumental_path),
    )
    logger.info(f"Saved time-stretched instrumental to {time_stretched_instrumental_path}")

    # Combine tracks
    pad = vocal_track.shape[0] - time_stretched_instrumental_track.shape[0]
    if pad > 0:
        padded_instrumental_track = np.pad(
            time_stretched_instrumental_track, ((0, pad), (0, 0)), "constant"
        )
    else:
        padded_instrumental_track = time_stretched_instrumental_track[: vocal_track.shape[0]]

    if channel == 1 and vocal_track.shape[1] == 2:
        padded_instrumental_track = np.repeat(padded_instrumental_track, 2, axis=1)
    elif channel == 2 and vocal_track.shape[1] == 1:
        vocal_track = np.repeat(vocal_track, 2, axis=1)

    logger.info("Mixing and normalizing tracks...")
    mixed_track = mix_audio_volumes(padded_instrumental_track, vocal_track, mix_weight, 1 - mix_weight)

    remixed_path = Path(output_path) / f"{Path(audio_path).stem}_{prompt}_remixed.mp3"
    save_numpy_as_audio(mixed_track, time_stretched_instrumental_sr, str(remixed_path))
    logger.info(f"Saved remixed track to {remixed_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MusicGen Remixer")
    parser.add_argument("--prompt", required=True, help="Prompt for generating the remix")
    parser.add_argument("--audio_path", required=True, help="Path to the audio file to remix")
    parser.add_argument(
        "--model_version",
        default="chord",
        choices=["chord", "chord-large", "stereo-chord", "stereo-chord-large"],
        help="Model version for generating the remix"
    )
    parser.add_argument(
        "--beat_sync_threshold",
        type=float,
        default=None,
        help="Threshold for beat synchronization (auto-set if None)"
    )
    parser.add_argument(
        "--upscale",
        action="store_true",
        help="Upscale audio to 48 kHz"
    )
    parser.add_argument(
        "--mix_weight",
        type=float,
        default=0.65,
        help="Weight for instrumental track (0 to 1)"
    )
    parser.add_argument(
        "--output_path",
        default="output",
        help="Directory to save output files"
    )
    args = parser.parse_args()
    main(
        args.prompt,
        args.audio_path,
        args.model_version,
        args.beat_sync_threshold,
        args.upscale,
        args.mix_weight,
        args.output_path,
    )

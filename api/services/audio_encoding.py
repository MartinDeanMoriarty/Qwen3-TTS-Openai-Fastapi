# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Audio encoding service for TTS API.
Handles conversion of raw audio to various formats (mp3, opus, aac, flac, wav, pcm).
"""

import asyncio
import io
import logging
import struct
from typing import Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)

AudioFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]

# Default sample rate for Qwen3-TTS output
DEFAULT_SAMPLE_RATE = 24000


def get_content_type(audio_format: AudioFormat) -> str:
    """Get MIME content type for audio format."""
    content_types = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }
    return content_types.get(audio_format, f"audio/{audio_format}")


def convert_to_wav(
    audio: np.ndarray,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    num_channels: int = 1,
    bits_per_sample: int = 16,
) -> bytes:
    """
    Convert numpy audio array to WAV format bytes.
    
    Args:
        audio: Audio data as numpy array (float32 normalized to [-1, 1])
        sample_rate: Sample rate in Hz
        num_channels: Number of audio channels
        bits_per_sample: Bits per sample (8 or 16)
    
    Returns:
        WAV file bytes
    """
    # Ensure audio is float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    # Normalize if needed
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val
    
    # Convert to int16
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Create WAV header
    num_samples = len(audio_int16)
    bytes_per_sample = bits_per_sample // 8
    byte_rate = sample_rate * num_channels * bytes_per_sample
    block_align = num_channels * bytes_per_sample
    data_size = num_samples * bytes_per_sample
    
    buffer = io.BytesIO()
    
    # RIFF header
    buffer.write(b'RIFF')
    buffer.write(struct.pack('<I', 36 + data_size))  # File size - 8
    buffer.write(b'WAVE')
    
    # Format chunk
    buffer.write(b'fmt ')
    buffer.write(struct.pack('<I', 16))  # Chunk size
    buffer.write(struct.pack('<H', 1))  # Audio format (PCM)
    buffer.write(struct.pack('<H', num_channels))
    buffer.write(struct.pack('<I', sample_rate))
    buffer.write(struct.pack('<I', byte_rate))
    buffer.write(struct.pack('<H', block_align))
    buffer.write(struct.pack('<H', bits_per_sample))
    
    # Data chunk
    buffer.write(b'data')
    buffer.write(struct.pack('<I', data_size))
    buffer.write(audio_int16.tobytes())
    
    return buffer.getvalue()


def convert_to_pcm(
    audio: np.ndarray,
    bits_per_sample: int = 16,
) -> bytes:
    """
    Convert numpy audio array to raw PCM bytes.
    
    Args:
        audio: Audio data as numpy array (float32 normalized to [-1, 1])
        bits_per_sample: Bits per sample (8 or 16)
    
    Returns:
        Raw PCM bytes
    """
    # Ensure audio is float32
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    
    # Normalize if needed
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val
    
    # Convert to int16
    audio_int16 = (audio * 32767).astype(np.int16)
    
    return audio_int16.tobytes()


def encode_audio(
    audio: np.ndarray,
    format: AudioFormat = "mp3",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> bytes:
    """
    Encode audio to the specified format.
    
    Args:
        audio: Audio data as numpy array (float32 normalized to [-1, 1])
        format: Target audio format
        sample_rate: Sample rate in Hz
    
    Returns:
        Encoded audio bytes
    """
    if format == "wav":
        return convert_to_wav(audio, sample_rate)
    
    if format == "pcm":
        return convert_to_pcm(audio)
    
    # For compressed formats, use pydub if available, otherwise fall back to wav
    try:
        from pydub import AudioSegment
        
        # Convert to WAV first
        wav_bytes = convert_to_wav(audio, sample_rate)
        
        # Load into pydub
        segment = AudioSegment.from_wav(io.BytesIO(wav_bytes))
        
        # Export to target format
        output = io.BytesIO()
        
        format_params = {
            "mp3": {"format": "mp3", "bitrate": "192k"},
            "opus": {"format": "opus", "bitrate": "128k"},
            "aac": {"format": "adts", "bitrate": "192k"},  # AAC in ADTS container
            "flac": {"format": "flac"},
        }
        
        params = format_params.get(format, {"format": format})
        export_format = params.pop("format", format)
        
        segment.export(output, format=export_format, **params)
        return output.getvalue()
        
    except ImportError:
        # Fall back to WAV if pydub not available
        logger.warning(f"pydub not available, returning WAV instead of {format}")
        return convert_to_wav(audio, sample_rate)
    except Exception as e:
        # Fall back to WAV on any encoding error
        logger.warning(f"Failed to encode to {format} ({e}), returning WAV")
        return convert_to_wav(audio, sample_rate)


def _pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio.astype(np.float32), -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def _streaming_wav_header(sample_rate: int) -> bytes:
    """WAV header with unknown length (0xFFFFFFFF), as streaming players expect."""
    return (
        b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data" + struct.pack("<I", 0xFFFFFFFF)
    )


_FFMPEG_STREAM_ARGS = {
    "mp3": ["-f", "mp3", "-b:a", "192k"],
    "opus": ["-f", "ogg", "-c:a", "libopus", "-b:a", "128k"],
    "aac": ["-f", "adts", "-c:a", "aac", "-b:a", "192k"],
    "flac": ["-f", "flac"],
}


async def encode_audio_streaming(
    audio_chunks,
    format: AudioFormat = "mp3",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
):
    """
    Encode an async stream of float audio chunks into one continuous audio stream.

    PCM and WAV are written directly. Compressed formats go through a single
    ffmpeg process for the whole stream: encoding chunks separately would put
    encoder padding, i.e. small gaps, at every seam.

    Args:
        audio_chunks: Async iterator of float32 numpy chunks in [-1, 1]
        format: Target audio format
        sample_rate: Sample rate in Hz

    Yields:
        Encoded bytes as they become available
    """
    if format in ("pcm", "wav"):
        if format == "wav":
            yield _streaming_wav_header(sample_rate)
        async for chunk in audio_chunks:
            if len(chunk):
                yield _pcm16(chunk)
        return

    # Without the probing/buffering flags ffmpeg sat on the first 160 ms of
    # input for over 2 s; with them the first bytes come out after ~50 ms.
    process = await asyncio.create_subprocess_exec(
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-probesize", "32", "-analyzeduration", "0", "-fflags", "nobuffer",
        "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
        *_FFMPEG_STREAM_ARGS[format], "-flush_packets", "1", "pipe:1",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
    )

    async def feed():
        try:
            async for chunk in audio_chunks:
                if len(chunk):
                    process.stdin.write(_pcm16(chunk))
                    await process.stdin.drain()
        finally:
            process.stdin.close()

    feeder = asyncio.create_task(feed())
    try:
        while True:
            data = await process.stdout.read(4096)
            if not data:
                break
            yield data
        await feeder  # re-raise generation errors
    finally:
        if not feeder.done():
            feeder.cancel()
        if process.returncode is None:
            process.kill()
            await process.wait()

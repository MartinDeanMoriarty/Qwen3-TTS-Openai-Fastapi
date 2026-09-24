# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Tests for text segmentation, the streaming audio encoder and the streaming endpoint.
"""

import base64
import json
import shutil
import struct

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.services.audio_encoding import encode_audio_streaming
from api.services.text_processing import split_into_segments


class TestSplitIntoSegments:
    def test_short_text_is_one_segment(self):
        assert split_into_segments("Hallo Welt.", 100) == ["Hallo Welt."]

    def test_empty_text_has_no_segments(self):
        assert split_into_segments("   ", 100) == []

    def test_sentences_are_packed_up_to_the_limit(self):
        text = "Erster Satz hier. Zweiter Satz hier. Dritter Satz hier."
        segments = split_into_segments(text, 40)
        assert segments == ["Erster Satz hier. Zweiter Satz hier.", "Dritter Satz hier."]
        assert " ".join(segments) == text

    def test_long_sentence_breaks_at_clauses_then_spaces(self):
        text = "Eins zwei drei, vier fünf sechs, " + "wort " * 30
        segments = split_into_segments(text.strip(), 30)
        assert all(len(s) <= 30 for s in segments)
        assert " ".join(segments).split() == text.split()


async def _chunks(*arrays):
    for a in arrays:
        yield a


async def _collect(agen):
    return b"".join([data async for data in agen])


class TestStreamingEncoder:
    @pytest.mark.asyncio
    async def test_pcm_is_concatenated_int16(self):
        a, b = np.full(10, 0.5, np.float32), np.full(5, -0.5, np.float32)
        data = await _collect(encode_audio_streaming(_chunks(a, b), "pcm", 24000))
        samples = np.frombuffer(data, dtype=np.int16)
        assert len(samples) == 15
        assert samples[0] == int(0.5 * 32767) and samples[-1] == int(-0.5 * 32767)

    @pytest.mark.asyncio
    async def test_wav_has_streaming_header(self):
        data = await _collect(encode_audio_streaming(_chunks(np.zeros(100, np.float32)), "wav", 24000))
        assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
        assert struct.unpack("<I", data[24:28])[0] == 24000
        assert len(data) == 44 + 200

    @pytest.mark.asyncio
    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
    async def test_mp3_is_one_continuous_stream(self):
        tone = np.sin(np.arange(24000) / 24000 * 2 * np.pi * 440).astype(np.float32) * 0.3
        data = await _collect(encode_audio_streaming(_chunks(tone[:12000], tone[12000:]), "mp3", 24000))
        assert len(data) > 1000
        # MPEG audio frame sync (possibly after an ID3 tag)
        assert data[:3] == b"ID3" or data[0] == 0xFF


class FakeBackend:
    """Streams three chunks without a model."""

    def is_ready(self):
        return True

    async def generate_speech_stream(self, text, voice, language="Auto", instruct=None, speed=1.0):
        for value in (0.1, 0.2, 0.3):
            yield np.full(240, value, np.float32), 24000


@pytest.fixture
def client(monkeypatch):
    from api.main import app
    from api.routers import openai_compatible

    async def fake_backend():
        return FakeBackend()

    monkeypatch.setattr(openai_compatible, "get_tts_backend", fake_backend)
    return TestClient(app)


class TestStreamingEndpoint:
    def test_stream_true_returns_audio_chunks(self, client):
        response = client.post("/v1/audio/speech", json={
            "model": "tts-1", "input": "Hallo Welt", "voice": "Jarvis", "response_format": "pcm", "stream": True})
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/pcm"
        assert len(response.content) == 3 * 240 * 2

    def test_sse_stream_format(self, client):
        response = client.post("/v1/audio/speech", json={
            "model": "tts-1", "input": "Hallo Welt", "voice": "Jarvis", "response_format": "pcm",
            "stream_format": "sse"})
        assert response.status_code == 200
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        assert events[-1] == {"type": "speech.audio.done"}
        audio = b"".join(base64.b64decode(e["audio"]) for e in events if e["type"] == "speech.audio.delta")
        assert len(audio) == 3 * 240 * 2

# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Custom voice samples for voice cloning with the Base model.

Every WAV/MP3 in the samples directory becomes a voice named after its file.
The clone prompt of a voice (its speaker embedding) is computed once and kept
on the CPU, so it survives a model unload and costs nothing on later requests.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

VOICE_SAMPLES_DIR = os.environ.get("VOICE_SAMPLES_DIR", "/app/voice-samples")


class VoiceLibrary:
    """Voice name lookup plus a cache of x-vector clone prompts."""

    def __init__(self, directory: str = VOICE_SAMPLES_DIR):
        self.directory = directory
        self._voices: Dict[str, dict] = {}
        self._prompts: Dict[str, Any] = {}

    def scan(self) -> None:
        """Build the name -> sample mapping from the samples directory."""
        self._voices = {}
        if not os.path.exists(self.directory):
            logger.info(f"Voice samples directory not found: {self.directory}")
            return

        samples_path = Path(self.directory)
        for ext in ["*.wav", "*.mp3"]:
            for audio_file in samples_path.glob(ext):
                # "General_Joe.wav" answers to "General_Joe", "General Joe" and lowercase forms
                voice_key = audio_file.stem
                voice_display = voice_key.replace("_", " ")
                # Prefer .wav over .mp3 if both exist
                if voice_key not in self._voices or audio_file.suffix == ".wav":
                    info = {"path": str(audio_file), "name": voice_display, "key": voice_key}
                    for alias in (voice_key, voice_key.lower(), voice_display, voice_display.lower()):
                        self._voices[alias] = info

        if self._voices:
            logger.info(f"Loaded {len(self.names())} custom voice samples from {self.directory}")

    def names(self) -> List[str]:
        return sorted(set(v["name"] for v in self._voices.values()))

    def find(self, voice_name: str) -> Optional[dict]:
        """Voice info by name: exact, lowercase, underscores, then first name only."""
        for candidate in (voice_name, voice_name.lower(), voice_name.replace(" ", "_")):
            if candidate in self._voices:
                return self._voices[candidate]
        first_name = voice_name.split()[0] if " " in voice_name else voice_name
        return self._voices.get(first_name.lower())

    def clone_prompt(self, model, voice_info: dict):
        """The x-vector clone prompt of a voice, computed on first use.

        `model` is the Qwen3TTSModel wrapper. The prompt tensors live on the CPU
        and are moved to the GPU by the model when it builds its inputs.
        """
        path = voice_info["path"]
        prompt = self._prompts.get(path)
        if prompt is None:
            items = model.create_voice_clone_prompt(ref_audio=path, x_vector_only_mode=True)
            for item in items:
                item.ref_spk_embedding = item.ref_spk_embedding.detach().cpu()
            prompt = self._prompts[path] = items
        return prompt

    def precompute(self, model) -> None:
        """Compute the clone prompt of every voice up front."""
        for name in self.names():
            self.clone_prompt(model, self.find(name))

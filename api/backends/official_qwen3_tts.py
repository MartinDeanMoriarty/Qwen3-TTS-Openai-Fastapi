# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Official Qwen3-TTS backend implementation.

This backend uses the official Qwen3-TTS Python implementation
from the qwen_tts package. Supports both CustomVoice and Base models.

- CustomVoice model: Uses built-in premium voices
- Base model: Supports voice cloning from reference audio samples
"""

import logging
import os
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

from .base import TTSBackend, run_on_gpu_thread
from ..services.text_processing import split_into_segments
from ..services.voices import VoiceLibrary

logger = logging.getLogger(__name__)

# Longer inputs are synthesized in sentence groups of at most this many
# characters, which bounds the talker's sequence length per generation.
MAX_SEGMENT_CHARS = int(os.getenv("TTS_MAX_SEGMENT_CHARS", "400"))

# Optional librosa import for speed adjustment
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


class OfficialQwen3TTSBackend(TTSBackend):
    """Official Qwen3-TTS backend using the qwen_tts package."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"):
        """
        Initialize the official backend.
        
        Args:
            model_name: HuggingFace model identifier
        """
        super().__init__()
        self.model_name = model_name
        self._ready = False
        self.voices = VoiceLibrary()
        self._is_base_model = "Base" in model_name
    
    async def initialize(self) -> None:
        """Initialize the backend and load the model."""
        if self._ready:
            logger.info("Official backend already initialized")
            return

        try:
            await run_on_gpu_thread(self._load)
        except Exception as e:
            logger.error(f"Failed to load official TTS backend: {e}")
            raise RuntimeError(f"Failed to initialize official TTS backend: {e}")

    def _load(self) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        if torch.cuda.is_available():
            self.device = "cuda:0"
            self.dtype = torch.bfloat16
            # Keep cuDNN autotuning off. The speech-tokenizer convolutions see a
            # new input length on every request, and with benchmark mode each
            # new length re-runs the algorithm search: measured +8.4 s per
            # request and +10 s per voice prompt on an RTX 4070 Ti.
            torch.backends.cudnn.benchmark = False
        else:
            self.device = "cpu"
            self.dtype = torch.float32

        logger.info(f"Loading Qwen3-TTS model '{self.model_name}' on {self.device}...")
        logger.info(f"Model type: {'Base (voice cloning)' if self._is_base_model else 'CustomVoice (built-in voices)'}")

        attn_impl = "sdpa"
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            pass
        logger.info(f"Attention implementation: {attn_impl}")

        self.model = Qwen3TTSModel.from_pretrained(
            self.model_name,
            device_map=self.device,
            dtype=self.dtype,
            attn_implementation=attn_impl,
        )

        self.voices.scan()
        if self._is_base_model:
            self.voices.precompute(self.model)

        self._ready = True
        logger.info(f"Official Qwen3-TTS backend loaded successfully on {self.device}")

    def _prepare(self, text: str, voice: str, language: str, instruct: Optional[str]) -> Dict[str, Any]:
        """Model inputs for one request: a cloned sample voice, else a built-in speaker."""
        custom_voice = self.voices.find(voice)
        if custom_voice:
            logger.info(f"Using voice cloning with sample: {custom_voice['name']}")
            return self.model._prepare_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=self.voices.clone_prompt(self.model, custom_voice),
            )

        if self._is_base_model:
            # Base model doesn't have built-in named voices
            available = self.voices.names()
            logger.error(f"Voice '{voice}' not found. Available custom voices: {available}")
            raise ValueError(
                f"Voice '{voice}' not found. Using Base model which requires custom voice samples. "
                f"Available voices: {available}"
            )

        return self.model._prepare_custom_voice(text=text, speaker=voice, language=language, instruct=instruct)

    def _codes(self, inputs: Dict[str, Any]):
        """Codec frames [T, 16] for prepared inputs, via transformers generate()."""
        codes_list, _ = self.model.model.generate(**inputs, **self.model._merge_generate_kwargs())
        return codes_list[0]

    def _decode(self, codes, ref_codes) -> Tuple[np.ndarray, int]:
        """Codec frames -> waveform via the speech tokenizer."""
        wavs, sr = self.model._decode_codes([codes], ref_codes)
        return wavs[0], sr

    def _generate(self, text: str, voice: str, language: str, instruct: Optional[str]) -> Tuple[np.ndarray, int]:
        audio, sr = [], None
        for segment in split_into_segments(text, MAX_SEGMENT_CHARS):
            inputs = self._prepare(segment, voice, language, instruct)
            ref_codes = inputs.get("voice_clone_prompt", {}).get("ref_code")
            wav, sr = self._decode(self._codes(inputs), ref_codes)
            audio.append(wav)
        if not audio:
            raise ValueError("Nothing to synthesize")
        return np.concatenate(audio), sr

    async def generate_speech(
        self,
        text: str,
        voice: str,
        language: str = "Auto",
        instruct: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech from text.

        If voice matches a custom sample, uses voice cloning.
        Otherwise uses built-in voices (CustomVoice model) or fails (Base model).
        """
        if not self._ready:
            await self.initialize()

        try:
            audio, sr = await run_on_gpu_thread(self._generate, text, voice, language, instruct)
        except Exception as e:
            logger.error(f"Speech generation failed: {e}")
            raise RuntimeError(f"Speech generation failed: {e}")

        if speed != 1.0 and LIBROSA_AVAILABLE:
            audio = librosa.effects.time_stretch(audio.astype(np.float32), rate=speed)
        return audio, sr

    def get_backend_name(self) -> str:
        """Return the name of this backend."""
        return "official"
    
    def get_model_id(self) -> str:
        """Return the model identifier."""
        return self.model_name
    
    def get_supported_voices(self) -> List[str]:
        """Return list of supported voice names."""
        voices = []
        
        # Add custom voices
        voices.extend(self.voices.names())
        
        # Add built-in voices if using CustomVoice model
        if not self._is_base_model:
            builtin = ["Vivian", "Ryan", "Serena", "Aiden", "Dylan", "Eric", 
                      "Uncle_Fu", "Ono_Anna", "Sohee"]
            voices.extend(builtin)
        
        return voices
    
    def get_supported_languages(self) -> List[str]:
        """Return list of supported language names."""
        return ["English", "Chinese", "Japanese", "Korean", "German", "French", 
                "Spanish", "Russian", "Portuguese", "Italian", "Auto"]
    
    def is_ready(self) -> bool:
        """Return whether the backend is initialized and ready."""
        return self._ready
    
    async def unload(self) -> None:
        """
        Unload the model from GPU memory to free VRAM.

        The voice clone prompts stay cached on the CPU, so the next request
        after a reload does not have to compute them again.
        """
        if not self._ready:
            logger.info("Backend not loaded, nothing to unload")
            return

        logger.info("Unloading Qwen3-TTS model from GPU memory...")
        await run_on_gpu_thread(self._release)
        logger.info("Model unloaded successfully - VRAM freed")

    def _release(self) -> None:
        import gc
        import torch

        self._ready = False
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            # cuBLAS keeps a workspace per stream inside the caching allocator;
            # left in place they pin the segments the weights lived in, and
            # empty_cache() then released nothing (4 GB stayed reserved)
            if hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
                torch._C._cuda_clearCublasWorkspaces()
            torch.cuda.empty_cache()

    def get_device_info(self) -> Dict[str, Any]:
        """Return device information."""
        info = {
            "device": str(self.device) if self.device else "unknown",
            "gpu_available": False,
            "gpu_name": None,
            "vram_total": None,
            "vram_used": None,
            "model_type": "Base (voice cloning)" if self._is_base_model else "CustomVoice",
            "custom_voices_count": len(self.voices.names()),
        }
        
        try:
            import torch
            
            if torch.cuda.is_available():
                info["gpu_available"] = True
                if torch.cuda.current_device() >= 0:
                    device_idx = torch.cuda.current_device()
                    info["gpu_name"] = torch.cuda.get_device_name(device_idx)
                    
                    props = torch.cuda.get_device_properties(device_idx)
                    info["vram_total"] = f"{props.total_memory / 1024**3:.2f} GB"
                    
                    if self._ready:
                        allocated = torch.cuda.memory_allocated(device_idx)
                        info["vram_used"] = f"{allocated / 1024**3:.2f} GB"
        except Exception as e:
            logger.warning(f"Could not get device info: {e}")
        
        return info

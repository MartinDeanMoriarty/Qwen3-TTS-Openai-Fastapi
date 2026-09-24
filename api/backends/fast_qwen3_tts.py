# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Fast Qwen3-TTS backend: the official model with a CUDA-graph decode loop.

Same weights, same prompt construction and same speech-tokenizer decoding as
the official backend; only the autoregressive loop is replaced by
`qwen_tts.inference.fast_generator.FastGenerator`. It also streams: audio is
decoded chunk by chunk while generation continues. Without a GPU it falls
back to the official generate() path.
"""

import logging
import os
from typing import Any, Dict, Iterator, Optional

import numpy as np
import torch

from .base import stream_on_gpu_thread
from .official_qwen3_tts import LIBROSA_AVAILABLE, MAX_SEGMENT_CHARS, OfficialQwen3TTSBackend
from ..services.text_processing import split_into_segments

logger = logging.getLogger(__name__)

# Talker KV-cache length. 2048 positions hold ~160 s of audio per segment.
MAX_SEQ_LEN = int(os.getenv("TTS_MAX_SEQ_LEN", "2048"))
# Frames per streamed chunk (80 ms each); the last entry repeats. Small first
# chunks give early audio, growing ones keep ahead of playback.
STREAM_CHUNK_FRAMES = tuple(int(n) for n in os.getenv("TTS_STREAM_CHUNK_FRAMES", "2,4,8,12").split(","))
# Earlier frames decoded along with each chunk so the seams match a full decode
STREAM_LEFT_CONTEXT = int(os.getenv("TTS_STREAM_LEFT_CONTEXT", "72"))
# Fuse the decode-step forwards with torch.compile (~24% faster steps on an
# RTX 4070 Ti). Costs ~40 s at the first start, ~10 s with a warm Inductor
# cache (TORCHINDUCTOR_CACHE_DIR on a volume).
COMPILE = os.getenv("TTS_COMPILE", "true").lower() == "true"
# "int8": int8 weights with one scale per output row for the talker and
# code-predictor transformer layers (activations stay bf16; embeddings and
# output heads are untouched). Decoding at batch size 1 is bound by reading
# the weights, so halving them speeds up every step.
QUANTIZE = os.getenv("TTS_QUANTIZE", "none").lower()


class FastQwen3TTSBackend(OfficialQwen3TTSBackend):
    """Official Qwen3-TTS model driven by a captured CUDA graph per decode step."""

    def __init__(self, model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"):
        super().__init__(model_name=model_name)
        self.generator = None
        self.stream_decoder = None

    def _load(self) -> None:
        super()._load()
        if not str(self.device).startswith("cuda"):
            logger.warning("No CUDA device: fast backend uses the official generate() path")
            return

        from qwen_tts.inference.fast_generator import FastGenerator, StreamingDecoder

        if QUANTIZE != "none":
            self._quantize(QUANTIZE)

        self.generator = FastGenerator(self.model, max_seq_len=MAX_SEQ_LEN, compile=COMPILE)
        self.generator.warmup()
        self.stream_decoder = StreamingDecoder(self.model.model.speech_tokenizer, left_context=STREAM_LEFT_CONTEXT)
        self.stream_decoder.capture()
        logger.info(f"CUDA graphs captured (max_seq_len={MAX_SEQ_LEN}, compile={COMPILE}, quantize={QUANTIZE})")
        if COMPILE:
            # Inductor's compile-worker pool is only needed while compiling. Left
            # running it holds ~650 MB of RAM, and at exit the server waited
            # for it long enough that Docker killed the container. Inductor
            # starts a new pool if anything has to be compiled later.
            from torch._inductor.async_compile import shutdown_compile_workers

            shutdown_compile_workers()

    def _quantize(self, mode: str) -> None:
        if mode == "int8":
            from qwen_tts.inference.int8_linear import quantize_linears_int8

            talker = self.model.model.talker
            # Transformer layers only; embeddings and output heads stay bf16
            count = sum(quantize_linears_int8(layers)
                        for layers in (talker.model.layers, talker.code_predictor.model.layers))
            torch.cuda.empty_cache()
            logger.info(f"{count} linear layers quantized to int8 weights")
        else:
            raise ValueError(f"Unknown TTS_QUANTIZE: {mode} (supported: none, int8)")

    def _codes(self, inputs: Dict[str, Any]):
        if self.generator is None:
            return super()._codes(inputs)
        return self.generator.generate_codes(inputs)

    def _decode(self, codes, ref_codes):
        # Short utterances fit one of the decoder's graph buckets
        short = len(codes) <= self.stream_decoder.graph_buckets[-1] if self.stream_decoder else False
        if not short or (ref_codes and ref_codes[0] is not None):
            return super()._decode(codes, ref_codes)
        wav = self.stream_decoder.decode(codes).float().cpu().numpy()
        return wav, self.model.model.speech_tokenizer.get_output_sample_rate()

    def _stream(self, text: str, voice: str, language: str, instruct: Optional[str]) -> Iterator[np.ndarray]:
        """Audio chunks of one request, generated and decoded incrementally."""
        for segment in split_into_segments(text, MAX_SEGMENT_CHARS):
            inputs = self._prepare(segment, voice, language, instruct)
            # ICL voice prompts carry reference codes that serve as decoder context
            prefix = (inputs.get("voice_clone_prompt", {}).get("ref_code") or [None])[0]
            if prefix is not None:
                prefix = prefix.to(self.generator.device)
            self.stream_decoder.reset(prefix_frames=0 if prefix is None else len(prefix))

            frames = 0
            for chunk in self.generator.stream_codes(inputs, chunk_frames=STREAM_CHUNK_FRAMES):
                frames += len(chunk)
                codes = self.generator.codes[:frames]
                if prefix is not None:
                    codes = torch.cat([prefix, codes])
                audio = self.stream_decoder.push(codes)
                if len(audio):
                    yield audio

    async def generate_speech_stream(
        self,
        text: str,
        voice: str,
        language: str = "Auto",
        instruct: Optional[str] = None,
        speed: float = 1.0,
    ):
        if not self._ready:
            await self.initialize()
        if self.generator is None:
            async for item in super().generate_speech_stream(text, voice, language, instruct, speed):
                yield item
            return

        sample_rate = self.model.model.speech_tokenizer.get_output_sample_rate()
        async for audio in stream_on_gpu_thread(self._stream, text, voice, language, instruct):
            if speed != 1.0 and LIBROSA_AVAILABLE:
                import librosa

                audio = librosa.effects.time_stretch(audio, rate=speed)
            yield audio, sample_rate

    def _release(self) -> None:
        # The graphs and static caches hold VRAM too
        self.generator = None
        self.stream_decoder = None
        super()._release()

    def get_backend_name(self) -> str:
        return "fast"

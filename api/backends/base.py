# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Base class for TTS backends.
"""

import asyncio
import functools
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

# All GPU work runs on this one thread. A single worker keeps requests in
# arrival order (Open WebUI sends one request per sentence, and the first
# sentence must not wait behind the third) and keeps the event loop free for
# health checks and streaming writes while the model is busy.
_GPU_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-gpu")


async def run_on_gpu_thread(fn, *args, **kwargs):
    """Run a blocking GPU call on the dedicated worker thread."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_GPU_EXECUTOR, functools.partial(fn, *args, **kwargs))


async def stream_on_gpu_thread(fn, *args, **kwargs):
    """Run a blocking generator on the GPU thread and yield its items as they come.

    The whole generator is one job on the GPU thread, so no other request can
    slip in between two of its items (the fast generator keeps per-request
    state on the GPU). Closing the async iterator, e.g. when the client
    disconnects, stops the generator before its next item.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    stop = threading.Event()
    done = object()

    def put(item):
        try:
            loop.call_soon_threadsafe(queue.put_nowait, item)
        except RuntimeError:  # event loop already closed
            stop.set()

    def produce():
        try:
            for item in fn(*args, **kwargs):
                if stop.is_set():
                    break
                put(item)
        except BaseException as exc:  # handed to the consumer
            put(exc)
        finally:
            put(done)

    job = loop.run_in_executor(_GPU_EXECUTOR, produce)
    try:
        while True:
            item = await queue.get()
            if item is done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        # Let the producer finish its current item before the GPU thread is
        # reused, and before the caller's event loop may go away
        await asyncio.wait([job])


class TTSBackend(ABC):
    """Abstract base class for TTS backends."""
    
    def __init__(self):
        """Initialize the backend."""
        self.model = None
        self.device = None
        self.dtype = None
    
    async def unload(self) -> None:
        """
        Unload the model from memory to free GPU VRAM.
        
        Override in subclasses for proper cleanup.
        Default implementation just sets model to None.
        """
        self.model = None
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the backend and load the model.
        
        This method should:
        - Load the model
        - Set up device and dtype
        - Perform any necessary warmup
        """
        pass
    
    @abstractmethod
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
        
        Args:
            text: The text to synthesize
            voice: Voice name/identifier to use
            language: Language code (e.g., "English", "Chinese", "Auto")
            instruct: Optional instruction for voice style/emotion
            speed: Speech speed multiplier (0.25 to 4.0)
        
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        pass
    
    async def generate_speech_stream(
        self,
        text: str,
        voice: str,
        language: str = "Auto",
        instruct: Optional[str] = None,
        speed: float = 1.0,
    ):
        """
        Yield (audio_chunk, sample_rate) as audio becomes available.

        Backends without incremental generation yield the whole utterance as
        one chunk.
        """
        yield await self.generate_speech(text, voice, language, instruct, speed)

    @abstractmethod
    def get_backend_name(self) -> str:
        """Return the name of this backend."""
        pass
    
    @abstractmethod
    def get_model_id(self) -> str:
        """Return the model identifier."""
        pass
    
    @abstractmethod
    def get_supported_voices(self) -> List[str]:
        """Return list of supported voice names."""
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """Return list of supported language names."""
        pass
    
    @abstractmethod
    def is_ready(self) -> bool:
        """Return whether the backend is initialized and ready."""
        pass
    
    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """
        Return device information.
        
        Returns:
            Dict with keys: device, gpu_available, gpu_name, vram_total, vram_used
        """
        pass

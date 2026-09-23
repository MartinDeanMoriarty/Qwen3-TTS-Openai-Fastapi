#!/usr/bin/env python3
# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""In-process latency benchmark of a TTS backend, without HTTP in between.

Runs inside the GPU container. The backend is chosen the same way the server
chooses it (TTS_BACKEND, TTS_MODEL_NAME), so this measures exactly what the
server would run.

    docker exec -w /tmp qwen3-tts-dev python /app/benchmarks/bench_engine.py --label engine-official
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import TEXTS, save_results, summarize  # noqa: E402


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


async def run(args):
    from api.backends.factory import get_backend

    backend = get_backend()
    results = {"config": vars(args), "backend": backend.get_backend_name(), "model": backend.get_model_id(), "cases": {}}

    t0 = time.perf_counter()
    await backend.initialize()
    sync()
    results["load_s"] = time.perf_counter() - t0
    print(f"backend {results['backend']}  model {results['model']}")
    print(f"load (incl. warmup/capture): {results['load_s']:.2f}s")

    t0 = time.perf_counter()
    await backend.generate_speech(text="Hallo.", voice=args.voice, language="German")
    sync()
    results["first_request_s"] = time.perf_counter() - t0
    print(f"first request (voice prompt + lazy init): {results['first_request_s']:.2f}s")

    stream = getattr(backend, "generate_speech_stream", None)
    print(f"{'case':<12} {'wall':>7} {'audio':>7} {'x real-time':>12} {'ttfa':>8}")
    for name, language, text in TEXTS:
        walls, audios, speeds, ttfas = [], [], [], []
        for _ in range(args.runs):
            sync()
            t0 = time.perf_counter()
            audio, sr = await backend.generate_speech(text=text, voice=args.voice, language=language)
            sync()
            wall = time.perf_counter() - t0
            duration = len(audio) / sr
            walls.append(wall)
            audios.append(duration)
            speeds.append(duration / wall)
            if stream is not None:
                sync()
                t0 = time.perf_counter()
                agen = stream(text=text, voice=args.voice, language=language)
                await agen.__anext__()
                ttfas.append(time.perf_counter() - t0)
                await agen.aclose()
        case = {
            "chars": len(text),
            "wall_s": summarize(walls),
            "audio_s": summarize(audios),
            "x_realtime": summarize(speeds),
            "ttfa_s": summarize(ttfas),
        }
        results["cases"][name] = case
        ttfa = f"{case['ttfa_s']['median'] * 1000:.0f}ms" if ttfas else "-"
        print(f"{name:<12} {case['wall_s']['median']:>6.2f}s {case['audio_s']['median']:>6.2f}s "
              f"{case['x_realtime']['median']:>11.2f}x {ttfa:>8}")

    if torch.cuda.is_available():
        results["max_memory_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
        print(f"peak VRAM allocated: {results['max_memory_allocated_gb']:.2f} GB")
    if args.label:
        print(f"saved {save_results(args.label, results)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--voice", default="Kyle")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--label", default=None)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()

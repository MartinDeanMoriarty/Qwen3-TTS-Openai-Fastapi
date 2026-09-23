#!/usr/bin/env python3
# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""End-to-end latency benchmark against the running HTTP server.

Measures what a client such as Open WebUI experiences: the full round trip of
POST /v1/audio/speech. With --stream it also measures the time to the first
audio byte. Standard library only, so it runs on the host.

    python3 benchmarks/bench_api.py --label baseline --voice Kyle
    python3 benchmarks/bench_api.py --label fast-stream --stream
    python3 benchmarks/bench_api.py --label cold --cold
"""

import argparse
import http.client
import io
import json
import sys
import time
import urllib.parse
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import TEXTS, save_results, summarize  # noqa: E402


def post(url: str, payload: dict, timeout: float = 600.0):
    """POST and read the body incrementally. Returns (ttfb_s, total_s, body)."""
    parsed = urllib.parse.urlparse(url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=timeout)
    body = json.dumps(payload).encode()
    start = time.perf_counter()
    conn.request("POST", parsed.path, body=body, headers={"Content-Type": "application/json"})
    response = conn.getresponse()
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {response.read()[:300]!r}")
    first = response.read1(65536) if hasattr(response, "read1") else response.read(1)
    ttfb = time.perf_counter() - start
    rest = response.read()
    total = time.perf_counter() - start
    conn.close()
    return ttfb, total, first + rest


def wav_duration(data: bytes) -> float:
    with wave.open(io.BytesIO(data)) as w:
        frames = w.getnframes()
        # Streamed WAV carries a placeholder length; fall back to the byte count.
        if frames <= 0 or frames >= 0x7FFFFFFF // (w.getsampwidth() * w.getnchannels()):
            frames = (len(data) - 44) // (w.getsampwidth() * w.getnchannels())
        return frames / w.getframerate()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8881")
    parser.add_argument("--voice", default="Kyle")
    parser.add_argument("--model", default="tts-1")
    parser.add_argument("--format", default="wav", help="wav gives audio duration and RTF")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--stream", action="store_true", help="request streaming and report time to first byte")
    parser.add_argument("--cold", action="store_true", help="unload the model first and time the reload")
    parser.add_argument("--no-language", action="store_true", help="omit language, like Open WebUI does")
    parser.add_argument("--label", default=None, help="save results to benchmarks/results/<label>.json")
    args = parser.parse_args()

    speech_url = args.url.rstrip("/") + "/v1/audio/speech"

    def payload(text, language):
        p = {"model": args.model, "input": text, "voice": args.voice, "response_format": args.format}
        if not args.no_language:
            p["language"] = language
        if args.stream:
            p["stream"] = True
        return p

    results = {"config": vars(args), "cases": {}}

    if args.cold:
        post(args.url.rstrip("/") + "/admin/unload", {})
        _, total, body = post(speech_url, payload(TEXTS[0][2], TEXTS[0][1]))
        results["cold_start_s"] = total
        print(f"cold start (reload + first request): {total:.2f}s")
    else:
        # One throwaway request so lazy loading does not count as a warm run.
        post(speech_url, payload("Hallo.", "German"))

    print(f"{'case':<12} {'wall':>7} {'ttfb':>7} {'audio':>7} {'x real-time':>12}")
    for name, language, text in TEXTS:
        walls, ttfbs, rtfs, durations = [], [], [], []
        for _ in range(args.runs):
            ttfb, total, body = post(speech_url, payload(text, language))
            walls.append(total)
            ttfbs.append(ttfb)
            if args.format == "wav":
                duration = wav_duration(body)
                durations.append(duration)
                rtfs.append(duration / total)
        case = {
            "chars": len(text),
            "wall_s": summarize(walls),
            "ttfb_s": summarize(ttfbs),
            "audio_s": summarize(durations),
            "x_realtime": summarize(rtfs),
        }
        results["cases"][name] = case
        audio = f"{case['audio_s']['median']:.2f}s" if durations else "-"
        speed = f"{case['x_realtime']['median']:.2f}x" if rtfs else "-"
        print(f"{name:<12} {case['wall_s']['median']:>6.2f}s {case['ttfb_s']['median']:>6.3f}s {audio:>7} {speed:>12}")

    if args.label:
        print(f"saved {save_results(args.label, results)}")


if __name__ == "__main__":
    main()

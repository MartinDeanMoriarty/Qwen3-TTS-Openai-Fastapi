# Qwen3-TTS Optimization Guide

What makes this server fast, how to switch each part, and what to watch out
for. Measurements and methodology: [docs/PERFORMANCE.md](docs/PERFORMANCE.md).

On an RTX 4070 Ti with the 1.7B Base model a short sentence takes ~0.35 s
(~7x real time) and streamed audio starts after ~40 ms; the stock
`transformers.generate()` path needed ~12 s for the same sentence.

## Where the time went

Qwen3-TTS generates one 80 ms audio frame per step: a 28-layer talker picks the
first codebook, then a 5-layer code predictor adds 15 more codebooks one by
one. At batch size 1 every step is thousands of tiny GPU kernels, so the cost
is launch overhead and weight reads, not arithmetic. Attention (Flash
Attention vs SDPA) and TF32 make no measurable difference.

## What is active (`TTS_BACKEND=fast`)

| Optimization | Where | Switch |
|---|---|---|
| CUDA graph per decode step (predictor + talker + sampling) | `qwen_tts/inference/fast_generator.py` | `TTS_BACKEND=fast` |
| Graphed prefill for x-vector voice clones | same | always with `fast` |
| Attention spans 256/512/1024/2048 over one KV storage | same | `TTS_MAX_SEQ_LEN` |
| torch.compile of the talker and predictor forwards | same | `TTS_COMPILE=true` |
| int8 weights, own Triton W8A16 kernel | `qwen_tts/inference/int8_linear.py` | `TTS_QUANTIZE=int8` |
| Streaming (`stream`, `stream_format`) with incremental decoding | `api/backends/fast_qwen3_tts.py` | per request |
| Speech-tokenizer decoder as CUDA graphs for short windows | `StreamingDecoder` | always with `fast` |
| Voice prompts computed once at startup, kept across unloads | `api/services/voices.py` | always |
| One GPU thread, requests in arrival order | `api/backends/base.py` | always |

`TTS_BACKEND=official` keeps the stock `generate()` path (with the quick fixes
below) as a fallback and for CPU use.

## What is deliberately off

* **`torch.backends.cudnn.benchmark`**: every request decodes a different
  length, and benchmark mode re-tunes the decoder's convolutions for each one.
  It cost ~8.4 s per request. Do not turn it back on.
* **Flash Attention 2**: no measurable gain here, long build.
* **Whole-step torch.compile**: 3 % faster than compiling the forwards, but
  ~250 s compile time.
* **int4 weights**: ~9 % faster than int8 but measurably worse quality.

## Startup and memory

* First start: ~2.5 min (torch.compile and the int8 kernel build). The
  compiled kernels are cached in the `qwen3-tts-compile-cache` volume; later
  starts take ~30–40 s. `TTS_COMPILE=false TTS_QUANTIZE=none` starts in ~10 s
  and runs at ~3x real time.
* VRAM: ~3.7 GB with int8 (process total), resident
  (`TTS_INACTIVITY_TIMEOUT_MINUTES=0`). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  (set in the image) saves ~1.3 GB of fragmentation.
* With auto-unload or `POST /admin/unload` the process keeps only its CUDA
  context (~0.4 GB). Reload plus the first request takes ~3.5 s: the compiled
  code is reused, only the graphs are captured again.
* The server captures graphs for the default sampling settings at startup.
  Requests with other settings would capture new graphs on first use.

## Streaming

```bash
# raw PCM/WAV chunks as they are generated (lowest latency)
curl -N http://localhost:8881/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"tts-1","input":"Hallo, wie geht es dir?","voice":"Jarvis","response_format":"wav","stream":true}' \
  --output out.wav

# server-sent events with base64 audio deltas (OpenAI stream_format "sse")
curl -N http://localhost:8881/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"tts-1","input":"Hallo!","voice":"Jarvis","response_format":"pcm","stream_format":"sse"}'
```

Compressed formats (mp3, opus, aac, flac) stream through a single ffmpeg
process; the first MP3 bytes arrive after ~175 ms instead of ~43 ms for WAV.

## Tuning knobs

| Variable | Default | Effect |
|---|---|---|
| `TTS_STREAM_CHUNK_FRAMES` | `2,4,8,12` | smaller first chunks = earlier audio, more decoder calls |
| `TTS_STREAM_LEFT_CONTEXT` | `72` | decoder context per chunk; seams vs. a full decode: ~36 dB SNR at 72, ~20 dB at 25 |
| `TTS_MAX_SEGMENT_CHARS` | `400` | long inputs are split into sentence groups of this size |
| `TTS_MAX_SEQ_LEN` | `2048` | talker KV positions per segment; a warning is logged if a segment hits it |

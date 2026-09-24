# Performance: what was slow, what changed, what it measured

All numbers in this document were measured on one machine, with the scripts in
[`benchmarks/`](../benchmarks) (raw JSON in `benchmarks/results/`):

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti, 12 GB (Ada, SM 8.9), driver 580, shared with Ollama |
| CPU / RAM | Intel i9-10900X, 64 GB |
| Model | `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, voice cloning with an x-vector (a private voice sample; the scripts default to the shipped voice "Jarvis") |
| Texts | five assistant-style utterances, four German, one English (`benchmarks/common.py`) |
| Software | PyTorch 2.11 (cu128), transformers 4.57.3, Python 3.12 |

"x real time" is seconds of audio per second of wall time; higher is better.
"TTFA" is the time to the first audio chunk when streaming.

## Result

End to end through the path Open WebUI uses (Open WebUI → owic-tts-router →
`POST /v1/audio/speech`, MP3, no language given):

| Text | Before | After |
|---|---|---|
| "Alles klar, ich kümmere mich darum." (2.4 s audio) | ~12 s | **0.40 s** |
| Weather sentence (~4.7 s audio) | ~15.5 s | **0.71 s** |
| Three sentences (~11 s audio) | ~25 s | **1.58 s** |

Directly against the API: non-streaming WAV runs at **6.8–7.4x real time**;
streaming WAV delivers the first audio after **42–44 ms**, streaming MP3 after
~175 ms (ffmpeg start and encoder delay).

Quality did not change measurably: Whisper large-v3-turbo transcribes every
sample of every configuration below without a word error, except int4 (see
below), and the speaker similarity stays at 0.981.

## Step by step (in-process, `benchmarks/bench_engine.py`)

| Stage | "Alles klar …" | 11 s of audio | x real time | TTFA | First request |
|---|---|---|---|---|---|
| 0. As shipped | 11.62 s | 23.87 s | 0.19–0.45 | – | 31.9 s |
| 1. Removed harmful "optimizations", cached voice prompts | 3.36 s | 15.34 s | 0.70–0.72 | – | 1.35 s |
| 2. CUDA-graph decode step | 0.80 s | 3.54 s | 2.93–3.22 | – | 0.35 s |
| 3. Streaming, graphed prefill and short decodes | 0.90 s | 3.47 s | 2.91–3.15 | 70 ms | |
| 4. torch.compile + attention spans | 0.52 s | 2.27 s | 4.34–4.81 | 49 ms | |
| 5. int8 weights (own Triton kernel) | **0.35 s** | **1.45 s** | **6.46–7.26** | **41 ms** | 0.25 s |

### 0. What was slow

* **`torch.backends.cudnn.benchmark = True`** (listed as an optimization) added
  **~8.4 s to every request**. The speech-tokenizer decoder sees a new input
  length on every request, and benchmark mode re-runs cuDNN's algorithm search
  for each new length. It also made the first use of each voice take 10–12 s.
  Without it the decode takes 0.03–0.07 s.
* **`torch.compile` had no effect.** It wrapped the model module, but the code
  calls `model.generate()`, which bypasses the compiled `forward`.
* Flash Attention 2 was not installed (SDPA was used), and TF32 does not apply
  to a bf16 model. Neither matters: attention is not the bottleneck.
* **The decode loop itself**: every 80 ms audio frame runs one talker step and,
  inside it, a nested `transformers.generate()` for 15 code-predictor steps.
  At batch size 1 that is ~7000 small kernels plus Python per frame:
  110–118 ms per frame, slower than real time, with the GPU mostly idle.
* Voice prompts were recomputed after every auto-unload (5 min idle), and
  every model load asked the Hugging Face Hub API whether the tokenizer is a
  Mistral one (a round trip, and loading failed offline).
* The base image ran Python **3.11.0rc1**, a release candidate on which
  TorchDynamo segfaulted while compiling the model.

### 1. Quick fixes

cuDNN benchmark off, the ineffective compile removed, voice prompts
precomputed at startup and kept on the CPU across unloads, the unused
speech-tokenizer encoder pass skipped for x-vector prompts, the model loaded
from the local snapshot, inference moved off the event loop onto one GPU
thread that serves requests in arrival order.

### 2. CUDA-graph decode step (`qwen_tts/inference/fast_generator.py`)

One whole decode step, i.e. 15 code-predictor steps, the talker step and both
samplers, is recorded once as a CUDA graph over static buffers
(transformers `StaticCache`) and replayed per frame. All per-step state
(cache position, text index, repetition-penalty history, frame buffer) lives
on the GPU; the host only reads the sampled token to detect the end.
The approach of using `StaticCache` under a captured forward follows
[faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) (MIT);
here predictor, talker and sampling share one graph.

Per frame: 24.1 ms instead of ~115 ms. Checks:

* Teacher-forced along a full `generate()` trajectory, the talker logits differ
  by at most 0.25 (1–2 bf16 ulps); the code predictor produces identical codes
  from identical inputs.
* Greedy decoding diverges after a few frames anyway, because the top-2 logit
  gap is often only 1–2 ulps; with sampling this is noise, not a difference.
* The graph replay is bit-identical to the same step run eagerly (test).

### 3. Streaming

* `stream: true` or `stream_format: "audio" | "sse"` on `/v1/audio/speech`.
  PCM/WAV are sent directly; MP3/Opus/AAC/FLAC go through one ffmpeg process
  per stream (separately encoded pieces would have a gap at every seam).
  ffmpeg needed `-probesize 32 -analyzeduration 0 -fflags nobuffer
  -flush_packets 1`; without them it held back the first 160 ms of input for
  more than 2 s.
* Chunks of 2, 4, 8, then 12 frames: the first audio comes early and every
  chunk is ready before the previous one has played.
* Each chunk is decoded with 72 frames of left context (the decoder
  transformer's sliding window). Against a full decode the seams measured
  ~20 dB SNR with upstream's 25 frames and ~36 dB with 72, which is also the
  distance of the bf16 decoder to an fp32 one (35.4 dB). The decoder holds
  back its last 555 samples per call until the next frame exists; they are
  emitted with the next chunk.
* The prompt of an x-vector voice clone always has 9–10 positions (the text is
  fed during decoding), so the prefill runs as a CUDA graph too: 14.6 ms
  instead of 31 ms.
* Decoder calls of up to 16/32 frames are launch-bound (~20 ms eager) and run
  as CUDA graphs (7/10 ms). From ~64 frames on the decoder is compute-bound
  (~0.42 ms per frame) and a graph no longer helps.

### 4. torch.compile and attention spans

* A step was ~7000 kernels, ~5200 of them tiny elementwise ops (RMSNorm pieces,
  RoPE, residual adds) at ~1.5 µs each. `torch.compile` on the talker and
  predictor forwards fuses them; the compiled forwards are captured inside the
  graph. 23.9 → 18.2 ms per step. Compiling the whole step including sampling
  gained only 3 % more and took 250 s to compile, so it is not used.
* Every step attended over its whole static KV buffer. With int8 weights a step
  took 13.6 ms with 2048 positions, 12.0 ms with 1024 and 11.0 ms with 512. The
  step now runs in the smallest of 256/512/1024/2048 positions that holds the
  current position (views of one shared KV storage, one graph each).

### 5. int8 weights (`qwen_tts/inference/int8_linear.py`)

At batch size 1 each step reads every weight once; cuBLAS already reads bf16 at
~500 GB/s, the memory bandwidth. The talker and code-predictor transformer
layers store int8 weights with one scale per output row (231 layers,
-1.4 GB VRAM), multiplied by a small Triton kernel with one fixed launch
configuration (swept over block sizes and warps). Embeddings and output heads
stay bf16.

| int8 path | x real time | Start (warm cache) |
|---|---|---|
| torchao `Int8WeightOnlyConfig` + compile | 6.0–6.8 | 104–376 s (Inductor coordinate-descent autotuning, not reliably cached) |
| same without torchao's autotuning config | 1.4–1.5 | 34 s |
| own Triton kernel | **6.5–7.3** | **30–39 s** (first start ~2.5 min) |

### Memory and unloading

* `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` lowered the reserved VRAM
  of the loaded server from ~4.7 GB to ~3.4 GB at unchanged speed; the process
  shows ~3.7 GB in `nvidia-smi`.
* Unloading used to leave ~4 GB reserved: only 67 MB were still allocated, but
  the per-stream cuBLAS workspaces sat inside the segments that had held the
  weights, so `empty_cache()` could release none of them. Clearing the
  workspaces on unload frees everything but the CUDA context (~0.4 GB).
  Reload plus the first request then takes ~3.5 s.

## Quality (`benchmarks/eval_quality.py`)

Each text is generated with fixed seeds; Whisper large-v3-turbo transcribes it
(word error rate against the input) and the Base model's own speaker encoder
compares the voice with the reference sample (cosine similarity).

| Configuration | Samples | WER | Speaker similarity (mean / min) |
|---|---|---|---|
| As shipped | 15 | 0 | 0.9808 / 0.9779 |
| CUDA graph | 15 | 0 | 0.9811 / 0.9779 |
| CUDA graph, streamed | 15 | 0 | 0.9811 / 0.9779 |
| + torch.compile | 25 | 0 | 0.9813 / 0.9777 |
| + int8 (torchao) | 25 | 0 | 0.9808 / 0.9759 |
| **+ int8 (own kernel, default)** | 25 | **0** | **0.9810 / 0.9772** |
| int4 talker (torchao, tinygemm), rejected | 25 | 0.0013 | 0.9793 / 0.9726 |

int4 was ~9 % faster than int8 but measurably worse (one misread word, lower
similarity) and is not offered.

## Configuration

| Variable | Default (compose) | Meaning |
|---|---|---|
| `TTS_BACKEND` | `fast` | `fast` = CUDA-graph engine, `official` = transformers `generate()` |
| `TTS_COMPILE` | `true` | torch.compile of the step forwards |
| `TTS_QUANTIZE` | `int8` | `int8` or `none` (bf16) |
| `TTS_MAX_SEQ_LEN` | `2048` | talker KV positions per segment (~160 s of audio) |
| `TTS_MAX_SEGMENT_CHARS` | `400` | longer inputs are synthesized in sentence groups |
| `TTS_STREAM_CHUNK_FRAMES` | `2,4,8,12` | streamed chunk sizes (80 ms frames), last one repeats |
| `TTS_STREAM_LEFT_CONTEXT` | `72` | frames of decoder context per streamed chunk |
| `TTS_DEFAULT_LANGUAGE` | `Auto` | used when neither request, model suffix nor text decide |
| `TTS_INACTIVITY_TIMEOUT_MINUTES` | `0` | auto-unload after idle minutes (0 = stay resident) |

Language: clients such as Open WebUI send none. It is taken from the model
suffix (`tts-1-de`), then the request, then detected from the text
(German/English), then `TTS_DEFAULT_LANGUAGE`. Previously a cloned voice
always got "English". The English-only text normalization (numbers spelled
out in English, "@" → "at") now only runs for English text.

## Reproducing

```bash
# end to end against the running server (standard library only)
python3 benchmarks/bench_api.py --no-language --label mine
python3 benchmarks/bench_api.py --stream --label mine-stream

# In-process and quality: these load a second copy of the model, which does
# not fit next to the server's on a 12 GB card, so unload the server's first
# (it reloads on the next request). --whisper downloads Whisper once.
curl -X POST localhost:8881/admin/unload
docker exec -w /tmp qwen3-tts-api python /app/benchmarks/bench_engine.py --label engine
docker exec -w /tmp -e HF_HUB_OFFLINE=0 qwen3-tts-api python /app/benchmarks/eval_quality.py --label q --whisper
```

The results land in `/app/benchmarks/results` inside the container; mount the
directory or `docker cp` them out.

## Not done, and why

| Idea (from the old guide) | Verdict |
|---|---|
| Dynamic batching | One user, one request at a time: batching adds latency, it does not remove it. |
| vLLM(-Omni) | Built for throughput across many requests and GPUs, and reserves a fixed share of VRAM up front, which does not fit a 12 GB card shared with Ollama. The engine above covers what matters at one request: static cache, graphs, streaming. |
| Tensor parallelism | One GPU. |
| FP8 | Supported on Ada, but at batch size 1 the activations would have to be quantized every step; int8 weights with bf16 activations already halve the bytes read. |
| Streaming into Open WebUI | Open WebUI 0.11 reads the whole response before playing it (`await r.read()`); it asks for one sentence at a time, so there the time to a whole sentence counts, which is what the numbers above cut. |

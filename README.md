# Qwen3-TTS OpenAI-Compatible API

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/qwen3_tts_logo.png" width="400"/>
</p>

> **Fork with Enhancements**: This version adds **28 custom cloned voices**, **VRAM auto-unload** for multi-GPU-app setups (Ollama, Kokoro TTS, etc.), and optimizations for **NVIDIA RTX 50-series Blackwell** GPUs.  
> See [INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md) for detailed usage.

An **OpenAI-compatible FastAPI server** for **Qwen3-TTS**, enabling drop-in replacement for OpenAI's TTS API. Built on the powerful [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) model by Alibaba Cloud.

## Features

| Feature | Description |
|---------|-------------|
| **OpenAI API Compatible** | Drop-in replacement for `POST /v1/audio/speech` |
| **28 Custom Voices** | Pre-cloned voices for narration, audiobooks, assistants |
| **VRAM Auto-Unload** | Free GPU memory after inactivity for other apps |
| **Voice Cloning** | Add your own voices with 3-10 second samples |
| **10+ Languages** | English, Chinese, Japanese, Korean, German, French, Spanish, Russian, Portuguese, Italian |
| **Multiple Formats** | MP3, Opus, AAC, FLAC, WAV, PCM |
| **Fast** | CUDA-graph engine with torch.compile and int8 weights: ~7x real time, a short sentence in ~0.35 s on an RTX 4070 Ti |
| **Streaming** | `stream: true` / `stream_format` (`audio`, `sse`); first audio after ~40 ms (WAV/PCM) |
| **Language detection** | German/English detected from the text when the client sends no language |

## Quick Start

### Using OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8881/v1",
    api_key="not-needed"
)

response = client.audio.speech.create(
    model="tts-1",
    voice="Jarvis",  # Or any of the 28 custom voices
    input="Hello! This is your local TTS server speaking.",
    response_format="mp3"
)

response.stream_to_file("output.mp3")
```

### Available Voices

**Male**: Jarvis, Paddington, Professor, Josh, John, Mark, Adam, Russell, Curt, Eustis, General_Joe, Grandpa, Nigel, Richard, Valentino, Wildebeest

**Female**: Lucy, Carmen, Caroline, Joanne, Victoria, Natasha, Bianca, Cecile, Emmaline, Monika, Tally, Villain

## Deployment

### Docker (Recommended)

```bash
# Clone and start
git clone https://github.com/YOUR_USERNAME/Qwen3-TTS-Openai-Fastapi.git
cd Qwen3-TTS-Openai-Fastapi

# Build and run with GPU
docker compose up -d qwen3-tts-gpu

# View logs
docker compose logs -f qwen3-tts-gpu
```

### Environment Variables

Defaults as set in `docker-compose.yml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8881` | Server port |
| `TTS_BACKEND` | `fast` | `fast` (CUDA graphs, GPU) or `official` (transformers `generate()`) |
| `TTS_COMPILE` | `true` | torch.compile the decode step (first start ~2.5 min, then cached) |
| `TTS_QUANTIZE` | `int8` | `int8` weights or `none` (bf16) |
| `TTS_MODEL_NAME` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Model to use |
| `TTS_INACTIVITY_TIMEOUT_MINUTES` | `0` | Auto-unload after idle (0=disabled) |
| `TTS_DEFAULT_LANGUAGE` | `Auto` | Language when request, model suffix and text leave it open |
| `VOICE_SAMPLES_DIR` | `/app/voice-samples` | Custom voice samples path |

More knobs (streaming chunk sizes, segment length, KV length) in
[OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md).

## VRAM Management

The server uses ~3.7 GB VRAM (1.7B model, int8 weights). Unloading frees all
but the CUDA context (~0.4 GB); reload plus the first request takes ~3.5 s.
Auto-unload frees memory when idle:

```bash
# Check status (shows idle time, countdown)
curl http://localhost:8881/health

# Manually free VRAM
curl -X POST http://localhost:8881/admin/unload

# Pre-load model
curl -X POST http://localhost:8881/admin/reload
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/audio/speech` | Generate speech (OpenAI-compatible) |
| `GET /v1/models` | List available models |
| `GET /v1/voices` | List available voices |
| `GET /health` | Health check with VRAM status |
| `POST /admin/unload` | Manually unload model |
| `POST /admin/reload` | Manually reload model |
| `GET /docs` | Swagger UI documentation |

## Adding Custom Voices

1. Prepare a 3-10 second clear audio sample (.wav)
2. Name it `VoiceName.wav` and copy to `sample-voices-xtts/`
3. Restart the container
4. Use with `voice="VoiceName"`

## Documentation

- **[INTEGRATION_GUIDE.md](INTEGRATION_GUIDE.md)** - Detailed integration examples (FastAPI, Node.js, Gradio, Home Assistant, Open WebUI)
- **[OPTIMIZATION_GUIDE.md](OPTIMIZATION_GUIDE.md)** - What makes it fast and how to switch it
- **[docs/PERFORMANCE.md](docs/PERFORMANCE.md)** - Measurements: what was slow, what changed, quality checks

## Credits

This project is built on the incredible work of the **Qwen Team at Alibaba Cloud**:

- [Qwen3-TTS Original Repository](https://github.com/QwenLM/Qwen3-TTS)
- [Qwen3-TTS Technical Blog](https://qwen.ai/blog?id=qwen3tts-0115)
- [Research Paper](https://arxiv.org/abs/2601.15621)
- [Hugging Face Models](https://huggingface.co/collections/Qwen/qwen3-tts)

### Citation

```BibTeX
@article{Qwen3-TTS,
  title={Qwen3-TTS Technical Report},
  author={Hangrui Hu and Xinfa Zhu and Ting He and Dake Guo and Bin Zhang and Xiong Wang and Zhifang Guo and Ziyue Jiang and Hongkun Hao and Zishan Guo and Xinyu Zhang and Pei Zhang and Baosong Yang and Jin Xu and Jingren Zhou and Junyang Lin},
  journal={arXiv preprint arXiv:2601.15621},
  year={2026}
}
```

## License

Apache 2.0 - See [LICENSE](LICENSE)

#!/usr/bin/env python3
# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""Quality check for a TTS backend: speaker similarity and word error rate.

Every speed change has to prove it did not cost quality. This generates each
test text with several fixed seeds and scores the audio two ways:

- speaker similarity: cosine between the Base model's own speaker embedding of
  the generated audio and of the reference sample (needs no download)
- WER/CER: Whisper transcribes the audio and is compared with the input text
  (--whisper, downloads openai/whisper-large-v3-turbo once, ~1.6 GB)

The WAVs are kept next to the JSON so the results can also be listened to.

    docker exec -w /tmp qwen3-tts-dev python /app/benchmarks/eval_quality.py --label q-official --whisper
"""

import argparse
import asyncio
import re
import statistics
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS_DIR, TEXTS, save_results, summarize  # noqa: E402

WHISPER_MODEL = "openai/whisper-large-v3-turbo"


# Whisper writes numbers as digits; the test texts spell them out.
NUMBER_WORDS = {"22": "zweiundzwanzig", "10": "ten"}


def normalize_words(text: str) -> list:
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^\w\s]", " ", text)
    return [NUMBER_WORDS.get(word, word) for word in text.split()]


def edit_distance(ref: list, hyp: list) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def error_rates(reference: str, hypothesis: str):
    ref_words, hyp_words = normalize_words(reference), normalize_words(hypothesis)
    wer = edit_distance(ref_words, hyp_words) / max(len(ref_words), 1)
    ref_chars, hyp_chars = list(" ".join(ref_words)), list(" ".join(hyp_words))
    cer = edit_distance(ref_chars, hyp_chars) / max(len(ref_chars), 1)
    return wer, cer


def speaker_embedding(model, audio: np.ndarray, sr: int) -> torch.Tensor:
    import librosa

    target = model.speaker_encoder_sample_rate
    if sr != target:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target)
    return model.extract_speaker_embedding(audio=audio.astype(np.float32), sr=target).float()


async def run(args):
    from api.backends.factory import get_backend

    backend = get_backend()
    await backend.initialize()
    base_model = backend.model.model  # Qwen3TTSForConditionalGeneration

    voice_path = Path(args.voice_dir) / f"{args.voice}.wav"
    ref_audio, ref_sr = sf.read(voice_path, dtype="float32")
    if ref_audio.ndim > 1:
        ref_audio = ref_audio.mean(axis=1)
    ref_emb = speaker_embedding(base_model, ref_audio, ref_sr)

    out_dir = RESULTS_DIR / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for name, language, text in TEXTS:
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if args.stream:
                chunks = [(c, sr) async for c, sr in backend.generate_speech_stream(
                    text=text, voice=args.voice, language=language)]
                audio, sr = np.concatenate([c for c, _ in chunks]), chunks[0][1]
            else:
                audio, sr = await backend.generate_speech(text=text, voice=args.voice, language=language)
            audio = np.asarray(audio, dtype=np.float32)
            wav_path = out_dir / f"{name}_s{seed}.wav"
            sf.write(wav_path, audio, sr)
            emb = speaker_embedding(base_model, audio, sr)
            similarity = torch.nn.functional.cosine_similarity(emb, ref_emb, dim=0).item()
            samples.append({
                "case": name, "seed": seed, "language": language, "text": text,
                "audio_s": len(audio) / sr, "speaker_similarity": similarity, "wav": str(wav_path),
            })
            print(f"{name:<12} seed {seed}  {len(audio) / sr:5.2f}s  similarity {similarity:.3f}")

    if args.whisper:
        from transformers import pipeline

        await backend.unload()
        asr = pipeline("automatic-speech-recognition", model=WHISPER_MODEL,
                       torch_dtype=torch.float16, device="cuda:0")
        for s in samples:
            audio, sr = sf.read(s["wav"], dtype="float32")
            hyp = asr({"raw": audio, "sampling_rate": sr},
                      generate_kwargs={"language": s["language"].lower(), "task": "transcribe"})["text"]
            s["transcript"] = hyp.strip()
            s["wer"], s["cer"] = error_rates(s["text"], hyp)
            print(f"{s['case']:<12} seed {s['seed']}  WER {s['wer']:.3f}  CER {s['cer']:.3f}  | {s['transcript']}")

    summary = {"speaker_similarity": summarize([s["speaker_similarity"] for s in samples])}
    if args.whisper:
        summary["wer"] = summarize([s["wer"] for s in samples])
        summary["cer"] = summarize([s["cer"] for s in samples])
    print("\nsummary:")
    for key, value in summary.items():
        print(f"  {key:<20} mean {value['mean']:.4f}  median {value['median']:.4f}  "
              f"min {value['min']:.4f}  max {value['max']:.4f}")
    per_case = {}
    for name, _, _ in TEXTS:
        sims = [s["speaker_similarity"] for s in samples if s["case"] == name]
        per_case[name] = {"speaker_similarity_mean": statistics.fmean(sims)}
        if args.whisper:
            per_case[name]["wer_mean"] = statistics.fmean([s["wer"] for s in samples if s["case"] == name])

    results = {"config": vars(args), "backend": backend.get_backend_name(), "summary": summary,
               "per_case": per_case, "samples": samples}
    print(f"saved {save_results(args.label, results)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True)
    parser.add_argument("--voice", default="Jarvis")
    parser.add_argument("--voice-dir", default="/app/sample-voices-xtts")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--whisper", action="store_true")
    parser.add_argument("--stream", action="store_true", help="score the concatenated streaming chunks")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()

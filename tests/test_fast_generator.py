# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
GPU tests for the CUDA-graph generator. Skipped without CUDA or without the
cached Base model (set TTS_TEST_MODEL to use another one).
"""

import os

import pytest
import torch

MODEL = os.getenv("TTS_TEST_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
VOICE = os.path.join(os.path.dirname(__file__), "..", "sample-voices-xtts", "Jarvis.wav")
GREEDY = dict(do_sample=False, subtalker_dosample=False)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.fixture(scope="module")
def setup():
    from qwen_tts import Qwen3TTSModel
    from qwen_tts.inference.fast_generator import FastGenerator
    from qwen_tts.inference.qwen3_tts_model import _cached_snapshot_dir

    if _cached_snapshot_dir(MODEL) is None:
        pytest.skip(f"{MODEL} is not in the local Hugging Face cache")
    tts = Qwen3TTSModel.from_pretrained(MODEL, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    prompt = tts.create_voice_clone_prompt(ref_audio=VOICE, x_vector_only_mode=True)
    generator = FastGenerator(tts, max_seq_len=512)
    return tts, generator, prompt


def _inputs(tts, prompt, text="Alles klar, ich kümmere mich darum."):
    return tts._prepare_voice_clone(text=text, language="German", voice_clone_prompt=prompt)


def test_graph_replay_matches_eager_step(setup):
    tts, gen, prompt = setup
    sampling = gen._sampling(GREEDY)
    key = gen._key(**sampling)
    span = gen.spans[0]
    graph = gen._graph(key, span)
    steps = 12

    with torch.inference_mode():
        gen._prefill(_inputs(tts, prompt), sampling)
        for _ in range(steps):
            gen._step(key, span)
        eager = gen.codes[:steps].clone()

        gen._prefill(_inputs(tts, prompt), sampling)
        for _ in range(steps):
            graph.replay()
        replayed = gen.codes[:steps].clone()

    assert torch.equal(eager, replayed)


def test_greedy_is_deterministic(setup):
    tts, gen, prompt = setup
    first = gen.generate_codes(_inputs(tts, prompt), **GREEDY)
    second = gen.generate_codes(_inputs(tts, prompt), **GREEDY)
    assert torch.equal(first, second)


def test_streamed_chunks_equal_full_generation(setup):
    tts, gen, prompt = setup
    full = gen.generate_codes(_inputs(tts, prompt), **GREEDY)
    chunks = [c.clone() for c in gen.stream_codes(_inputs(tts, prompt), chunk_frames=4, **GREEDY)]
    assert all(len(c) <= 4 for c in chunks)
    assert torch.equal(torch.cat(chunks), full)


def test_generation_crosses_attention_spans(setup):
    tts, gen, prompt = setup
    assert gen.spans == [256, 512]
    text = ("Ich lese dir jetzt eine etwas längere Nachricht vor, damit die Erzeugung über die erste "
            "Grenze des Zwischenspeichers hinausläuft. Der Termin am Montag wurde auf Dienstag verschoben, "
            "weil der Raum belegt ist. Bitte bring die Unterlagen zum Projekt mit, und denk an den Bericht, "
            "den wir letzte Woche besprochen haben. Außerdem möchte der Kunde wissen, ob die Lieferung "
            "noch in diesem Monat möglich ist. Zum Schluss noch eine Erinnerung: Morgen früh um acht "
            "Uhr beginnt die Besprechung mit dem gesamten Team im großen Saal.")
    codes = gen.generate_codes(_inputs(tts, prompt, text))
    assert codes.shape[0] > 256  # generation ran past the first span into the second
    codebook = tts.model.config.talker_config.code_predictor_config.vocab_size
    assert int(codes.min()) >= 0 and int(codes.max()) < codebook
    wavs, sr = tts._decode_codes([codes])
    assert abs(len(wavs[0]) / sr - codes.shape[0] / 12.5) < 0.2


def test_sampled_codes_are_valid(setup):
    tts, gen, prompt = setup
    codes = gen.generate_codes(_inputs(tts, prompt))
    codebook = tts.model.config.talker_config.code_predictor_config.vocab_size
    assert codes.ndim == 2 and codes.shape[1] == 16
    assert 10 < codes.shape[0] < 200  # ~2.5 s of speech is ~30 frames
    assert int(codes.min()) >= 0 and int(codes.max()) < codebook

    wavs, sr = tts._decode_codes([codes])
    assert sr == 24000
    assert abs(len(wavs[0]) / sr - codes.shape[0] / 12.5) < 0.2


def test_decoder_graph_buckets_match_eager(setup):
    tts, gen, prompt = setup
    from qwen_tts.inference.fast_generator import StreamingDecoder

    decoder = StreamingDecoder(tts.model.speech_tokenizer, graph_buckets=(32,))
    codes = gen.generate_codes(_inputs(tts, prompt))[:20]
    with torch.inference_mode():
        graphed = decoder.decode(codes).float()
        eager = decoder.decoder(codes.T.unsqueeze(0)).view(-1).float()
    assert graphed.shape == eager.shape
    # Zero padding behind the frames only shifts bf16 rounding
    snr = 10 * torch.log10((eager ** 2).sum() / ((eager - graphed) ** 2).sum())
    assert snr > 25

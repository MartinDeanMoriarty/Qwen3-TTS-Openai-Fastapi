# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
CUDA-graph generation for Qwen3-TTS (12 Hz tokenizer, batch size 1).

The stock path runs `transformers.generate()` for the talker and, inside every
talker step, a second `generate()` for the 15 code-predictor steps. At batch
size 1 that is hundreds of tiny kernel launches per 80 ms audio frame with
Python in between, and the GPU mostly waits: ~115 ms per frame on an RTX 4070 Ti,
slower than real time.

Here one whole decode step (code predictor, talker and both samplers) is
recorded once as a CUDA graph over static buffers and replayed per frame. All
per-step state lives on the GPU: cache position, text index, repetition-penalty
history and the frame buffer, so a step is a single graph launch plus one
device-to-host read of the sampled token to detect the end.

The use of transformers' StaticCache under a captured forward follows
faster-qwen3-tts by Andres Marafioti (MIT License,
https://github.com/andimarafioti/faster-qwen3-tts). Unlike that project, the
predictor loop, the talker step and all sampling share one graph and no
Python runs between them.

Sampling mirrors `generate()`: repetition penalty, the min-new-tokens EOS block
and suppressed tokens, then temperature, top-k and top-p, in that order, on
float32 logits.
"""

import logging
from typing import Dict, Iterator, Sequence, Union

import torch
import torch.nn.functional as F
from transformers import StaticCache

logger = logging.getLogger(__name__)

MIN_NEW_TOKENS = 2
# Prompts up to this many positions run their prefill as a captured graph
MAX_GRAPHED_PREFILL = 64
# Attention spans of the captured decode steps. Every step attends over its
# whole static KV buffer, used or not: on an RTX 4070 Ti a step took 13.6 ms
# with 2048 positions and 11.0 ms with 512. A step runs in the smallest span
# that holds the current position; all spans share one KV storage.
ATTENTION_SPANS = (256, 512, 1024)


def _warp(logits: torch.Tensor, temperature, top_k: int, top_p: float) -> torch.Tensor:
    """Temperature, top-k and top-p exactly like the HF logits warpers."""
    logits = logits / temperature
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.shape[-1])).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=False)
        cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cumulative <= (1 - top_p)
        remove[..., -1:] = False
        logits = logits.masked_fill(remove.scatter(-1, sorted_idx, remove), float("-inf"))
    return logits


def _sample(logits: torch.Tensor, do_sample: bool, temperature, top_k: int, top_p: float) -> torch.Tensor:
    """[1, V] float32 logits -> [1, 1] token id."""
    if not do_sample:
        return logits.argmax(dim=-1, keepdim=True)
    probs = F.softmax(_warp(logits, temperature, top_k, top_p), dim=-1)
    return torch.multinomial(probs, num_samples=1)


class FastGenerator:
    """Replays a captured decode step for one request at a time.

    Args:
        tts: the `Qwen3TTSModel` wrapper (model on a CUDA device).
        max_seq_len: talker KV-cache length; prefill plus generated frames must fit.
        compile: fuse the talker and predictor forwards with torch.compile before
            capture. A step is ~7000 kernels, ~5200 of them tiny elementwise ops
            (RMSNorm pieces, RoPE, residual adds) that cost launch latency even
            inside a graph; Inductor merges them.
    """

    def __init__(self, tts, max_seq_len: int = 2048, compile: bool = False):
        self.tts = tts
        self.model = tts.model
        self.talker = self.model.talker
        self.predictor = self.talker.code_predictor
        self.tcfg = self.talker.config
        self.pcfg = self.predictor.config
        self.device = self.talker.device
        self.dtype = self.talker.dtype
        self.max_seq_len = max_seq_len

        self.hidden = self.tcfg.hidden_size
        self.vocab = self.tcfg.vocab_size
        self.groups = self.tcfg.num_code_groups
        self.eos = self.tcfg.codec_eos_token_id

        dev, dt = self.device, self.dtype
        # Talker KV cache, views of its first `span` positions, and the key
        # positions the decode masks are built from
        self.talker_cache = StaticCache(config=self.tcfg, max_cache_len=max_seq_len)
        self.key_positions = torch.arange(max_seq_len, device=dev)
        self.spans = [n for n in ATTENTION_SPANS if n < max_seq_len] + [max_seq_len]
        # Code predictor: 2 prefill positions + 14 decode positions
        pred_len = self.groups + 1
        self.pred_cache = StaticCache(config=self.pcfg, max_cache_len=pred_len)
        self.pred_prefill_pos = torch.arange(2, device=dev)
        self.pred_decode_pos = [torch.tensor([2 + i], device=dev) for i in range(self.groups - 2)]
        pred_keys = torch.arange(pred_len, device=dev)

        def causal(rows):
            allowed = pred_keys.unsqueeze(0) <= rows.unsqueeze(1)
            mask = torch.zeros(allowed.shape, dtype=dt, device=dev).masked_fill(~allowed, torch.finfo(dt).min)
            return {"full_attention": mask.view(1, 1, *allowed.shape)}

        self.pred_prefill_mask = causal(self.pred_prefill_pos)
        self.pred_decode_masks = [causal(p) for p in self.pred_decode_pos]
        self._allocate(self.talker_cache, self.tcfg)
        self._allocate(self.pred_cache, self.pcfg)
        self.talker_views = {n: self._cache_view(n) for n in self.spans}

        # Per-request state, all on the GPU so the graph can advance it
        self.token = torch.zeros(1, 1, dtype=torch.long, device=dev)
        self.past_hidden = torch.zeros(1, 1, self.hidden, dtype=dt, device=dev)
        self.pos = torch.zeros(1, dtype=torch.long, device=dev)
        self.gen_step = torch.zeros(1, dtype=torch.long, device=dev)
        self.n_generated = torch.zeros(1, dtype=torch.long, device=dev)
        self.step = torch.zeros(1, dtype=torch.long, device=dev)
        self.presence = torch.zeros(self.vocab, dtype=torch.bool, device=dev)
        self.trailing = torch.zeros(1, max_seq_len, self.hidden, dtype=dt, device=dev)
        self.trailing_len = torch.zeros(1, dtype=torch.long, device=dev)
        self.tts_pad = torch.zeros(1, 1, self.hidden, dtype=dt, device=dev)
        self.codes = torch.zeros(max_seq_len, self.groups, dtype=torch.long, device=dev)
        # Scalars that may change between requests without a recapture
        self.temperature = torch.ones(1, dtype=torch.float32, device=dev)
        self.sub_temperature = torch.ones(1, dtype=torch.float32, device=dev)
        self.rep_penalty = torch.ones(1, dtype=torch.float32, device=dev)

        suppress = torch.zeros(self.vocab, dtype=torch.bool, device=dev)
        suppress[self.vocab - 1024:] = True
        suppress[self.eos] = False
        self.suppress_mask = suppress
        self.eos_mask = torch.zeros(self.vocab, dtype=torch.bool, device=dev)
        self.eos_mask[self.eos] = True

        self._graphs: Dict[tuple, torch.cuda.CUDAGraph] = {}
        self._prefill_graphs: Dict[int, tuple] = {}
        self._pool = None

        # Decode-step forwards; the prefill keeps the plain modules (variable length)
        self._talker_forward = self.talker.model.forward
        self._pred_forward = self.predictor.model.forward
        if compile:
            self._talker_forward = torch.compile(self._talker_forward, dynamic=False)
            self._pred_forward = torch.compile(self._pred_forward, dynamic=False)

    def _allocate(self, cache: StaticCache, config) -> None:
        """StaticCache allocates lazily on its first update; do it before any capture."""
        kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        dummy = torch.zeros(1, kv_heads, 1, head_dim, dtype=self.dtype, device=self.device)
        for layer in cache.layers:
            if not layer.is_initialized:
                layer.lazy_initialization(dummy)

    def _cache_view(self, span: int) -> StaticCache:
        """A StaticCache over the first `span` positions of the full talker cache."""
        if span == self.max_seq_len:
            return self.talker_cache
        view = StaticCache(config=self.tcfg, max_cache_len=span)
        for layer, full in zip(view.layers, self.talker_cache.layers):
            layer.__dict__.update({k: v for k, v in full.__dict__.items() if k not in ("keys", "values")})
            layer.max_cache_len = span
            layer.keys = full.keys[:, :, :span]
            layer.values = full.values[:, :, :span]
        return view

    def span_for(self, position: int) -> int:
        """Smallest attention span that holds `position`."""
        return next(n for n in self.spans if position < n)

    # ------------------------------------------------------------------ step

    def _step(self, key: tuple, span: int) -> None:
        """One frame: 15 codebooks from the predictor, then the next first-codebook token."""
        do_sample, top_k, top_p, sub_do_sample, sub_top_k, sub_top_p = key
        talker, pred = self.talker, self.predictor

        # Code predictor over [talker hidden, embedding of the current token]
        last_id_hidden = talker.get_input_embeddings()(self.token)
        embeds = [last_id_hidden]
        h = pred.small_to_mtp_projection(torch.cat((self.past_hidden, last_id_hidden), dim=1))
        out = self._pred_forward(inputs_embeds=h, attention_mask=self.pred_prefill_mask,
                                 past_key_values=self.pred_cache, cache_position=self.pred_prefill_pos,
                                 use_cache=True)
        logits = pred.lm_head[0](out.last_hidden_state[:, -1]).float()
        tok = _sample(logits, sub_do_sample, self.sub_temperature, sub_top_k, sub_top_p)
        frame = [self.token.view(1), tok.view(1)]
        for i in range(1, self.groups - 1):
            emb = pred.model.codec_embedding[i - 1](tok)
            embeds.append(emb)
            out = self._pred_forward(inputs_embeds=pred.small_to_mtp_projection(emb),
                                     attention_mask=self.pred_decode_masks[i - 1], past_key_values=self.pred_cache,
                                     cache_position=self.pred_decode_pos[i - 1], use_cache=True)
            logits = pred.lm_head[i](out.last_hidden_state[:, -1]).float()
            tok = _sample(logits, sub_do_sample, self.sub_temperature, sub_top_k, sub_top_p)
            frame.append(tok.view(1))
        embeds.append(pred.model.codec_embedding[self.groups - 2](tok))

        # Talker input: sum of all 16 codebook embeddings plus the next text position
        inputs_embeds = torch.cat(embeds, dim=1).sum(1, keepdim=True)
        text_idx = torch.clamp(self.gen_step, max=self.max_seq_len - 1)
        text = torch.where(self.gen_step < self.trailing_len, self.trailing.index_select(1, text_idx), self.tts_pad)
        inputs_embeds = inputs_embeds + text

        mask = torch.zeros(span, dtype=self.dtype, device=self.device)
        mask = mask.masked_fill(self.key_positions[:span] > self.pos, torch.finfo(self.dtype).min).view(1, 1, 1, -1)
        position_ids = self.pos.view(1, 1, 1).expand(3, 1, 1)
        out = self._talker_forward(inputs_embeds=inputs_embeds, attention_mask=mask,
                                   past_key_values=self.talker_views[span],
                                   cache_position=self.pos, position_ids=position_ids, use_cache=True)
        hidden = out.last_hidden_state
        logits = talker.codec_head(hidden[:, -1]).float()

        penalized = torch.where(logits < 0, logits * self.rep_penalty, logits / self.rep_penalty)
        logits = torch.where(self.presence, penalized, logits)
        logits = logits.masked_fill(self.eos_mask & (self.n_generated < MIN_NEW_TOKENS), float("-inf"))
        logits = logits.masked_fill(self.suppress_mask, float("-inf"))
        next_tok = _sample(logits, do_sample, self.temperature, top_k, top_p)

        self.codes.index_copy_(0, self.step, torch.cat(frame).view(1, -1))
        self.token.copy_(next_tok)
        self.past_hidden.copy_(hidden)
        self.presence.index_fill_(0, next_tok.view(1), True)
        self.n_generated.add_(1)
        self.pos.add_(1)
        self.gen_step.add_(1)
        self.step.add_(1)

    def _graph(self, key: tuple, span: int) -> torch.cuda.CUDAGraph:
        graph = self._graphs.get((key, span))
        if graph is not None:
            return graph
        logger.info(f"Capturing CUDA graph for decode step (sampling {key}, attention span {span})")
        # Warm up on a side stream (allocator and cuBLAS workspaces), then capture.
        # The warmup runs mutate the state buffers; every request resets them.
        self.pos.fill_(1)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.step.zero_()
                self._step(key, span)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)

        graph = torch.cuda.CUDAGraph()
        self.step.zero_()
        with torch.cuda.graph(graph, pool=self._pool):
            self._step(key, span)
        self._pool = graph.pool()
        torch.cuda.synchronize(self.device)
        self._graphs[(key, span)] = graph
        return graph

    # --------------------------------------------------------------- public

    @torch.inference_mode()
    def warmup(self, **sampling) -> None:
        """Capture the graphs for the given (default) sampling settings up front."""
        key = self._key(**self._sampling(sampling))
        for span in self.spans:
            self._graph(key, span)

    def _sampling(self, kwargs: dict) -> dict:
        merged = self.tts._merge_generate_kwargs(**kwargs)
        return {k: merged[k] for k in (
            "do_sample", "top_k", "top_p", "temperature", "repetition_penalty",
            "subtalker_dosample", "subtalker_top_k", "subtalker_top_p", "subtalker_temperature",
            "max_new_tokens")}

    @staticmethod
    def _key(do_sample, top_k, top_p, subtalker_dosample, subtalker_top_k, subtalker_top_p, **_) -> tuple:
        return (bool(do_sample), int(top_k), float(top_p),
                bool(subtalker_dosample), int(subtalker_top_k), float(subtalker_top_p))

    def _prefill_body(self, embeds, mask, cache_position, position_ids):
        """Talker forward over the prompt into the static cache -> (last hidden, float logits)."""
        cache = self.talker_views[mask.shape[-1]]
        out = self.talker.model(inputs_embeds=embeds, attention_mask=mask, past_key_values=cache,
                                cache_position=cache_position, position_ids=position_ids, use_cache=True)
        hidden = out.last_hidden_state[:, -1:]
        return hidden, self.talker.codec_head(hidden[:, -1]).float()

    def _prefill_inputs(self, length: int):
        span = self.span_for(length - 1)
        cache_position = torch.arange(length, device=self.device)
        mask = torch.zeros(length, span, dtype=self.dtype, device=self.device)
        mask = mask.masked_fill(self.key_positions[:span].unsqueeze(0) > cache_position.unsqueeze(1),
                                torch.finfo(self.dtype).min)
        position_ids = cache_position.view(1, 1, -1).expand(3, 1, -1)
        return mask.view(1, 1, length, -1), cache_position, position_ids

    def _prefill_graph(self, length: int):
        """Captured prefill for one prompt length, made on first use.

        Voice cloning with an x-vector feeds the text during decoding, so its
        prompt has the same few positions for every request; eagerly that
        forward costs ~30 ms of launch overhead for ~10 positions.
        """
        entry = self._prefill_graphs.get(length)
        if entry is not None:
            return entry
        embeds = torch.zeros(1, length, self.hidden, dtype=self.dtype, device=self.device)
        args = self._prefill_inputs(length)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                self._prefill_body(embeds, *args)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            hidden, logits = self._prefill_body(embeds, *args)
        self._pool = graph.pool()
        # The graph reads `args` by address, so the entry keeps them alive
        entry = self._prefill_graphs[length] = (graph, embeds, hidden, logits, args)
        return entry

    def _prefill(self, inputs: dict, sampling: dict) -> int:
        """Run the prompt into the static cache and seed the per-step state."""
        embeds, _, trailing, tts_pad = self.model.build_talker_inputs(**inputs)
        if embeds.shape[0] != 1:
            raise ValueError("FastGenerator handles one request at a time")
        length = embeds.shape[1]
        if length + 2 >= self.max_seq_len:
            raise ValueError(f"Prompt of {length} positions does not fit max_seq_len={self.max_seq_len}")

        # A single unpadded prompt: positions 0..length-1 on all three mRoPE
        # axes and no rope offset, which is what the stock prefill computes too.
        if length <= MAX_GRAPHED_PREFILL:
            graph, embeds_buf, hidden, logits, _ = self._prefill_graph(length)
            embeds_buf.copy_(embeds)
            graph.replay()
        else:
            hidden, logits = self._prefill_body(embeds, *self._prefill_inputs(length))
        logits = logits.masked_fill(self.eos_mask, float("-inf"))  # min_new_tokens: nothing generated yet
        logits = logits.masked_fill(self.suppress_mask, float("-inf"))
        first = _sample(logits, sampling["do_sample"], sampling["temperature"],
                        sampling["top_k"], sampling["top_p"])

        text_len = trailing.shape[1]
        self.trailing[:, :text_len].copy_(trailing)
        self.trailing_len.fill_(text_len)
        self.tts_pad.copy_(tts_pad)
        self.token.copy_(first)
        self.past_hidden.copy_(hidden)
        self.pos.fill_(length)
        self.gen_step.zero_()
        self.n_generated.fill_(1)
        self.step.zero_()
        self.presence.zero_()
        self.presence.index_fill_(0, first.view(1), True)
        self.temperature.fill_(sampling["temperature"])
        self.sub_temperature.fill_(sampling["subtalker_temperature"])
        self.rep_penalty.fill_(sampling["repetition_penalty"])
        return length

    @torch.inference_mode()
    def stream_codes(self, inputs: dict, chunk_frames: Union[int, Sequence[int]] = 1,
                     **kwargs) -> Iterator[torch.Tensor]:
        """Yield codec frames ([n, 16] long, on the GPU) as they are generated.

        `inputs` are the `model.generate` inputs from one of the wrapper's
        `_prepare_*` methods; `kwargs` are sampling overrides. `chunk_frames` is
        a chunk size or a schedule of sizes whose last entry repeats, e.g.
        (2, 4, 8, 12) for a quick first chunk that grows while playback runs.
        The yielded tensors are views of a buffer that the next request reuses,
        and `self.codes[:n]` always holds every frame generated so far.
        """
        schedule = [chunk_frames] if isinstance(chunk_frames, int) else list(chunk_frames)
        sampling = self._sampling(kwargs)
        key = self._key(**sampling)
        graphs = {span: self._graph(key, span) for span in self.spans}
        length = self._prefill(inputs, sampling)

        limit = min(sampling["max_new_tokens"], self.max_seq_len - length - 1)
        token = self.token
        emitted = frames = chunks = 0
        while frames < limit:
            if token.item() == self.eos:
                break
            graphs[self.span_for(length + frames)].replay()
            frames += 1
            if frames - emitted >= schedule[min(chunks, len(schedule) - 1)]:
                yield self.codes[emitted:frames]
                emitted = frames
                chunks += 1
        if frames >= limit:
            logger.warning(f"Stopped at {frames} frames: prompt of {length} positions hit max_seq_len={self.max_seq_len}")
        if frames > emitted:
            yield self.codes[emitted:frames]

    @torch.inference_mode()
    def generate_codes(self, inputs: dict, **kwargs) -> torch.Tensor:
        """All codec frames of one request, [T, 16] long on the GPU."""
        chunks = list(self.stream_codes(inputs, chunk_frames=self.max_seq_len, **kwargs))
        return torch.cat(chunks).clone() if chunks else self.codes[:0].clone()


class StreamingDecoder:
    """Turns a growing sequence of codec frames into audio, chunk by chunk.

    Each chunk is decoded together with up to `left_context` earlier frames
    rather than re-decoding the whole utterance. Against a full decode, 25
    frames of context (what upstream `chunked_decode` uses) gave ~20 dB SNR at
    the seams, 72 frames (the decoder transformer's sliding window) ~36 dB,
    which is also how far the bf16 decoder itself is from an fp32 one.

    The decoder's transposed convolutions hold back the last few hundred samples
    of every window until the next frame exists, so a window of n frames yields
    n * upsample - holdback samples; those samples are emitted with the next chunk.

    Short windows are launch-bound (~20 ms eager for 2 or 32 frames), so they
    run as CUDA graphs over zero-padded buffers of `graph_buckets` frames: the
    decoder is causal, and padding on the right changes the kept samples only
    at bf16 rounding level. From ~64 frames on the decoder is compute-bound
    and a graph no longer helps.
    """

    def __init__(self, speech_tokenizer, left_context: int = 72, graph_buckets=(16, 32)):
        self.decoder = speech_tokenizer.model.decoder
        self.upsample = int(speech_tokenizer.get_decode_upsample_rate())
        self.quantizers = self.decoder.config.num_quantizers
        self.left_context = left_context
        self.graph_buckets = sorted(graph_buckets)
        self._graphs = {}
        self._pool = None
        self.holdback = self._measure_holdback()
        self.emitted = 0

    @torch.inference_mode()
    def _measure_holdback(self) -> int:
        codes = torch.zeros(1, self.quantizers, 2, dtype=torch.long, device=self.decoder.device)
        return 2 * self.upsample - self.decoder(codes).shape[-1]

    @torch.inference_mode()
    def capture(self) -> None:
        for frames in self.graph_buckets:
            self._bucket(frames)

    def _bucket(self, frames: int):
        entry = self._graphs.get(frames)
        if entry is not None:
            return entry
        codes = torch.zeros(1, self.quantizers, frames, dtype=torch.long, device=self.decoder.device)
        stream = torch.cuda.Stream(device=codes.device)
        stream.wait_stream(torch.cuda.current_stream(codes.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.decoder(codes)
        torch.cuda.current_stream(codes.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            wav = self.decoder(codes)
        self._pool = graph.pool()
        entry = self._graphs[frames] = (graph, codes, wav)
        return entry

    @torch.inference_mode()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """[T, 16] frames -> T * upsample - holdback samples (1-D, on the GPU)."""
        frames = len(codes)
        bucket = next((b for b in self.graph_buckets if b >= frames), None)
        if bucket is None:
            return self.decoder(codes.T.unsqueeze(0)).view(-1)
        graph, buffer, wav = self._bucket(bucket)
        buffer.zero_()
        buffer[0, :, :frames].copy_(codes.T)
        graph.replay()
        return wav.view(-1)[: frames * self.upsample - self.holdback]

    def reset(self, prefix_frames: int = 0) -> None:
        """Start a new utterance; `prefix_frames` of context (ICL reference codes) are not emitted."""
        self.emitted = prefix_frames * self.upsample

    @torch.inference_mode()
    def push(self, codes: torch.Tensor):
        """All frames so far ([T, 16] on the GPU) -> the audio not yet emitted (float32 numpy)."""
        first = max(0, self.emitted // self.upsample - self.left_context)
        wav = self.decode(codes[first:])
        end = first * self.upsample + wav.shape[-1]
        audio = wav[self.emitted - first * self.upsample:].float().cpu().numpy()
        self.emitted = end
        return audio

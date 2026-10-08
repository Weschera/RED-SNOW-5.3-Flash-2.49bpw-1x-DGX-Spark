#!/usr/bin/env python3
"""Staged native EXL3 OpenAI chat API. No engine code or weights are modified.

Minimal /v1/chat/completions + /v1/models + /health server over exllamav3's own
Generator/Job, one generation worker thread, no engine patch and no weight edit.

Run it through ../serve.sh, which owns every path, pin and sizing value.

Two deliberate choices are in this file and both are documented in the README:
  * the chat-template gate compares WHITESPACE-NORMALIZED text, because this pack's
    chat_template.jinja differs from the chat_template embedded in its
    tokenizer_config.json by a single blank line (upstream inconsistency, not a quant
    artifact) and a strict compare aborts startup; and
  * MiMo turn terminators are appended to the stop set on top of the config's eos.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import socket
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from protocol import APIError, dumps, loads, normalize_request, sampling_options
from glm_protocol import GlmOutputParser as OutputParser, render_prompt, token_budget
from worker import GenerationWorker


def verified_template(hf, model_dir, expected=None):
    # Whitespace-normalized on purpose: this pack's chat_template.jinja carries one extra
    # blank line versus the template embedded in tokenizer_config.json, so a byte-exact
    # compare raises RuntimeError("Model file and embedded tokenizer templates disagree")
    # and the server never starts. Normalizing only collapses whitespace runs - a template
    # that is genuinely different still fails here. The digest below is unchanged by it.
    template = hf.get_chat_template()
    digest = hashlib.sha256(template.encode()).hexdigest()
    path = Path(model_dir) / "chat_template.jinja"
    if path.exists() and ' '.join(path.read_text().split()) != ' '.join(template.split()):
        raise RuntimeError("Loaded template does not match the model chat_template.jinja")
    config_path = Path(model_dir) / "tokenizer_config.json"
    if config_path.exists():
        embedded = json.loads(config_path.read_text()).get("chat_template")
        if isinstance(embedded, str) and ' '.join(embedded.split()) != ' '.join(template.split()):
            raise RuntimeError("Model file and embedded tokenizer templates disagree")
    if expected and expected != digest:
        raise RuntimeError("Model template hash differs from the staged launch profile")
    return digest


def usage(last, prompt_tokens):
    new = int(last.get("new_tokens", 0))
    accepted = int(last.get("accepted_draft_tokens", 0))
    rejected = int(last.get("rejected_draft_tokens", 0))
    elapsed = float(last.get("time_generate", 0))
    return {"prompt_tokens": prompt_tokens, "completion_tokens": new,
            "total_tokens": prompt_tokens + new,
            "decode_tok_s": round(new / elapsed, 2) if elapsed > 0 else 0,
            "draft_accept": accepted / (accepted + rejected) if accepted + rejected else None,
            "time_prefill": last.get("time_prefill", 0), "time_generate": elapsed,
            "time_enqueued": last.get("time_enqueued", 0)}


class Application:
    def __init__(self, hf, tokenizer, worker, sampler_factory, served_name, context_limit,
                 cache_size, template_hash, stop_ids, request_timeout=1800, keepalive=5):
        self.hf, self.tokenizer, self.worker, self.sampler_factory = hf, tokenizer, worker, sampler_factory
        self.served_name, self.context_limit, self.cache_size = served_name, context_limit, cache_size
        self.template_hash, self.stop_ids = template_hash, stop_ids
        self.request_timeout, self.keepalive = request_timeout, keepalive
        # Native encode() changes tokenizer flags. Serialize only CPU prompt
        # preparation, never the full generation or HTTP stream.
        self.prompt_lock = threading.Lock()

    def prepare(self, body):
        request = normalize_request(body, self.served_name)
        with self.prompt_lock:
            prompt, thinking = render_prompt(self.hf, request)
            ids = self.tokenizer.encode(prompt, add_bos=False, encode_special_tokens=True)
        count = int(ids.shape[-1])
        max_new = token_budget(count, request.max_tokens, self.context_limit)
        stops = [] if body.get("ignore_eos") else list(self.stop_ids)
        additional = body.get("stop") or []
        stops.extend([additional] if isinstance(additional, str) else additional)
        kwargs = dict(input_ids=ids, max_new_tokens=max_new, sampler=self.sampler_factory(body),
                      stop_conditions=stops, seed=body.get("seed"), decode_special_tokens=True)
        return request, OutputParser(request, thinking), count, kwargs


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def app(self):
        return self.server.app

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def log_message(self, fmt, *args):
        # Never log request bodies or generated text.
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def send_json(self, status, body):
        raw = dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.wfile.write(raw)
        self.wfile.flush()

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/health", "/v1/health"):
            state = self.app.worker.status()
            self.send_json(200 if state["healthy"] else 503,
                           {"status": "healthy" if state["healthy"] else "unhealthy", **state})
        elif path in ("/models", "/v1/models"):
            self.send_json(200, {"object": "list", "data": [{"id": self.app.served_name,
                "object": "model", "owned_by": "vcruz305-exllamav3",
                "max_model_len": self.app.context_limit, "cache_size": self.app.cache_size,
                "max_active_requests": self.app.worker.max_active,
                "chat_template_sha256": self.app.template_hash}]})
        else:
            self.send_json(404, APIError("Not found", 404).payload())

    def do_POST(self):
        self.close_connection = True
        ticket, started_stream = None, False
        try:
            if urlparse(self.path).path not in ("/v1/chat/completions", "/chat/completions"):
                raise APIError("Not found", 404)
            if self.headers.get("Transfer-Encoding"):
                raise APIError("Send JSON with Content-Length")
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise APIError("Invalid Content-Length") from exc
            if not 0 < size <= 8 * 1024 * 1024:
                raise APIError("Request size must be between 1 byte and 8 MiB", 413)
            raw = self.rfile.read(size)
            if len(raw) != size:
                raise APIError("Incomplete request body")
            try:
                body = loads(raw)
            except (ValueError, UnicodeError) as exc:
                raise APIError("Invalid JSON") from exc
            request, parser, prompt_tokens, kwargs = self.app.prepare(body)
            ticket = self.app.worker.submit(**kwargs)
            cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
            base = {"id": cid, "created": created, "model": self.app.served_name}

            def frame(delta, finish=None):
                self.sse({**base, "object": "chat.completion.chunk", "choices": [
                    {"index": 0, "delta": delta, "finish_reason": finish}]})

            if request.stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self.send_header("Connection", "close")
                self.end_headers()
                started_stream = True
                frame({"role": "assistant", "content": ""})
            deadline = time.monotonic() + self.app.request_timeout
            last_sent = time.monotonic()
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise APIError("Request exceeded the server time limit", 504, "request_timeout")
                try:
                    event = ticket.events.get(timeout=min(self.app.keepalive, remaining))
                except queue.Empty:
                    if request.stream:
                        self.keepalive()
                        last_sent = time.monotonic()
                    continue
                if isinstance(event, APIError):
                    raise event
                deltas = parser.feed(event.get("text") or "")
                if request.stream:
                    for delta in deltas:
                        frame(delta)
                        last_sent = time.monotonic()
                    if time.monotonic() - last_sent >= self.app.keepalive:
                        self.keepalive()
                        last_sent = time.monotonic()
                if not event.get("eos"):
                    continue
                msg, finish, final_deltas = parser.finish(event.get("eos_reason", "stop"))
                stats = usage(event, prompt_tokens)
                if request.stream:
                    for delta in final_deltas:
                        frame(delta)
                    frame({}, finish)
                    if (body.get("stream_options") or {}).get("include_usage"):
                        self.sse({**base, "object": "chat.completion.chunk", "choices": [], "usage": stats})
                    self.sse_done()
                else:
                    self.send_json(200, {**base, "object": "chat.completion", "choices": [
                        {"index": 0, "message": msg, "finish_reason": finish}], "usage": stats})
                print(dumps({"request_id": cid, "finish_reason": finish, "usage": stats}), flush=True)
                break
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            pass  # finally cancels just this request, on the generator thread.
        except Exception as exc:
            if isinstance(exc, APIError):
                error = exc
            else:
                traceback.print_exc()
                error = APIError("Internal server error", 500, "server_error")
            try:
                if started_stream:
                    self.sse(error.payload())
                    self.sse_done()
                else:
                    self.send_json(error.status, error.payload())
            except (BrokenPipeError, ConnectionResetError, socket.timeout):
                pass
        finally:
            if ticket is not None:
                self.app.worker.cancel(ticket)

    def sse(self, obj):
        self.wfile.write(("data: " + dumps(obj) + "\n\n").encode())
        self.wfile.flush()

    def keepalive(self):
        self.wfile.write(b": keep-alive\n\n")
        self.wfile.flush()

    def sse_done(self):
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


class Server(ThreadingHTTPServer):
    daemon_threads = True


def make_server(app, host="127.0.0.1", port=0):
    server = Server((host, port), Handler)
    server.app = app
    return server


def main():
    # Optional source-tree runtime. EXL3_ROOT is set by serve.sh only when the recipe was
    # installed from a fork checkout (--from-source); with the released wheel there is
    # nothing to add and the interpreter's own exllamav3 is used.
    root = os.environ.get("EXL3_ROOT", "").strip()
    if root:
        root_path = Path(root).expanduser()
        if not (root_path / "exllamav3").is_dir():
            print(f"EXL3_ROOT={root_path} is not an exllamav3 checkout", file=sys.stderr)
            raise SystemExit(2)
        sys.path.insert(0, str(root_path))
    # Runtime imports occur only on an explicit launch, never on CPU tests.
    import torch
    from exllamav3 import Generator, Job, model_init
    from exllamav3.generator.sampler import ComboSampler
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(allow_abbrev=False)
    model_init.add_args(parser, cache=True, add_sampling_args=False, add_draft_model_args=True,
                        default_cache_size=65536, default_autosplit_max_batch_size=1)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8096)
    parser.add_argument("--served-model-name", default="RED-SNOW-5.3-FLASH-EXL3")
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--max-active-requests", type=int, default=1)
    parser.add_argument("--max-pending-requests", type=int, default=32)
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--expected-template-sha256")
    args = parser.parse_args()
    if not (1 <= args.max_active_requests <= args.autosplit_max_batch_size):
        parser.error("max-active-requests must be positive and <= autosplit_max_batch_size")
    if not 1 <= args.max_model_len <= args.cache_size or args.max_pending_requests < 0 or args.request_timeout <= 0:
        parser.error("Invalid context, queue, or timeout limits")
    # Occupied port aborts before loading model weights. Never displace a serve.
    with socket.socket() as check:
        check.bind((args.host, args.port))
    hf = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    digest = verified_template(hf, args.model_dir, args.expected_template_sha256)
    print(f"Canonical chat template sha256={digest}", flush=True)
    model, config, cache, tokenizer, draft_model, _, draft_cache = model_init.init(args)
    gen = Generator(model=model, cache=cache, tokenizer=tokenizer,
                    max_batch_size=args.max_active_requests, max_chunk_size=args.chunk_size,
                    draft_model=draft_model, draft_cache=draft_cache,
                    num_draft_tokens=args.num_draft_tokens, ngram_match_min=args.ngram_match_min,
                    dynamic_draft_tokens=args.dynamic_draft, draft_confidence=args.draft_confidence,
                    cpu_cache_size=int(args.cpu_cache_size * 1024**3),
                    recurrent_cache_size=int(args.recurrent_cache_size * 1024**3))
    stops = list(config.eos_token_id_list or [])
    # MiMo turn terminators (Qwen-style vocabulary; config eos = [151645]).
    for name in ("<|user|>", "<|observation|>", "<|endoftext|>"):
        try:
            token_id = tokenizer.single_id(name)
        except Exception:
            token_id = None
        if token_id is not None and token_id not in stops:
            stops.append(token_id)
        stops.append(name)
    print(f"Stop conditions: {stops}", flush=True)

    def sampler(body):
        return ComboSampler(**sampling_options(body))

    worker = GenerationWorker(gen, Job, args.max_active_requests, args.max_pending_requests,
                              inference_context=torch.inference_mode)
    app = Application(hf, tokenizer, worker, sampler, args.served_model_name, args.max_model_len,
                      args.cache_size, digest, stops, args.request_timeout)
    server = make_server(app, args.host, args.port)
    print(f"Native API listening {args.host}:{args.port}; max_active={args.max_active_requests}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        worker.close()


if __name__ == "__main__":
    main()

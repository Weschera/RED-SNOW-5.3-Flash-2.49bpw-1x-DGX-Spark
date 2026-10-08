"""GLM-5.3-Flash / RED-SNOW adapter for Vic's native exllamav3 /v1 server (MiMo recipe server/).

Changes vs protocol.py (Qwen/MiMo): GLM prompt tail (<|assistant|>), GLM tool-call format
<tool_call>NAME<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>, and no default output cap
(budget = whatever the configured window leaves, matching vLLM's uncapped behavior).
Malformed or unknown calls fall back to plain content (vLLM glm47 parser behavior), never a 500.
"""
import re
import uuid

import protocol as P
from protocol import APIError, dumps, loads

CALL = re.compile(r"<tool_call>\s*([^\s<]+)\s*(.*?)</tool_call>", re.S)
ARG = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.S)


def render_prompt(hf_tokenizer, request):
    try:
        prompt = hf_tokenizer.apply_chat_template(request.messages, tools=request.template_tools or None,
                                                 add_generation_prompt=True, tokenize=False,
                                                 **request.template_kwargs)
    except Exception as exc:
        raise APIError(f"Chat template rejected the request: {exc}") from exc
    tail = prompt.rsplit("<|assistant|>", 1)[-1]
    return prompt, tail.rfind("<think>") > tail.rfind("</think>")


def token_budget(prompt_tokens, requested, context_limit, default_output=None):
    # exllamav3 pages are 256 tokens and MTP drafting needs headroom past the last token:
    # keep one page spare so prompt + budget never exceeds the allocated cache.
    remaining = context_limit - prompt_tokens - 256
    if remaining < 1:
        raise APIError("Prompt leaves no output capacity in the configured context")
    return remaining if requested is None else min(requested, remaining)


class GlmOutputParser:
    def __init__(self, request, starts_in_reasoning):
        self.request = request
        self.mode = "reasoning" if starts_in_reasoning else "content"
        self.buf = ""
        self.content, self.reasoning, self.calls = [], [], []
        self.calltext = ""

    def _emit(self, text, events):
        if not text:
            return
        key = "reasoning_content" if self.mode == "reasoning" else "content"
        (self.reasoning if self.mode == "reasoning" else self.content).append(text)
        events.append({key: text})

    def feed(self, text):
        events = []
        self.buf += text
        while self.buf:
            if self.mode == "calls":
                self.calltext += self.buf
                self.buf = ""
                break
            markers = ["<think>", "</think>"] + (["<tool_call>"] if self.mode == "content" and self.request.specs else [])
            hits = [(self.buf.find(m), m) for m in markers if m in self.buf]
            if not hits:
                hold = P.suffix_hold(self.buf, markers)
                end = len(self.buf) - hold
                self._emit(self.buf[:end], events)
                self.buf = self.buf[end:]
                break
            at, marker = min(hits)
            self._emit(self.buf[:at], events)
            self.buf = self.buf[at + len(marker):]
            if marker == "<tool_call>":
                self.mode, self.calltext = "calls", "<tool_call>"
            else:
                self.mode = "reasoning" if marker == "<think>" else "content"
        return events

    def _value(self, spec, key, raw):
        raw = raw.strip("\n")
        if spec.string_parameter(key):
            return raw
        try:
            return loads(raw.strip())
        except (ValueError, TypeError):
            return raw

    def finish(self, eos_reason):
        events = []
        if self.mode != "calls" and self.buf:
            self._emit(self.buf, events)
        self.buf = ""
        leftover = ""
        if self.mode == "calls":
            pos = 0
            for m in CALL.finditer(self.calltext):
                name, body = m.group(1), m.group(2)
                spec = self.request.specs.get(name)
                if spec is None:
                    leftover += m.group(0)
                    continue
                args = {k.strip(): self._value(spec, k.strip(), v) for k, v in ARG.findall(body)}
                call = {"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                        "function": {"name": name, "arguments": dumps(args)}}
                events.append({"tool_calls": [{"index": len(self.calls), **call}]})
                self.calls.append(call)
                pos = m.end()
            rest = self.calltext[pos:] if self.calls else self.calltext
            rest = leftover + rest.strip()
            if rest:
                self.mode = "content"
                self._emit(rest, events)
        limited = eos_reason == "max_new_tokens"
        finish = "length" if limited else ("tool_calls" if self.calls else "stop")
        msg = {"role": "assistant", "content": "".join(self.content) or None}
        if self.reasoning:
            msg["reasoning_content"] = "".join(self.reasoning)
        if self.calls:
            msg["tool_calls"] = self.calls
        return msg, finish, events

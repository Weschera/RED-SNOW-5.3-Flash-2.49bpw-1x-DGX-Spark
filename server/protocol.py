"""CPU-only request/template and incremental Qwen output handling.

No inference runtime imports. The model's local template is authoritative.
Tool requests are descriptions for the client; this module executes no tools.
"""
from __future__ import annotations

import copy
import json
import math
import re
import uuid
from dataclasses import dataclass

from jsonschema import Draft202012Validator, SchemaError, ValidationError
from referencing import Registry


class APIError(Exception):
    def __init__(self, message, status=400, code="invalid_request_error"):
        super().__init__(message)
        self.status, self.code = status, code

    def payload(self):
        return {"error": {"message": str(self), "type": self.code, "code": self.code}}


class GenerationError(APIError):
    def __init__(self, message):
        super().__init__(message, 502, "invalid_model_output")


def loads(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("Non-finite JSON number")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


NAME = re.compile(r"^[^\s<>\"=]+$")


def _local_refs(schema):
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key in ("$ref", "$dynamicRef") and (not isinstance(value, str) or not value.startswith("#")):
                raise APIError("Tool schemas must use local references")
            _local_refs(value)
    elif isinstance(schema, list):
        for value in schema:
            _local_refs(value)


class ToolSpec:
    def __init__(self, tool):
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise APIError("Only function tools are supported")
        function = tool.get("function")
        if not isinstance(function, dict):
            raise APIError("Tool function must be an object")
        self.name = function.get("name")
        if not isinstance(self.name, str) or not NAME.fullmatch(self.name):
            raise APIError("Invalid function name")
        self.schema = function.get("parameters", {"type": "object", "properties": {}})
        if not isinstance(self.schema, dict):
            raise APIError("Tool parameters must be a JSON Schema object")
        _local_refs(self.schema)
        try:
            Draft202012Validator.check_schema(self.schema)
        except SchemaError as exc:
            raise APIError("Invalid tool parameter schema") from exc
        # Explicit registry has no remote retrieval callback.
        self.validator = Draft202012Validator(self.schema, registry=Registry())
        self.tool = copy.deepcopy(tool)
        self.tool["function"]["parameters"] = self.schema

    def property_schema(self, name):
        # Direct-property schemas, including local $defs, cover the usual
        # OpenAI function schema. Full-object validation also handles composed
        # schemas; unknown property types are buffered until parameter close.
        schema = self.schema.get("properties", {}).get(name, {})
        visited = set()
        while isinstance(schema, dict) and "$ref" in schema:
            ref = schema["$ref"]
            if ref in visited or not ref.startswith("#/"):
                break
            visited.add(ref)
            resolved = self.schema
            try:
                for part in ref[2:].split("/"):
                    resolved = resolved[part.replace("~1", "/").replace("~0", "~")]
            except (KeyError, TypeError):
                break
            schema = resolved
        return schema

    def string_parameter(self, name):
        schema = self.property_schema(name)
        return isinstance(schema, dict) and schema.get("type") == "string"

    def value(self, name, raw):
        text = frame_value(raw)
        if self.string_parameter(name):
            return text
        schema = self.schema.get("properties", {}).get(name, {})
        try:
            candidate = loads(text.strip())
        except (ValueError, TypeError):
            candidate = text
        # Prefer typed JSON when it satisfies a declared property; fall back
        # to the raw string for unions/enums that require a string value.
        try:
            if list(self.validator.descend(candidate, schema)):
                if not list(self.validator.descend(text, schema)):
                    return text
        except Exception as exc:
            raise GenerationError("Unable to resolve tool parameter schema") from exc
        return candidate

    def validate(self, args):
        try:
            self.validator.validate(args)
        except ValidationError as exc:
            path = ".".join(map(str, exc.absolute_path)) or "arguments"
            raise GenerationError(f"Tool {self.name}: invalid {path} ({exc.validator})") from exc
        except Exception as exc:
            raise GenerationError(f"Tool {self.name}: unresolvable parameter schema") from exc


def frame_value(raw):
    """Remove only the XML framing newline; preserve code/string whitespace."""
    frame = ""
    if raw.startswith("\r\n"):
        frame = "\r\n"
        raw = raw[2:]
    elif raw.startswith("\n"):
        frame = "\n"
        raw = raw[1:]
    if frame == "\r\n" and raw.endswith("\r\n"):
        raw = raw[:-2]
    elif frame and raw.endswith("\n"):
        raw = raw[:-1]
    return raw


def normalize_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise APIError("messages must be a nonempty list")
    result, call_ids, pending = [], set(), set()
    for original in messages:
        if not isinstance(original, dict):
            raise APIError("Each message must be an object")
        msg = copy.deepcopy(original)
        role = msg.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            raise APIError("Unsupported message role")
        content = msg.get("content")
        if content is None:
            content = ""
        if isinstance(content, list):
            if any(not isinstance(p, dict) or p.get("type") != "text" or not isinstance(p.get("text"), str) for p in content):
                raise APIError("This native endpoint supports text content only")
            content = "".join(p["text"] for p in content)
        if not isinstance(content, str):
            raise APIError("Message content must be text")
        msg["content"] = content
        if role == "system":
            if any(m["role"] != "system" for m in result):
                raise APIError("System messages must precede the conversation")
            if result:
                result[0]["content"] += "\n\n" + content
                continue
        if pending and role != "tool":
            raise APIError("Tool calls must be followed by their tool results")
        if "reasoning_content" in msg and not isinstance(msg["reasoning_content"], (str, type(None))):
            raise APIError("reasoning_content must be text or null")
        calls = msg.get("tool_calls")
        if calls:
            if role != "assistant" or not isinstance(calls, list):
                raise APIError("tool_calls belongs on an assistant message")
            for call in calls:
                if not isinstance(call, dict) or call.get("type") != "function":
                    raise APIError("Invalid historical tool call")
                cid, fn = call.get("id"), call.get("function")
                if not isinstance(cid, str) or not cid or cid in call_ids or not isinstance(fn, dict):
                    raise APIError("Historical tool calls require unique IDs and functions")
                if not isinstance(fn.get("name"), str) or not NAME.fullmatch(fn["name"]):
                    raise APIError("Invalid historical function name")
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = loads(args)
                    except ValueError as exc:
                        raise APIError("Historical tool arguments must be JSON objects") from exc
                if not isinstance(args, dict):
                    raise APIError("Historical tool arguments must be JSON objects")
                fn["arguments"] = args  # HF template expects an object, not JSON text.
                call_ids.add(cid)
                pending.add(cid)
        if role == "tool":
            cid = msg.get("tool_call_id")
            if not isinstance(cid, str) or cid not in pending:
                raise APIError("Tool result must match an unanswered tool_call_id")
            pending.remove(cid)
        result.append(msg)
    if pending:
        raise APIError("Missing tool results")
    if not any(m["role"] == "user" for m in result):
        raise APIError("At least one user message is required")
    return result


@dataclass
class Request:
    body: dict
    messages: list
    specs: dict
    template_tools: list
    template_kwargs: dict
    choice: str
    parallel: bool
    max_tokens: int | None
    stream: bool


def normalize_request(body, served_name):
    if not isinstance(body, dict):
        raise APIError("Request must be a JSON object")
    if body.get("model", served_name) != served_name:
        raise APIError("Unknown model", 404, "model_not_found")
    if body.get("n", 1) != 1:
        raise APIError("Only n=1 is supported")
    for key in ("logprobs", "logit_bias", "functions", "function_call"):
        if body.get(key):
            raise APIError(f"{key} is unsupported")
    fmt = body.get("response_format")
    if fmt is not None and (not isinstance(fmt, dict) or fmt.get("type") != "text"):
        raise APIError("Only response_format.type=text is supported")
    for key in ("stream", "parallel_tool_calls", "ignore_eos"):
        if key in body and not isinstance(body[key], bool):
            raise APIError(f"{key} must be boolean")
    stream_options = body.get("stream_options")
    stream_options = {} if stream_options is None else stream_options
    if not isinstance(stream_options, dict) or not isinstance(stream_options.get("include_usage", False), bool):
        raise APIError("Invalid stream_options")
    messages = normalize_messages(body.get("messages"))
    tools = body.get("tools")
    tools = [] if tools is None else tools
    if not isinstance(tools, list):
        raise APIError("tools must be a list")
    specs = {}
    for tool in tools:
        spec = ToolSpec(tool)
        if spec.name in specs:
            raise APIError("Duplicate tool names")
        specs[spec.name] = spec
    choice = body.get("tool_choice", "auto" if tools else "none")
    instruction = ""
    if isinstance(choice, dict):
        fn = choice.get("function")
        name = fn.get("name") if isinstance(fn, dict) else None
        if choice.get("type") != "function" or name not in specs:
            raise APIError("tool_choice names an unavailable function")
        specs = {name: specs[name]}
        instruction = f"For this turn, call the function {name}."
        choice = "required"
    if choice not in ("auto", "none", "required"):
        raise APIError("Unsupported tool_choice")
    if choice == "required":
        if not specs:
            raise APIError("tool_choice=required needs tools")
        instruction = instruction or "For this turn, respond with at least one of the available function calls."
    if choice == "none":
        specs = {}
    if instruction:
        if messages[0]["role"] == "system":
            messages[0]["content"] += "\n\n" + instruction
        else:
            messages.insert(0, {"role": "system", "content": instruction})
    kwargs = copy.deepcopy(body.get("chat_template_kwargs"))
    kwargs = {} if kwargs is None else kwargs
    if not isinstance(kwargs, dict):
        raise APIError("chat_template_kwargs must be an object")
    # vLLM-compatible: ignore template kwargs this GLM template does not use (e.g. thinking_mode)
    kwargs = {k: v for k, v in kwargs.items() if k in {"enable_thinking", "preserve_thinking", "reasoning_effort", "clear_thinking"}}
    for key in ("enable_thinking", "preserve_thinking"):
        if key in kwargs and not isinstance(kwargs[key], bool):
            raise APIError(f"{key} must be boolean")
    if "reasoning_effort" in body:
        kwargs["reasoning_effort"] = body["reasoning_effort"]
    max_tokens = body.get("max_completion_tokens")
    if max_tokens is None:
        max_tokens = body.get("max_tokens")
    if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 1):
        raise APIError("max_tokens must be a positive integer")
    stop = body.get("stop") or []
    if isinstance(stop, str):
        stop = [stop]
    if not isinstance(stop, list) or any(not isinstance(s, str) or not s for s in stop):
        raise APIError("stop must contain nonempty strings")
    if "seed" in body and body["seed"] is not None and type(body["seed"]) is not int:
        raise APIError("seed must be an integer")
    for key, lower, upper in (("temperature", 0, 100), ("top_p", 0, 1), ("min_p", 0, 1),
                               ("presence_penalty", -2, 2), ("frequency_penalty", -2, 2),
                               ("repetition_penalty", 0.01, 100)):
        value = body.get(key)
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper):
            raise APIError(f"Invalid {key}")
    if "top_k" in body and (type(body["top_k"]) is not int or body["top_k"] < 0):
        raise APIError("top_k must be a nonnegative integer")
    return Request(body, messages, specs, [s.tool for s in specs.values()], kwargs, choice,
                   body.get("parallel_tool_calls", True), max_tokens, body.get("stream", False))


def sampling_options(body):
    """Keep original greedy defaults; honor explicit per-request sampling."""
    def pick(key, default):
        value = body.get(key)
        return default if value is None else value
    temp = pick("temperature", 0.0)
    return dict(temperature=temp, top_p=pick("top_p", 1.0),
                top_k=pick("top_k", 1 if temp <= 0 else 0),
                min_p=pick("min_p", 0.0 if temp <= 0 else 0.08),
                rep_p=pick("repetition_penalty", 1.0), pres_p=pick("presence_penalty", 0.0),
                freq_p=pick("frequency_penalty", 0.0), rep_sustain_range=1024,
                rep_decay_range=1024, temp_last=True, adaptive_target=1.0, adaptive_decay=0.9)


def render_prompt(hf_tokenizer, request):
    try:
        prompt = hf_tokenizer.apply_chat_template(request.messages, tools=request.template_tools,
                                                 add_generation_prompt=True, tokenize=False,
                                                 **request.template_kwargs)
    except Exception as exc:
        raise APIError(f"Chat template rejected the request: {exc}") from exc
    # Detect the actual rendered state, including a model-forced thinking mode.
    tail = prompt.rsplit("<|im_start|>assistant\n", 1)[-1]
    starts_in_reasoning = tail.rfind("<think>") > tail.rfind("</think>")
    return prompt, starts_in_reasoning


def token_budget(prompt_tokens, requested, context_limit, default_output=2048):
    remaining = context_limit - prompt_tokens
    if remaining < 1:
        raise APIError("Prompt leaves no output capacity in the configured context")
    if requested is None:
        return min(default_output, remaining)
    if requested > remaining:
        raise APIError("Prompt plus requested output exceeds the configured context")
    return requested


def suffix_hold(text, markers):
    for n in range(min(len(text), max(map(len, markers)) - 1), 0, -1):
        if any(marker.startswith(text[-n:]) for marker in markers):
            return n
    return 0


class OutputParser:
    """Incremental reasoning/content split and schema-checked Qwen calls.

    A call's final JSON brace is withheld until </tool_call> is present and
    its complete arguments validate. Stream fragments are never executable
    authorization: only a terminal tool_calls finish indicates valid calls.
    """
    def __init__(self, request, starts_in_reasoning):
        self.request = request
        self.mode = "reasoning" if starts_in_reasoning else "content"
        self.buf = ""
        self.content, self.reasoning, self.calls = [], [], []
        self.current = None
        self.args = {}
        self.arg_fragments = []
        self.param, self.value_parts = None, []
        self.string_mode, self.string_started, self.string_tail = False, False, ""
        self.string_frame = ""
        self.events = []

    def _text(self, text):
        if not text:
            return
        key = "reasoning_content" if self.mode == "reasoning" else "content"
        (self.reasoning if self.mode == "reasoning" else self.content).append(text)
        self.events.append({key: text})

    def _argument(self, text):
        self.arg_fragments.append(text)
        self.events.append({"tool_calls": [{"index": len(self.calls), "function": {"arguments": text}}]})

    def _open_function(self, name):
        if name not in self.request.specs:
            raise GenerationError("Model requested an unavailable function")
        if self.calls and not self.request.parallel:
            raise GenerationError("Model returned multiple calls with parallel_tool_calls=false")
        self.current = {"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                        "function": {"name": name, "arguments": ""}}
        self.args, self.arg_fragments = {}, ["{"]
        self.events.append({"tool_calls": [{"index": len(self.calls), **self.current,
                            "function": {"name": name, "arguments": "{"}}]})

    def _value_chunk(self, text, final=False):
        self.value_parts.append(text)
        if not self.string_mode:
            return
        piece = self.string_tail + text
        self.string_tail = ""
        if not self.string_started:
            if not final and piece in ("", "\r"):
                self.string_tail = piece
                return
            self.string_started = True
            if piece.startswith("\r\n"):
                self.string_frame = "\r\n"
                piece = piece[2:]
            elif piece.startswith("\n"):
                self.string_frame = "\n"
                piece = piece[1:]
        if final:
            if self.string_frame == "\r\n" and piece.endswith("\r\n"):
                piece = piece[:-2]
            elif self.string_frame and piece.endswith("\n"):
                piece = piece[:-1]
        elif piece.endswith("\r\n"):
            piece, self.string_tail = piece[:-2], "\r\n"
        elif piece.endswith("\n") or piece.endswith("\r"):
            piece, self.string_tail = piece[:-1], piece[-1:]
        if piece:
            self._argument(dumps(piece)[1:-1])
        if final:
            self.string_tail = ""

    def _parameter_close(self):
        self._value_chunk("", final=True)
        spec = self.request.specs[self.current["function"]["name"]]
        value = spec.value(self.param, "".join(self.value_parts))
        self.args[self.param] = value
        self._argument('"' if self.string_mode else dumps(value))
        self.mode = "parameters"

    def _header(self, prefix, pattern):
        self.buf = self.buf.lstrip()
        if ">" not in self.buf:
            if prefix.startswith(self.buf) or self.buf.startswith(prefix):
                return None
            raise GenerationError("Malformed tool-call structure")
        header, self.buf = self.buf.split(">", 1)
        match = re.fullmatch(pattern, header + ">")
        if not match:
            raise GenerationError("Malformed tool-call header")
        return match.group(1)

    def feed(self, text):
        self.events = []
        self.buf += text
        while self.buf:
            if self.mode in ("content", "reasoning"):
                markers = ["<think>", "</think>"]
                if self.mode == "content" and self.request.specs:
                    markers.append("<tool_call>")
                hits = [(self.buf.find(m), m) for m in markers if m in self.buf]
                if not hits:
                    hold = suffix_hold(self.buf, markers)
                    end = len(self.buf) - hold
                    self._text(self.buf[:end])
                    self.buf = self.buf[end:]
                    break
                at, marker = min(hits)
                self._text(self.buf[:at])
                self.buf = self.buf[at + len(marker):]
                self.mode = {"<think>": "reasoning", "</think>": "content", "<tool_call>": "function"}[marker]
            elif self.mode == "function":
                name = self._header("<function=", r"<function=([^\s<>\"=]+)>")
                if name is None:
                    break
                self._open_function(name)
                self.mode = "parameters"
            elif self.mode == "parameters":
                self.buf = self.buf.lstrip()
                if self.buf.startswith("</function>"):
                    self.buf = self.buf[len("</function>"):]
                    self.mode = "outer_close"
                elif "</function>".startswith(self.buf):
                    break
                else:
                    name = self._header("<parameter=", r"<parameter=([^\s<>\"=]+)>")
                    if name is None:
                        break
                    if name in self.args:
                        raise GenerationError("Duplicate tool parameter")
                    self.param, self.value_parts = name, []
                    spec = self.request.specs[self.current["function"]["name"]]
                    self.string_mode = spec.string_parameter(name)
                    self.string_started, self.string_tail = False, ""
                    self.string_frame = ""
                    self._argument(("," if self.args else "") + dumps(name) + ":" + ('"' if self.string_mode else ""))
                    self.mode = "value"
            elif self.mode == "value":
                marker = "</parameter>"
                at = self.buf.find(marker)
                if at < 0:
                    hold = suffix_hold(self.buf, [marker])
                    end = len(self.buf) - hold
                    self._value_chunk(self.buf[:end])
                    self.buf = self.buf[end:]
                    break
                self._value_chunk(self.buf[:at])
                self.buf = self.buf[at + len(marker):]
                self._parameter_close()
            elif self.mode == "outer_close":
                self.buf = self.buf.lstrip()
                marker = "</tool_call>"
                if self.buf.startswith(marker):
                    self.buf = self.buf[len(marker):]
                    spec = self.request.specs[self.current["function"]["name"]]
                    spec.validate(self.args)
                    expected = dumps(self.args)
                    if "".join(self.arg_fragments) + "}" != expected:
                        raise GenerationError("Tool argument stream did not match validated arguments")
                    self._argument("}")
                    self.current["function"]["arguments"] = expected
                    self.calls.append(self.current)
                    self.current = None
                    self.mode = "content"
                elif marker.startswith(self.buf):
                    break
                else:
                    raise GenerationError("Missing tool-call closing wrapper")
        if text and not self.events and self.current is not None:
            self.events.append({"tool_calls": [{"index": len(self.calls), "function": {"arguments": ""}}]})
        return self.events

    def finish(self, eos_reason):
        self.events = []
        partial = self.mode not in ("content", "reasoning")
        if not partial and self.buf:
            if self.request.specs and "<tool_call>".startswith(self.buf) and self.mode == "content":
                partial = True
            else:
                self._text(self.buf)
        self.buf = ""
        limited = eos_reason == "max_new_tokens" or partial
        if self.request.choice == "required" and not self.calls and not limited:
            raise GenerationError("Model did not produce the required tool call")
        finish = "length" if limited else ("tool_calls" if self.calls else "stop")
        msg = {"role": "assistant", "content": "".join(self.content) or None}
        if self.reasoning:
            msg["reasoning_content"] = "".join(self.reasoning)
        if self.calls and not limited:
            msg["tool_calls"] = self.calls
        return msg, finish, self.events

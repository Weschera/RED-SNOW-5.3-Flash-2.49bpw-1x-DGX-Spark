#!/usr/bin/env bash
# Smoke test: plain chat (prints decode tok/s + MTP acceptance) and a tool call.
set -euo pipefail
U="${URL:-http://localhost:8899}/v1/chat/completions"
M=RED-SNOW-5.3-FLASH-EXL3-2.49
curl -s "$U" -H 'Content-Type: application/json' -d '{"model":"'$M'","messages":[{"role":"user","content":"Write a Python LRU cache class with O(1) get/put and unit tests."}],"temperature":0.7,"top_p":0.95,"min_p":0,"chat_template_kwargs":{"enable_thinking":false}}' \
 | python3 -c 'import json,sys;d=json.load(sys.stdin);u=d["usage"];print("chat:",u["completion_tokens"],"tok,",u.get("decode_tok_s"),"tok/s, MTP accept",u.get("draft_accept"))'
curl -s "$U" -H 'Content-Type: application/json' -d '{"model":"'$M'","messages":[{"role":"user","content":"Weather in Paris for 3 days? Use the tool."}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"]}}}],"chat_template_kwargs":{"enable_thinking":false}}' \
 | python3 -c 'import json,sys;c=json.load(sys.stdin)["choices"][0];print("tools:",c["finish_reason"],json.dumps(c["message"].get("tool_calls"))[:200])'

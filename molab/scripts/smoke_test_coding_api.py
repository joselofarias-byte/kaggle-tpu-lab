#!/usr/bin/env python3
"""Test real OpenAI-compatible endpoint for streaming and tool calls.
Requires MOLAB_URL, MOLAB_API_KEY, MODEL_ID. No writes, no tool execution.
"""
import json, os, sys, urllib.request, urllib.error
base=os.environ["MOLAB_URL"].rstrip("/")
if not base.startswith("https://") and not base.startswith("http://127.0.0.1:"):
    sys.exit("Refusing insecure remote URL; HTTPS required")
base=base.removesuffix("/v1")+"/v1"
key=os.environ["MOLAB_API_KEY"]
model=os.environ["MODEL_ID"]
def request(path, payload=None):
    data=None if payload is None else json.dumps(payload).encode()
    req=urllib.request.Request(base+path,data=data,headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"})
    return urllib.request.urlopen(req,timeout=120)
with request("/models") as r:
    models=json.load(r)
assert model in [m["id"] for m in models["data"]], "Model not listed"
print("PASS models")
with request("/chat/completions",{"model":model,"messages":[{"role":"user","content":"Respond only OK"}],"max_tokens":64,"stream":True}) as r:
    assert "text/event-stream" in r.headers.get("Content-Type",""), "Not SSE"
    chunks=0
    for raw in r:
        if raw.startswith(b"data: ") and raw.strip()!=b"data: [DONE]":
            json.loads(raw[6:])
            chunks+=1
    assert chunks>0,"No SSE chunks"
print("PASS streaming")
payload={"model":model,"temperature":0,"max_tokens":256,"stream":False,
 "messages":[{"role":"user","content":"Use the add tool to add 17 and 25. Do not answer directly."}],
 "tools":[{"type":"function","function":{"name":"add","description":"Add two integers","parameters":{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"]}}}],
 "tool_choice":"required"}
with request("/chat/completions",payload) as r:
    response=json.load(r)
calls=response["choices"][0]["message"].get("tool_calls") or []
assert calls, "Model did not produce tool_calls"
assert calls[0]["function"]["name"]=="add",calls
args=json.loads(calls[0]["function"]["arguments"])
assert args.get("a")==17 and args.get("b")==25,args
print("PASS tool calling")
print("READY: model API passed all three gates")

#!/usr/bin/env python3
"""JolufaSoft Molab inference API. Run inside an authenticated GPU Molab session.
Install: pip install 'fastapi>=0.115' 'uvicorn>=0.30' 'transformers==4.57.6' 'accelerate>=1.0' torch safetensors
Set MOLAB_API_KEY and MODEL_ID before starting. Bind to loopback only.
"""
import os, time, secrets, threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-0.6B")
API_KEY = os.environ.get("MOLAB_API_KEY", "")
if len(API_KEY) < 24:
    raise SystemExit("MOLAB_API_KEY must be at least 24 characters")
tokenizer = None
model = None
lock = threading.Lock()

@asynccontextmanager
async def lifespan(app):
    global tokenizer, model
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: select GPU in Molab Compute")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto",
        low_cpu_mem_usage=True, trust_remote_code=False
    )
    model.eval()
    yield
    del model

app = FastAPI(title="JolufaSoft Molab API", lifespan=lifespan)

class Message(BaseModel):
    role: str
    content: str

class Completion(BaseModel):
    model: str
    messages: list[Message]
    temperature: float = Field(default=0.2, ge=0, le=2)
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    stream: bool = False
    tools: list | None = None
    tool_choice: str | dict | None = None

def auth(authorization: str | None):
    if not authorization or not authorization.startswith("Bearer ") or not secrets.compare_digest(authorization[7:], API_KEY):
        raise HTTPException(401, "Invalid API token")

@app.get("/health")
def health(authorization: str | None = Header(default=None)):
    auth(authorization)
    return {"ok": True, "model": MODEL_ID, "gpu": torch.cuda.get_device_name(0)}

@app.get("/v1/models")
def models(authorization: str | None = Header(default=None)):
    auth(authorization)
    return {"object":"list","data":[{"id":MODEL_ID,"object":"model","owned_by":"jolufasoft"}]}

@app.post("/v1/chat/completions")
def completions(req: Completion, authorization: str | None = Header(default=None)):
    auth(authorization)
    if req.stream:
        raise HTTPException(400, "Streaming not implemented")
    if req.tools or req.tool_choice:
        raise HTTPException(400, "Tool calling not implemented by this inference bridge")
    if req.model != MODEL_ID:
        raise HTTPException(404, "Unknown model")
    with lock:
        messages = [m.model_dump() for m in req.messages]
        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except (TypeError, ValueError):
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        kwargs = {"max_new_tokens":req.max_tokens, "do_sample":req.temperature > 0,
                  "pad_token_id":tokenizer.eos_token_id}
        if req.temperature > 0:
            kwargs["temperature"] = req.temperature
        with torch.inference_mode():
            out = model.generate(**inputs, **kwargs)
        n_input = inputs["input_ids"].shape[-1]
        output_tokens = out[0][n_input:]
        answer = tokenizer.decode(output_tokens, skip_special_tokens=True)
    now = int(time.time())
    return {"id":f"chatcmpl-{now}","object":"chat.completion","created":now,"model":MODEL_ID,
            "choices":[{"index":0,"message":{"role":"assistant","content":answer},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":n_input,"completion_tokens":len(output_tokens),"total_tokens":n_input+len(output_tokens)}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("MOLAB_PORT","18080")), log_level="info")

# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.17", "torch>=2.8", "transformers==4.57.6", "accelerate>=1.0", "safetensors", "huggingface-hub>=0.34,<1"]
# ///
import marimo
app = marimo.App(width="full")

@app.cell
def _():
    import marimo as mo
    import torch, time, sys, platform
    return mo, torch, time, sys, platform

@app.cell
def _(mo, torch, sys, platform):
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        report = f"GPU: {p.name} | VRAM: {p.total_memory / 2**30:.2f} GiB | CUDA: {torch.version.cuda}"
    else:
        report = "GPU no disponible: revisá Compute en Molab."
    mo.md(f"# Molab: prueba de GPU e inferencia\n\n{report}\n\nPython {sys.version.split()[0]} | PyTorch {torch.__version__} | {platform.machine()}")
    return

@app.cell
def _(mo):
    start = mo.ui.run_button(label="Probar inferencia Qwen3-0.6B (descarga y ejecución)")
    mo.vstack([mo.md("Prueba pequeña para confirmar acceso a CUDA antes de cargar modelos grandes."), start])
    return start

@app.cell
def _(mo, torch, time, start):
    mo.stop(not start.value, mo.md("Pulsá **Probar inferencia** para comenzar."))
    if not torch.cuda.is_available():
        raise RuntimeError("Molab no entregó una GPU CUDA a este notebook.")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = "Qwen/Qwen3-0.6B"
    t0 = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    loaded = time.monotonic() - t0
    messages = [{"role": "user", "content": "Respondé en español en una oración: ¿qué es CUDA?"}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(rendered, return_tensors="pt").to("cuda")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t1 = time.monotonic()
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=96, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - t1
    tokens = outputs.shape[-1] - inputs["input_ids"].shape[-1]
    answer = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    vram = torch.cuda.max_memory_allocated() / 2**30
    mo.md(f"## Resultado\n\n{answer}\n\n**Carga:** {loaded:.1f} s | **Generación:** {tokens} tokens / {elapsed:.2f} s = {tokens / max(elapsed, 0.001):.1f} tok/s | **Pico VRAM:** {vram:.2f} GiB")
    return

if __name__ == "__main__":
    app.run()

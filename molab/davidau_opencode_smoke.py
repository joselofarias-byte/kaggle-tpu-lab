"""Exercise a real OpenCode read-tool round trip against Molab's local API."""
import argparse, json, os, pathlib, subprocess, sys, time
def main():
    p=argparse.ArgumentParser(); p.add_argument("--base",type=pathlib.Path,default=pathlib.Path("davidau-gguf-lab"))
    a=p.parse_args(); base=a.base.resolve(); work=base/"opencode-smoke"; work.mkdir(exist_ok=True)
    binary=base/"opencode-runtime/node_modules/.bin/opencode"
    fixture=work/"fixture.txt"; sentinel="JOLUFASOFT_MTP_OPENCODE_OK"
    fixture.write_text(sentinel+"\n")
    config=json.loads((base/"opencode.local.json").read_text())
    config["permission"]={"*":"deny","read":"allow"}
    cfg=work/"opencode.json"; cfg.write_text(json.dumps(config,indent=2))
    key=(base/".private/api-key").read_text().strip()
    env=os.environ.copy(); env["MOLAB_API_KEY"]=key; env["OPENCODE_CONFIG"]=str(cfg)
    version=subprocess.check_output([str(binary),"--version"],text=True).strip()
    command=[str(binary),"run","--format","json","-m","molab/jolufasoft-davidau-mtp",
        f"Usá la herramienta read para leer este archivo exacto: {fixture}. Devolvé su contenido sin agregar comentarios."]
    start=time.monotonic()
    result=subprocess.run(command,cwd=work,env=env,capture_output=True,text=True,timeout=180)
    stdout=result.stdout.replace(key,"[REDACTED]"); stderr=result.stderr.replace(key,"[REDACTED]")
    (base/"opencode-smoke-final.jsonl").write_text(stdout)
    (base/"opencode-smoke-final.stderr").write_text(stderr)
    events=[json.loads(line) for line in stdout.splitlines() if line.strip().startswith("{")]
    tools=[e for e in events if e.get("type")=="tool_use"]
    successful=[e for e in tools if e.get("part",{}).get("tool")=="read"
        and e["part"].get("state",{}).get("status")=="completed"
        and sentinel in e["part"]["state"].get("output","")]
    texts=[e.get("part",{}).get("text","") for e in events if e.get("type")=="text"]
    passed=result.returncode==0 and bool(successful) and any(sentinel in t for t in texts)
    summary={"version":version,"returncode":result.returncode,"wall_seconds":time.monotonic()-start,
        "tool_calls":len(tools),"successful_read_calls":len(successful),"final_text":texts,
        "passed":passed,"scope":"OpenCode and llama-server run inside Molab; no external route tested"}
    (base/"opencode-validation.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps(summary,ensure_ascii=False))
    if not passed: sys.exit(1)
if __name__=="__main__": main()

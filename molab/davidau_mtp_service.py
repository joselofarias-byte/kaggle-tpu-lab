"""Authenticated loopback service for the verified Molab DavidAU MTP runtime."""
import argparse, json, os, pathlib, secrets, signal, subprocess, time, urllib.request, urllib.error
MODEL="Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf"
ALIAS="jolufasoft-davidau-mtp"
def paths(base):
    private=base/".private"; private.mkdir(mode=0o700,exist_ok=True); private.chmod(0o700)
    return private,private/"api-key",private/"server.json"
def read_key(base):
    _,key,_=paths(base)
    if not key.exists():
        fd=os.open(key,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,"w") as f: f.write(secrets.token_hex(32))
    key.chmod(0o600)
    return key.read_text().strip()
def req(base,port,path,body=None,auth=True,stream=False):
    headers={"Content-Type":"application/json"}
    if auth: headers["Authorization"]="Bearer "+read_key(base)
    r=urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}"+path,
        data=None if body is None else json.dumps(body).encode(),headers=headers),timeout=180)
    if stream: return r
    with r: return json.load(r)
def alive(state):
    try:
        cmd=pathlib.Path(f"/proc/{state['pid']}/cmdline").read_bytes().decode().split("\0")
        return any(x.endswith("/llama-server") for x in cmd) and str(state["port"]) in cmd
    except (FileNotFoundError,KeyError): return False
def start(base,port,context):
    _,key,statefile=paths(base); read_key(base)
    if statefile.exists():
        state=json.loads(statefile.read_text())
        if alive(state):
            if state["port"]!=port: raise RuntimeError("Existing service uses a different port")
            return state
    binary=base/"llama.cpp/build-cu134/bin/llama-server"
    model=base/"models"/MODEL
    if not binary.is_file() or not model.is_file(): raise FileNotFoundError("Run davidau_mtp_prepare.py first")
    args=[str(binary),"-m",str(model),"--alias",ALIAS,"-ngl","99","-c",str(context),
        "-np","1","-fa","on","--jinja","--reasoning","off","--spec-type","draft-mtp",
        "--spec-draft-n-max","3","--host","127.0.0.1","--port",str(port),
        "--api-key-file",str(key),"--no-webui"]
    with (base/"api-server.log").open("w") as log:
        p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    state={"pid":p.pid,"port":port,"context":context,"alias":ALIAS,"loopback_only":True}
    statefile.write_text(json.dumps(state,indent=2)); statefile.chmod(0o600)
    deadline=time.monotonic()+180
    while time.monotonic()<deadline:
        if p.poll() is not None: raise RuntimeError(f"Server exited {p.returncode}; see api-server.log")
        try: req(base,port,"/health"); return state
        except (OSError,urllib.error.HTTPError): time.sleep(.5)
    p.terminate(); raise TimeoutError("Server startup")
def stop(base):
    _,_,statefile=paths(base)
    if not statefile.exists(): return {"stopped":True}
    s=json.loads(statefile.read_text())
    if alive(s):
        os.kill(s["pid"],signal.SIGTERM)
        for _ in range(40):
            if not alive(s): break
            time.sleep(.25)
        if alive(s): raise RuntimeError("Server did not stop")
    statefile.unlink(missing_ok=True)
    return {"stopped":True}
def test(base,port):
    result={}
    for label,auth,wrong in [("missing",False,False),("invalid",False,True)]:
        try:
            if wrong:
                with urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/models",
                    headers={"Authorization":"Bearer invalid-test-key"}),timeout=10) as r:
                    result[label+"_status"]=r.status
            else:
                req(base,port,"/v1/models",auth=auth); result[label+"_status"]=200
        except urllib.error.HTTPError as e: result[label+"_status"]=e.code
        if result[label+"_status"] not in (401,403): raise AssertionError("Authentication not enforced")
    result["models"]=req(base,port,"/v1/models")
    body={"model":ALIAS,"messages":[{"role":"user","content":"Respondé exactamente: API_MTP_OK"}],
        "temperature":0,"max_tokens":64,"chat_template_kwargs":{"enable_thinking":False}}
    t=time.monotonic(); result["chat"]=req(base,port,"/v1/chat/completions",body)
    result["chat_wall_seconds"]=time.monotonic()-t
    result["stream_chunks"]=0; result["stream_text"]=""; result["stream_done"]=False
    with req(base,port,"/v1/chat/completions",{**body,"stream":True},stream=True) as r:
        for line in r:
            if not line.startswith(b"data: "): continue
            data=line[6:].strip()
            if data==b"[DONE]": result["stream_done"]=True; break
            chunk=json.loads(data)
            result["stream_chunks"]+=1
            for c in chunk.get("choices",[]): result["stream_text"]+=c.get("delta",{}).get("content","") or ""
    assert result["stream_done"] and "API_MTP_OK" in result["stream_text"]
    (base/"api-validation.json").write_text(json.dumps(result,indent=2,ensure_ascii=False))
    return result
def config(base,port,context):
    data={"$schema":"https://opencode.ai/config.json","model":"molab/"+ALIAS,
        "provider":{"molab":{"npm":"@ai-sdk/openai-compatible","name":"Molab DavidAU MTP",
            "options":{"baseURL":f"http://127.0.0.1:{port}/v1","apiKey":"{env:MOLAB_API_KEY}"},
            "models":{ALIAS:{"name":"DavidAU 27B Q4_K_M MTP","tool_call":True,
                "limit":{"context":context,"output":2048}}}}}}
    path=base/"opencode.local.json"; path.write_text(json.dumps(data,indent=2))
    return str(path)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["start","stop","status","test"])
    p.add_argument("--base",type=pathlib.Path,default=pathlib.Path("davidau-gguf-lab"))
    p.add_argument("--port",type=int,default=18993)
    p.add_argument("--context",type=int,default=32768)
    a=p.parse_args(); base=a.base.resolve(); base.mkdir(parents=True,exist_ok=True)
    if a.action=="start":
        s=start(base,a.port,a.context); config(base,a.port,a.context); print(json.dumps(s))
    elif a.action=="stop": print(json.dumps(stop(base)))
    elif a.action=="test":
        d=test(base,a.port)
        print(json.dumps({"auth_missing":d["missing_status"],"auth_invalid":d["invalid_status"],
            "stream_done":d["stream_done"],"stream_text":d["stream_text"],
            "chat_wall_seconds":d["chat_wall_seconds"]},ensure_ascii=False))
    else:
        _,_,sf=paths(base); s=json.loads(sf.read_text()) if sf.exists() else {}
        print(json.dumps({"running":alive(s),**s}))
if __name__=="__main__": main()

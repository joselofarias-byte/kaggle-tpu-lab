"""Prepare private Tailscale access to the authenticated Molab API.
No install, login, or network exposure happens merely by importing this file.
Run install/login/serve only after the user approves joining their tailnet.
"""
import argparse, hashlib, json, os, pathlib, subprocess, tarfile, time, urllib.request
VERSION="1.104.5"
URL=f"https://pkgs.tailscale.com/stable/tailscale_{VERSION}_amd64.tgz"
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["install","daemon","login","serve","status","unserve"])
    p.add_argument("--base",type=pathlib.Path,default=pathlib.Path("davidau-gguf-lab"))
    a=p.parse_args(); base=a.base.resolve(); private=base/".private"; private.mkdir(mode=0o700,exist_ok=True)
    bindir=base/"tailscale-runtime"; bindir.mkdir(exist_ok=True)
    socket=private/"tailscaled.sock"
    ts=[str(bindir/"tailscale"),"--socket",str(socket)]
    def run(args,timeout=60):
        return subprocess.run(args,check=True,capture_output=True,text=True,timeout=timeout)
    if a.action=="install":
        archive=bindir/"tailscale.tgz"
        with urllib.request.urlopen(URL,timeout=120) as r: data=r.read()
        with urllib.request.urlopen(URL+".sha256",timeout=30) as r: expected=r.read().decode().split()[0]
        if hashlib.sha256(data).hexdigest()!=expected: raise RuntimeError("Tailscale archive checksum mismatch")
        archive.write_bytes(data)
        with tarfile.open(archive,"r:gz") as tar:
            for name in ["tailscale","tailscaled"]:
                member=next(m for m in tar.getmembers() if pathlib.PurePosixPath(m.name).name==name and m.isfile())
                with tar.extractfile(member) as src: (bindir/name).write_bytes(src.read())
                (bindir/name).chmod(0o700)
        print(run(ts+["version"]).stdout)
    elif a.action=="daemon":
        if socket.exists(): print("Socket exists; inspect status before starting another daemon"); return
        with (private/"tailscaled.log").open("w") as log:
            proc=subprocess.Popen([str(bindir/"tailscaled"),"--tun=userspace-networking",
                "--state="+str(private/"tailscaled.state"),"--socket="+str(socket)],
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (private/"tailscaled.pid").write_text(str(proc.pid))
        for _ in range(50):
            if socket.exists(): print("Daemon ready"); return
            if proc.poll() is not None: raise RuntimeError("Daemon exited; see private log")
            time.sleep(.1)
        raise TimeoutError("Tailscale socket")
    elif a.action=="login":
        # No SSH service, advertised routes, exit node, or route/DNS imports.
        result=subprocess.run(ts+["up","--hostname=jolufasoft-molab-llm",
            "--accept-routes=false","--accept-dns=false","--timeout=10s"],
            capture_output=True,text=True,timeout=20)
        log=private/"tailscale-login.txt"; log.write_text(result.stdout+"\n"+result.stderr); log.chmod(0o600)
        print("Login instructions saved privately. Read them only for an authenticated browser handoff.")
    elif a.action=="serve":
        state=json.loads(run(ts+["status","--json"]).stdout)
        if state.get("BackendState")!="Running": raise RuntimeError("Complete Tailscale login first")
        # Serve is tailnet-only. Never use Funnel here.
        result=run(ts+["serve","--bg","--https=443","http://127.0.0.1:18993"])
        print(result.stdout)
        print(run(ts+["serve","status"]).stdout)
    elif a.action=="unserve":
        print(run(ts+["serve","--https=443","off"]).stdout)
    else:
        s=json.loads(run(ts+["status","--json"]).stdout)
        print(json.dumps({"backend":s.get("BackendState"),"self":s.get("Self",{}).get("DNSName"),
            "ips":s.get("TailscaleIPs")},indent=2))
if __name__=="__main__": main()

# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface-hub>=0.34,<1"]
# ///
"""Prepare the pinned Molab CUDA/MTP experiment without exposing any server."""
import argparse, hashlib, json, os, pathlib, shutil, subprocess, sys, time
REPO="DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF"
REV="b9bce7e859c9b5047aca884e5bff76f8a0443de1"
FILE="Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf"
SHA="bc7a6cf2bcc78d1190aaf04d1ab1c5cb845b6ff23aa0e7d24fe0d2ea6d3a7c7c"
COMMIT="cd959bae99ff4bed22bef2604f45241c09bbb8c8"
def run(args,**kw):
    subprocess.run([str(x) for x in args],check=True,**kw)
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--base",type=pathlib.Path,default=pathlib.Path("davidau-gguf-lab"))
    a=p.parse_args(); base=a.base.resolve(); base.mkdir(parents=True,exist_ok=True)
    from huggingface_hub import hf_hub_download
    start=time.monotonic()
    model=pathlib.Path(hf_hub_download(REPO,FILE,revision=REV,local_dir=base/"models"))
    with model.open("rb") as f: digest=hashlib.file_digest(f,"sha256").hexdigest()
    if model.stat().st_size!=18498573856 or digest!=SHA: raise RuntimeError("GGUF size/hash mismatch")
    (base/"mtp-model.json").write_text(json.dumps(dict(path=str(model),bytes=model.stat().st_size,sha256=digest,download_hash_seconds=time.monotonic()-start),indent=2))
    run(["uv","pip","install","--python",sys.executable,"cmake==4.4.4","ninja==1.13.2",
         "nvidia-cuda-nvcc==13.4.92","nvidia-cuda-crt==13.4.92","nvidia-cuda-runtime==13.4.92",
         "nvidia-nvvm==13.4.92","nvidia-cuda-nvrtc==13.4.92","nvidia-cublas==13.8.0.4","nvidia-cuda-cccl==13.3.4.3.1"])
    roots=[pathlib.Path(x)/"nvidia/cu13" for x in sys.path if x and pathlib.Path(x).is_dir()]
    cuda=next(x for x in roots if (x/"bin/nvcc").exists() and (x/"include/cccl").exists())
    for lib in (cuda/"lib").glob("*.so.*"):
        link=lib.with_name(lib.name.split(".so.")[0]+".so")
        if not link.exists(): link.symlink_to(lib.name)
    if not (cuda/"lib64").exists(): (cuda/"lib64").symlink_to("lib",target_is_directory=True)
    src=base/"llama.cpp"
    if not (src/".git").exists():
        if src.exists():
            backup=base/("llama.cpp.sandbox-backup-"+str(time.time_ns()))
            src.rename(backup)
            print("Preserved incomplete sandbox sources:", backup, flush=True)
        run(["git","clone","--depth","1","--branch","v0.9.0","https://github.com/ggml-org/llama.cpp.git",src])
    actual=subprocess.check_output(["git","-C",str(src),"rev-parse","HEAD"],text=True).strip()
    if actual!=COMMIT: raise RuntimeError("Unexpected llama.cpp revision: "+actual)
    env=os.environ.copy(); env["PATH"]=str(pathlib.Path(sys.executable).parent)+":"+str(cuda/"bin")+":"+env["PATH"]
    env["CUDACXX"]=str(cuda/"bin/nvcc")
    with (base/"build-resume.log").open("w") as log:
        run(["cmake","-S",src,"-B",src/"build-cu134","-G","Ninja","-DGGML_CUDA=ON","-DCMAKE_CUDA_ARCHITECTURES=120","-DCUDAToolkit_ROOT="+str(cuda),"-DLLAMA_CURL=OFF","-DLLAMA_BUILD_TESTS=OFF"],env=env,stdout=log,stderr=subprocess.STDOUT)
        run(["cmake","--build",src/"build-cu134","-j","4","--target","llama-cli","llama-bench","llama-server"],env=env,stdout=log,stderr=subprocess.STDOUT)
    shutil.copy2(pathlib.Path(__file__).with_name("davidau_mtp_validate.py"),base/"validate_mtp.py")
    run([sys.executable,base/"validate_mtp.py"])
if __name__=="__main__": main()

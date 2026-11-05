import os,sys,subprocess,pathlib,json,traceback
base=pathlib.Path(sys.argv[1]); os.chdir(base)
try:
    def run(args):
        print('RUN',args,flush=True); subprocess.run(args,check=True)
    if not pathlib.Path('llama.cpp/.git').exists():
        run(['git','clone','--depth','1','--branch','v0.9.0','https://github.com/ggml-org/llama.cpp.git','llama.cpp'])
    cuda='/tmp/uv-venv/lib/python3.13/site-packages/nvidia/cu13'
    os.environ['PATH']=str(pathlib.Path(sys.executable).parent)+':'+cuda+'/bin:'+os.environ['PATH']
    os.environ['CUDACXX']=cuda+'/bin/nvcc'
    run(['cmake','-S','llama.cpp','-B','llama.cpp/build-cu134','-G','Ninja','-DGGML_CUDA=ON','-DCMAKE_CUDA_ARCHITECTURES=120','-DCUDAToolkit_ROOT='+cuda,'-DLLAMA_CURL=OFF','-DLLAMA_BUILD_TESTS=OFF'])
    run(['cmake','--build','llama.cpp/build-cu134','-j','4','--target','llama-cli','llama-bench','llama-server'])
    pathlib.Path('build-status.json').write_text(json.dumps({'ok':True}))
except Exception as e:
    traceback.print_exc(); pathlib.Path('build-status.json').write_text(json.dumps({'ok':False,'error':str(e)}))

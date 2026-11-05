"""Measure the exact DavidAU Q4_K_M MTP GGUF with a locally built CUDA runtime."""
import json, os, pathlib, secrets, subprocess, sys, threading, time, urllib.request, urllib.error, traceback
BASE=pathlib.Path(__file__).resolve().parent
OUT=BASE/'mtp-evidence'; OUT.mkdir(exist_ok=True)
BIN=BASE/'llama.cpp/build-cu134/bin'
MODEL=BASE/'models/Qwen3.8-27B-TurboFCFusion-735-882-Here-Uncen-NEO-CODER-MAX-MTP-Q4_K_M.gguf'
def save(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False))
def command(args,name,timeout=300):
    p=subprocess.run([str(a) for a in args],capture_output=True,text=True,timeout=timeout)
    (OUT/name).write_text(p.stdout+'\nSTDERR\n'+p.stderr)
    if p.returncode: raise RuntimeError(f'{name}: exit {p.returncode}: {p.stderr[-2000:]}')
    return p.stdout
KEY=secrets.token_hex(32)
URL='http://127.0.0.1:18993'
def request(path,body=None,auth=True):
    headers={'Content-Type':'application/json'}
    if auth: headers['Authorization']='Bearer '+KEY
    req=urllib.request.Request(URL+path,data=None if body is None else json.dumps(body).encode(),headers=headers)
    with urllib.request.urlopen(req,timeout=180) as r: return json.load(r)
def chat(messages,**kw):
    return request('/v1/chat/completions',dict(messages=messages,max_tokens=160,temperature=0,seed=0,stream=False,chat_template_kwargs={'enable_thinking':False},**kw))
def main():
    deadline=time.monotonic()+3600
    while not all((BIN/x).is_file() for x in ['llama-cli','llama-server','llama-bench']):
        if time.monotonic()>deadline: raise TimeoutError('CUDA build exceeded 60 minutes')
        time.sleep(5)
    while not (BASE/'mtp-model.json').exists(): time.sleep(2)
    save('model.json',json.loads((BASE/'mtp-model.json').read_text()))
    command(['nvidia-smi'],'nvidia-smi.txt')
    command(['df','-h','.'],'disk.txt')
    command(['git','-C',BASE/'llama.cpp','rev-parse','HEAD'],'llama-commit.txt')
    for name in ['llama-cli','llama-server','llama-bench']:
        command([BIN/name,'--help'],name+'-help.txt')
    command([BIN/'llama-server','--list-devices'],'devices.txt')
    command([BIN/'llama-cli','-m',MODEL,'-ngl','99','-c','2048','-n','48','--single-turn','--reasoning','off','-p','Respondé en español: ¿qué es CUDA?'],'cli-spanish.txt')
    bench=command([BIN/'llama-bench','-m',MODEL,'-ngl','99','-p','512','-n','128','-r','3','-o','json'],'bench.txt')
    save('bench.json',json.loads(bench))
    for mode in ['none','draft-mtp']:
        log=(OUT/('server-'+mode+'.log')).open('w')
        args=[str(BIN/'llama-server'),'-m',str(MODEL),'-ngl','99','-c','8192','-np','1','-fa','on','--jinja','--reasoning','off','--host','127.0.0.1','--port','18993','--spec-type',mode,'--spec-draft-n-max','3']
        env=os.environ.copy(); env['LLAMA_API_KEY']=KEY
        p=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            start=time.monotonic()
            while True:
                if p.poll() is not None: raise RuntimeError(f'{mode} server exited {p.returncode}')
                try: request('/health'); break
                except Exception:
                    if time.monotonic()-start>180: raise TimeoutError('server startup')
                    time.sleep(.5)
            load=time.monotonic()-start
            try: request('/v1/models',auth=False); secured=False
            except urllib.error.HTTPError as e: secured=e.code in (401,403)
            samples=[]; stop=threading.Event()
            def sample():
                while not stop.is_set():
                    try: samples.append(int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0]))
                    except Exception: pass
                    stop.wait(.2)
            t=threading.Thread(target=sample,daemon=True); t.start()
            try:
                runs=[]
                for i in range(3):
                    t0=time.monotonic(); r=chat([{'role':'user','content':'Explicá en español en unas 100 palabras por qué ejecutar un LLM localmente puede ser útil.'}]); runs.append({'seconds':time.monotonic()-t0,'response':r})
                save('inference-'+mode+'.json',{'load_seconds':load,'auth_enforced':secured,'runs':runs,'peak_gpu_total_mib':max(samples) if samples else None})
                save('props-'+mode+'.json',request('/props'))
                tools=[{'type':'function','function':{'name':'sumar','description':'Suma dos enteros.','parameters':{'type':'object','properties':{'a':{'type':'integer'},'b':{'type':'integer'}},'required':['a','b'],'additionalProperties':False}}}]
                msgs=[{'role':'user','content':'Usá sumar para sumar 17 y 25 y luego decime el resultado en español.'}]
                first=chat(msgs,tools=tools,tool_choice='required')
                save('tool-syntax-'+mode+'.json',first)
                msg=first['choices'][0]['message']; calls=msg.get('tool_calls',[])
                actual=[]; follow=None
                for call in calls:
                    if call['function']['name']!='sumar': raise ValueError('unapproved tool')
                    a=json.loads(call['function']['arguments'])
                    if set(a)!={'a','b'} or any(type(a[k]) is not int for k in a): raise ValueError('invalid arguments')
                    actual.append({'tool_call_id':call['id'],'result':a['a']+a['b']})
                if actual:
                    msgs.append(msg)
                    msgs.extend({'role':'tool','tool_call_id':a['tool_call_id'],'content':str(a['result'])} for a in actual)
                    follow=chat(msgs,tools=tools,tool_choice='none')
                save('tool-execution-'+mode+'.json',{'syntactic_tool_calls':len(calls),'executed_by':'Python allowlist dispatcher, never by model','executions':actual,'followup':follow})
            finally: stop.set(); t.join(5)
        except Exception as e:
            save('error-'+mode+'.json',{'error':str(e),'traceback':traceback.format_exc()})
        finally:
            p.terminate()
            try: p.wait(20)
            except subprocess.TimeoutExpired: p.kill(); p.wait()
            log.close()
    save('status.json',{'finished':True,'errors':[x.name for x in OUT.glob('error-*.json')]})
if __name__=='__main__':
    try: main()
    except Exception as e:
        save('fatal.json',{'error':str(e),'traceback':traceback.format_exc()}); raise

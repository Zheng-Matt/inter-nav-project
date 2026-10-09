import argparse,datetime,json,os,signal,subprocess,time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--log',required=True); p.add_argument('--timeout',type=float,required=True)
p.add_argument('command',nargs=argparse.REMAINDER)
a=p.parse_args(); command=a.command[1:] if a.command[0]=='--' else a.command
path=Path(a.log); start=time.monotonic()
def snapshot():
    text=subprocess.check_output(['nvidia-smi','-i','0,5,7','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    return {row[0]:{'uuid':row[1],'memory_mib':int(row[2]),'utilization_percent':int(row[3])} for row in (line.split(', ') for line in text.strip().splitlines())}
record={'command':command,'cwd':os.getcwd(),'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'timeout_seconds':a.timeout,'gpu_initial':snapshot()}
peak={index:value['memory_mib'] for index,value in record['gpu_initial'].items()}
with path.open('x') as output,path.with_suffix('.gpu.jsonl').open('x') as telemetry:
    process=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    path.with_suffix('.pid').write_text(str(process.pid))
    record['pid']=process.pid
    path.with_suffix('.running.json').write_text(json.dumps(record,indent=2)+'\n')
    while process.poll() is None:
        gpus=snapshot()
        for index,value in gpus.items(): peak[index]=max(peak[index],value['memory_mib'])
        telemetry.write(json.dumps({'elapsed_seconds':round(time.monotonic()-start,3),'gpus':gpus})+'\n'); telemetry.flush()
        if time.monotonic()-start>a.timeout or path.with_suffix('.stop').exists():
            record['stop_reason']='timeout' if time.monotonic()-start>a.timeout else 'requested_cleanup'
            os.killpg(process.pid,signal.SIGTERM)
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL); process.wait()
            break
        time.sleep(1)
    record.update(exit_code=process.returncode,wall_seconds=round(time.monotonic()-start,3),gpu_peak_total_mib=peak,gpu_final=snapshot(),finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
path.with_suffix('.resource.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)

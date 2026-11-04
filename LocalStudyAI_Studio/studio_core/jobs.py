from __future__ import annotations
from pathlib import Path
import os
import subprocess
import sys
import uuid
from datetime import datetime
from .common import ROOT, JOBS_DIR, atomic_json, read_json

JOB_FILE=JOBS_DIR/'active_job.json'

def status():
    info=read_json(JOB_FILE,{})
    if not info:return {'status':'idle'}
    pid=info.get('pid')
    try:
        import psutil
        proc=psutil.Process(pid) if pid else None
        active=bool(proc and proc.is_running() and proc.status()!=psutil.STATUS_ZOMBIE
                    and (not info.get('created') or abs(proc.create_time()-info['created'])<0.1)
                    and any('studio_indexer.py' in arg for arg in proc.cmdline()))
    except Exception: active=False
    if active:return {**info,'status':'running'}
    log=Path(info.get('log',''))
    txt=log_tail(info,limit=4096) if log.is_file() else ''
    progress=read_json(ROOT/'index'/'studio'/'progress.json',{})
    if info.get('job_id') and progress.get('job_id') != info['job_id']:
        return {**info,'status':'exited_check_log'}
    if progress.get('stage')=='completed':
        return {**info,'status':'completed'}
    if progress.get('stage') in ('error','partial_failure'):
        return {**info,'status':progress['stage']}
    if '저장 완료' in txt or '임베딩 완료!' in txt or '변경 사항 없음' in txt:
        return {**info,'status':'finished_or_exited'}
    return {**info,'status':'exited_check_log'}

def start_job(mode='studio', category='all'):
    from studio_indexer import CATEGORIES
    if category not in ('all',*CATEGORIES):
        raise ValueError(f'지원하지 않는 임베딩 종류: {category}')
    if mode != 'studio':
        raise ValueError('기존 인덱스 보존을 위해 Studio 증분 작업만 허용합니다.')
    existing=status()
    if existing['status']=='running':raise RuntimeError('이미 임베딩 작업이 실행 중입니다.')
    filename='studio_indexer.py' if mode=='studio' else 'build_index.py'
    script=ROOT/'LocalStudyAI_Studio'/filename
    if not script.exists():raise FileNotFoundError(f'{script} 파일이 없습니다')
    JOBS_DIR.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    log=JOBS_DIR/f'index-{mode}-{category}-{stamp}.log'
    env=os.environ.copy()
    env['PYTHONUTF8']='1'
    env['PYTHONIOENCODING']='utf-8'
    env.setdefault('HF_HOME',str(ROOT/'models'/'huggingface'))
    env.setdefault('HF_HUB_CACHE',str(ROOT/'models'/'huggingface'/'hub'))
    env.setdefault('TORCH_HOME',str(ROOT/'models'/'torch'))
    job_id=uuid.uuid4().hex
    cmd=[sys.executable,'-u',str(script),'--job-id',job_id,'--type',category]
    with log.open('wb') as out:
        proc=subprocess.Popen(cmd,cwd=str(ROOT),
            stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,env=env,
            creationflags=(getattr(subprocess,'CREATE_NO_WINDOW',0) if os.name=='nt' else 0))
    try:
        import psutil
        created=psutil.Process(proc.pid).create_time()
    except Exception:
        created=None
    job={'pid':proc.pid,'created':created,'job_id':job_id,'mode':mode,'category':category,'log':str(log),'started_at':datetime.now().isoformat(timespec='seconds')}
    atomic_json(JOB_FILE,job)
    return job

def log_tail(info,limit=32768,max_lines=150):
    filename = info.get('log')
    if not filename:
        return ''
    path=Path(filename)
    if not path.is_file():
        return ''
    with path.open('rb') as stream:
        stream.seek(0,2)
        start=max(0,stream.tell()-limit)
        stream.seek(start)
        raw=stream.read()
    if start>0:
        raw=raw.partition(b'\n')[2]
    # Old logs may be Windows cp949; newly launched children use UTF-8.
    try:
        text=raw.decode('utf-8')
    except UnicodeDecodeError:
        text=raw.decode('cp949',errors='replace')
    return '\n'.join(text.splitlines()[-max_lines:])

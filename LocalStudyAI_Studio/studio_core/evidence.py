"""Optional Xevn review; never changes embeddings or silently drops evidence."""
import json
import os
from pathlib import Path
import subprocess
import sys
from .common import ROOT
from .search import related_text
from .index_lock import index_lock

LABELS={'direct':'직접 근거','partial':'부분 근거','irrelevant':'무관','unknown':'판단 불가'}


def model_path():
    return ROOT/'models'/'Xevn-4B'


def available():
    path=model_path()
    try:
        data=json.loads((path/'model.safetensors.index.json').read_text(encoding='utf-8'))
        return all((path/name).is_file() for name in set(data['weight_map'].values()))
    except (OSError,ValueError,KeyError):
        return False


def apply_judgments(matches, decisions):
    if len(matches)!=len(decisions):
        raise ValueError('Xevn 판정 개수가 일치하지 않습니다')
    reviewed=[]
    for row, decision in zip(matches,decisions):
        answer=decision.get('answer','unknown')
        if answer not in LABELS:
            answer='unknown'
        # Probabilities are not calibrated correctness/confidence for this corpus.
        reviewed.append({**row,'evidence_label':LABELS[answer],'evidence_class':answer})
    rank={'direct':0,'partial':1,'unknown':2,'irrelevant':3}
    return sorted(reviewed,key=lambda r:rank[r['evidence_class']])


def review_evidence(question,matches,all_records,timeout=240):
    if not available():
        raise RuntimeError('Xevn 로컬 모델 설치가 완료되지 않았습니다.')
    payload={'question':question[:2000], 'passages':[
        related_text(row,all_records)[:900] for row in matches[:3]]}
    env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8',
         'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    with index_lock(ROOT/'studio_logs'/'xevn', '다른 Xevn 검토가 실행 중입니다.'):
        proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name('xevn_worker.py')),
                               str(model_path())],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,text=True,encoding='utf-8',env=env,
                              creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            output,_=proc.communicate(json.dumps(payload,ensure_ascii=False),timeout=timeout)
            if proc.returncode:
                raise RuntimeError('Xevn 로컬 실행에 실패했습니다.')
            decisions=json.loads(output)
            return apply_judgments(matches[:3],decisions)+matches[3:]
        except subprocess.TimeoutExpired:
            # Windows venv launchers may own another Python process. Stop the
            # complete owned tree so a timeout cannot leak a loaded 4B model.
            import psutil
            try:
                children=psutil.Process(proc.pid).children(recursive=True)
            except psutil.NoSuchProcess:
                children=[]
            for child in reversed(children):
                try: child.kill()
                except psutil.NoSuchProcess: pass
            proc.kill(); proc.communicate()
            raise RuntimeError('Xevn 검토 시간이 초과되어 기존 검색 결과를 유지합니다.') from None


import os
import sys
import json
import subprocess
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parents[1]


def run_py(script,root):
    env={**os.environ,'LOCALSTUDY_ROOT':str(root),'PYTHONPATH':str(HERE)}
    return subprocess.run([sys.executable,'-u',str(HERE/script)],env=env,capture_output=True,text=True)


def test_legacy_reuse_and_registry(tmp_path):
    root=tmp_path
    (root/'pdfs').mkdir();(root/'videos').mkdir();(root/'index').mkdir()
    (root/'pdfs'/'old.pdf').write_bytes(b'synthetic pdf marker')
    (root/'videos'/'old.mp4').write_bytes(b'synthetic video marker')
    records=[{'kind':'pdf_text','file':'pdfs/old.pdf','page':1,'text':'old pdf','preview':None},
             {'kind':'video','file':'videos/old.mp4','start':0.0,'end':20.0,'text':'video','preview':None}]
    np.save(root/'index'/'vectors.npy',np.array([[1,0,0],[0,1,0]],dtype='float32'))
    (root/'index'/'records.json').write_text(json.dumps(records),encoding='utf8')
    p=run_py('studio_indexer.py',root)
    assert p.returncode==0,(p.stdout,p.stderr)
    assert '기존 재사용 2개 / 신규·수정 0개' in p.stdout
    assert not (root/'index'/'studio'/'vectors.npy').exists()
    assert not (root/'index'/'studio'/'records.json').exists()
    manifest=json.loads((root/'index'/'studio'/'manifest.json').read_text())
    assert len(manifest['files'])==2
    assert {v['mode'] for v in manifest['files'].values()}=={'legacy'}
    # Also verifies existing data was not overwritten.
    assert (root/'index'/'records.json').read_text()==json.dumps(records)


def test_combined_overrides_legacy(tmp_path,monkeypatch):
    # Load separate index with same source and verify studio shadows legacy.
    root=tmp_path; (root/'index'/'studio').mkdir(parents=True)
    (root/'pdfs').mkdir()
    (root/'pdfs'/'book.pdf').write_text('a')
    old=[{'kind':'pdf_text','file':'pdfs/book.pdf','page':1,'text':'before'}]
    new=[{'kind':'pdf_text','file':'pdfs/book.pdf','page':1,'text':'after'}]
    for folder,rows,vec in [(root/'index',old,[[1,0,0]]),(root/'index'/'studio',new,[[0,1,0]])]:
        (folder/'records.json').write_text(json.dumps(rows))
        np.save(folder/'vectors.npy',np.asarray(vec,dtype=np.float32))
    monkeypatch.setenv('LOCALSTUDY_ROOT',str(root))
    for name in tuple(sys.modules):
        if name.startswith('studio_core'):
            del sys.modules[name]
    sys.path.insert(0,str(HERE))
    from studio_core.search import load_combined,search_index
    data=load_combined()
    assert len(data.records)==1 and data.records[0]['text']=='after'
    hits=search_index(data,np.array([0,1,0],dtype=np.float32),{'pdf_text'},k=2)
    assert len(hits)==1 and hits[0]['origin']=='studio'


def test_thinking_only_returns_useful_error(monkeypatch):
    import types
    class Client:
        def __init__(self,**kw):pass
        def show(self,model): return types.SimpleNamespace(capabilities=['completion','thinking'])
        def chat(self,**kw):
            assert kw['think'] is False
            yield {'message':{'content':'','thinking':'internal reasoning'}}
    monkeypatch.setitem(sys.modules,'ollama',types.SimpleNamespace(Client=Client))
    from studio_core.providers import stream_answer
    try:
        list(stream_answer('Qwen3-VL (로컬)','fake-qwen','system','user',[]))
    except RuntimeError as e:
        assert '최종 답변' in str(e)
    else:raise AssertionError('empty response must raise a helpful error')


def test_local_options_only_installed(monkeypatch):
    import types
    from studio_core.providers import installed_models,local_model_options,PROVIDERS
    class Client:
        def __init__(self,**kw): pass
        def list(self):
            return types.SimpleNamespace(models=[types.SimpleNamespace(model=n) for n in ['custom-name:tag','embed-only']])
        def show(self,name):
            return types.SimpleNamespace(capabilities=['completion'] if name=='custom-name:tag' else ['embedding'])
    monkeypatch.setitem(sys.modules,'ollama',types.SimpleNamespace(Client=Client))
    assert installed_models()==['custom-name:tag']
    assert local_model_options(PROVIDERS[0],installed_models())==['custom-name:tag']
    assert local_model_options('Gemini API',installed_models())==[]


def test_gemini_filter_uses_capability():
    from studio_core.providers import is_current_gemini_model
    for name in ['older-version','future-model','custom-name']:
        assert is_current_gemini_model(name,['generateContent'])
        assert not is_current_gemini_model(name,['embedContent'])


def test_gemini_list_api_key_grounded(monkeypatch):
    import types
    from studio_core.providers import available_gemini_models
    holder={}
    class Client:
        def __init__(self,api_key,**kw):
            holder['key']=api_key
            self.models=types.SimpleNamespace(list=lambda:[
                types.SimpleNamespace(name='models/actual-model',supported_actions=['generateContent']),
                types.SimpleNamespace(name='models/embedding-only',supported_actions=['embedContent'])])
        def __enter__(self): return self
        def __exit__(self,*args): holder['closed']=True
    import google
    fake=types.SimpleNamespace(Client=Client)
    monkeypatch.setattr(google,'genai',fake,raising=False)
    monkeypatch.setitem(sys.modules,'google.genai',fake)
    assert available_gemini_models('test-key')==['actual-model']
    assert holder=={'key':'test-key','closed':True}


def test_streamlit_width_migration():
    source=(HERE/'studio_app.py').read_text(encoding='utf-8')
    assert 'use_container_width' not in source
    assert 'RECOMMENDED' not in source
    assert 'width="stretch"' in source

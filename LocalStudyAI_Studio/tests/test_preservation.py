import json
import sys
import types
import numpy as np
import pytest


def test_mmap_overlay_preserves_base_and_query(tmp_path,monkeypatch):
    from studio_core import search
    from studio_core.storage import publish_update
    root=tmp_path; index=root/'index'; studio=index/'studio'
    studio.mkdir(parents=True); (root/'code').mkdir()
    for name in ['a.cs','b.cs']: (root/'code'/name).write_text('text')
    rows=[{'file':'code/a.cs','kind':'source_code','text':'old','chunk_index':0},
          {'file':'code/b.cs','kind':'source_code','text':'untouched','chunk_index':0}]
    np.save(studio/'vectors.npy',np.array([[2,0],[0,3]],dtype=np.float32))
    (studio/'records.json').write_text(json.dumps(rows))
    before={p.name:p.read_bytes() for p in studio.iterdir()}
    publish_update(studio,np.array([[0,1]],dtype=np.float32),[{**rows[0],'text':'new'}])
    monkeypatch.setattr(search,'ROOT',root);monkeypatch.setattr(search,'INDEX_DIR',index)
    monkeypatch.setattr(search,'STUDIO_DIR',studio)
    data=search.load_combined()
    assert {r['text'] for r in data.records}=={'new','untouched'}
    assert all(isinstance(vec,np.memmap) for vec,_ in data.vectors.blocks)
    query=np.array([0,2],dtype=np.float32)
    hits=search.search_index(data,query,{'source_code'},k=6)
    np.testing.assert_array_equal(query,[0,2])
    assert len(hits)==2 and all(abs(x['score']-1)<1e-6 for x in hits)
    assert all((studio/name).read_bytes()==value for name,value in before.items())


def test_changed_file_keeps_old_checkpoint(tmp_path,monkeypatch):
    import studio_indexer as ind
    monkeypatch.setattr(ind,'ROOT',tmp_path)
    monkeypatch.setattr(ind,'STUDIO_PARTS',tmp_path/'parts')
    source=tmp_path/'example.cs';source.write_text('old')
    old=ind.prepare(source);(old/'keep.npz').write_bytes(b'checkpoint')
    source.write_text('changed')
    new=ind.prepare(source)
    assert new!=old
    assert (old/'keep.npz').read_bytes()==b'checkpoint'
    assert ind.prepare(source)==new


@pytest.mark.parametrize('encoding',['utf-8','cp949'])
def test_bounded_korean_logs(tmp_path,encoding):
    from studio_core.jobs import log_tail
    p=tmp_path/'log.txt'
    p.write_bytes(('한글 로그입니다\n'*1000).encode(encoding))
    result=log_tail({'log':str(p)},limit=1024,max_lines=15)
    assert len(result.splitlines())==15
    assert '\ufffd' not in result and '한글 로그입니다' in result
    assert log_tail({})==''


def test_bad_pointer_fails_without_touching_base(tmp_path):
    from studio_core.storage import index_folders
    studio=tmp_path/'studio';studio.mkdir()
    p=studio/'updates.json';p.write_text('["../outside"]')
    with pytest.raises(ValueError): index_folders(tmp_path,studio)
    assert p.read_text()=='["../outside"]'


def test_xevn_labels_are_advisory_and_preserve_every_hit():
    from studio_core.evidence import apply_judgments
    rows=[{'file':'a','score':.9},{'file':'b','score':.8},{'file':'c','score':.7}]
    result=apply_judgments(rows,[{'answer':'irrelevant'},{'answer':'direct'},{'answer':'invalid'}])
    assert [r['file'] for r in result]==['b','c','a']
    assert len(rows)==len(result) and all('evidence_label' not in r for r in rows)
    assert result[1]['evidence_class']=='unknown'


def test_api_failure_never_echoes_key(monkeypatch):
    from studio_core.providers import available_gemini_models
    import google
    class BadClient:
        def __init__(self,**kw): raise ValueError('SECRET_KEY_IN_URL')
    fake=types.SimpleNamespace(Client=BadClient)
    monkeypatch.setattr(google,'genai',fake,raising=False)
    monkeypatch.setitem(sys.modules,'google.genai',fake)
    with pytest.raises(RuntimeError) as caught: available_gemini_models('SECRET_KEY_IN_URL')
    assert 'SECRET' not in str(caught.value)


def test_writer_lock_excludes_second_writer(tmp_path):
    from studio_core.index_lock import index_lock
    with index_lock(tmp_path):
        with pytest.raises(RuntimeError):
            with index_lock(tmp_path): pass
    with index_lock(tmp_path): pass


def test_wrong_dimension_update_does_not_publish(tmp_path):
    from studio_core.storage import publish_update
    studio=tmp_path/'studio';studio.mkdir()
    np.save(studio/'vectors.npy',np.ones((2,3),dtype=np.float32))
    (studio/'records.json').write_text('[{},{}]')
    before=(studio/'vectors.npy').read_bytes()
    with pytest.raises(ValueError): publish_update(studio,np.ones((1,4)),[{}])
    assert not (studio/'updates.json').exists()
    assert (studio/'vectors.npy').read_bytes()==before


def test_xevn_timeout_kills_owned_tree(tmp_path,monkeypatch):
    from studio_core import evidence as e
    monkeypatch.setattr(e,'ROOT',tmp_path)
    monkeypatch.setattr(e,'available',lambda:True)
    killed=[]
    class Child:
        def kill(self): killed.append('child')
    class Proc:
        pid=123
        def __init__(self,*args,**kwargs): self.count=0
        def communicate(self,*args,**kwargs):
            self.count+=1
            if self.count==1: raise e.subprocess.TimeoutExpired('worker',.01)
            return '', ''
        def kill(self): killed.append('parent')
    monkeypatch.setattr(e.subprocess,'Popen',Proc)
    import psutil
    monkeypatch.setattr(psutil,'Process',lambda pid:types.SimpleNamespace(children=lambda recursive:[Child()]))
    with pytest.raises(RuntimeError,match='초과'):
        e.review_evidence('query',[{'text':'evidence'}],[],timeout=.01)
    assert killed==['child','parent']

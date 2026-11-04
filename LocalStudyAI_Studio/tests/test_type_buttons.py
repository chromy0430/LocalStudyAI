"""v0.4: Type-scoped indexing must not erase other types or re-embed old inputs."""
import json
import sys
import types

import numpy as np
import pytest

import studio_indexer as ind
from studio_core import jobs


def test_category_selection_includes_documents_and_code_independently(tmp_path):
    pdf=tmp_path/'pdfs'/'a.pdf'
    vid=tmp_path/'videos'/'a.mp4'
    code=tmp_path/'code'/'Sample.cs'
    doc=tmp_path/'documents'/'readme.md'
    for p in (pdf,vid,code,doc):
        p.parent.mkdir(exist_ok=True)
        p.write_text('demo')
    args=([pdf],[vid],[doc,code])
    assert ind.files_for_category(*args,'pdf') == [pdf]
    assert ind.files_for_category(*args,'video') == [vid]
    assert ind.files_for_category(*args,'code') == [code]
    assert ind.files_for_category(*args,'text') == [doc]
    assert ind.files_for_category(*args,'all') == [pdf,vid,doc,code]
    with pytest.raises(ValueError):
        ind.files_for_category(*args,'audio')


def test_separate_index_runs_preserve_other_vectors_and_deletions(monkeypatch,tmp_path):
    root=tmp_path
    for name in ('pdfs','videos','documents','code'):
        (root/name).mkdir()
    pdf=root/'pdfs'/'Book.pdf';pdf.write_bytes(b'pdf')
    video=root/'videos'/'Film.mp4';video.write_bytes(b'video')
    code=root/'code'/'Player.cs';code.write_text('void Move() {}')
    text=root/'documents'/'Notes.txt';text.write_text('notes')

    for name, value in {
        'ROOT':root,
        'PDF_DIR':root/'pdfs',
        'VIDEO_DIR':root/'videos',
        'DOCUMENT_DIR':root/'documents',
        'CODE_DIR':root/'code',
        'STUDIO_DIR':root/'index'/'studio',
        'STUDIO_PARTS':root/'index'/'studio'/'parts',
    }.items():
        monkeypatch.setattr(ind,name,value)
    monkeypatch.setattr(ind,'legacy_paths',lambda:set())
    history=[]
    monkeypatch.setattr(ind,'set_status',lambda **kw:history.append(kw))

    calls=[]
    def fake_index(model,path,whisper=None):
        folder=ind.prepare(path)
        kind={'pdf':'pdf_text','video':'video','code':'source_code','text':'text_document'}[ind.category_of(path)]
        rec={'kind':kind,'file':path.relative_to(root).as_posix(),
             'text':path.name,'start':0,'end':20,'page':1,'preview':None}
        ind.write_shard(folder/'fragment.npz',np.array([[1,1,1]],dtype=np.float32),[rec])
        calls.append((ind.category_of(path),path.name))
    monkeypatch.setattr(ind,'index_pdf',fake_index)
    monkeypatch.setattr(ind,'index_document',fake_index)
    monkeypatch.setattr(ind,'index_video',fake_index)

    class FakeModel:
        def __init__(self,*a,**kw):pass
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:False)))
    monkeypatch.setitem(sys.modules,'sentence_transformers',types.SimpleNamespace(SentenceTransformer=FakeModel))
    monkeypatch.setitem(sys.modules,'faster_whisper',types.SimpleNamespace(WhisperModel=FakeModel))
    from studio_core import search
    monkeypatch.setattr(search,'ROOT',root)
    monkeypatch.setattr(search,'INDEX_DIR',root/'index')
    monkeypatch.setattr(search,'STUDIO_DIR',root/'index/studio')
    def published():
        return search.load_combined().records

    ind.main('code')
    assert [r['kind'] for r in published()]==['source_code']
    assert calls==[('code','Player.cs')]
    ind.main('code')  # no model loading or indexing required
    assert calls==[('code','Player.cs')]

    ind.main('pdf')
    assert {r['kind'] for r in published()}=={'source_code','pdf_text'}
    ind.main('video')
    ind.main('text')
    assert {r['kind'] for r in published()}=={'source_code','pdf_text','video','text_document'}
    assert len(calls)==4
    assert len(json.loads((root/'index'/'studio'/'manifest.json').read_text())['files'])==4

    # Missing source data remains on disk; the search reader hides the source.
    code.unlink()
    ind.main('pdf')
    assert len(published())==3
    ind.main('code')
    assert {r['kind'] for r in published()}=={'pdf_text','video','text_document'}
    assert len(calls)==4  # no unnecessary model run
    assert ind.part_dir(code).exists()
    assert len(json.loads((root/'index/studio/updates.json').read_text()))==4

    # Even last-source deletion must preserve every published snapshot.
    pdf.unlink();video.unlink();text.unlink()
    ind.main('all')
    assert len(json.loads((root/'index/studio/updates.json').read_text()))==4
    with pytest.raises(ValueError): published()


def test_job_passes_selected_type_to_child(monkeypatch,tmp_path):
    script=tmp_path/'LocalStudyAI_Studio'/'studio_indexer.py'
    script.parent.mkdir()
    script.write_text('# stub')
    monkeypatch.setattr(jobs,'ROOT',tmp_path)
    monkeypatch.setattr(jobs,'JOBS_DIR',tmp_path/'studio_logs')
    monkeypatch.setattr(jobs,'JOB_FILE',tmp_path/'studio_logs'/'active_job.json')
    monkeypatch.setattr(jobs,'status',lambda:{'status':'idle'})
    launched=[]
    class FakeProcess:
        pid=9870
    def fake_popen(command,**kw):
        launched.append(command)
        return FakeProcess()
    monkeypatch.setattr(jobs.subprocess,'Popen',fake_popen)
    result=jobs.start_job(category='video')
    assert launched[0][-2:]==['--type','video']
    assert result['category']=='video'
    assert 'video' in result['log']
    with pytest.raises(ValueError):
        jobs.start_job(category='other')


def test_ui_has_all_type_buttons():
    from pathlib import Path
    text=(Path(__file__).resolve().parents[1]/'studio_app.py').read_text(encoding='utf8')
    for label in ('PDF 임베딩','동영상 임베딩','코드 임베딩','일반 문서 임베딩'):
        assert label in text
    assert "start_job('studio',category=category)" in text
    assert "start_job('studio',category='all')" in text

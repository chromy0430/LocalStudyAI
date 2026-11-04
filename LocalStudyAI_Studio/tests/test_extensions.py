import sys
import types
from pathlib import Path

import numpy as np

from studio_core.search import candidate_id, source_description
from studio_core.credentials import get_gemini_api_key, save_gemini_api_key, delete_gemini_api_key
import studio_indexer as indexer


class StubEncoder:
    def __init__(self):
        self.calls = 0

    def encode(self, inputs, **_):
        self.calls += 1
        return np.ones((len(inputs), 6), dtype=np.float32)


def patch_roots(monkeypatch, root):
    monkeypatch.setattr(indexer, 'ROOT', root)
    monkeypatch.setattr(indexer, 'STUDIO_PARTS', root/'index'/'studio'/'parts')
    monkeypatch.setattr(indexer, 'STUDIO_PREVIEWS', root/'index'/'studio'/'previews')
    monkeypatch.setattr(indexer, 'DOCUMENT_DIR', root/'documents')
    monkeypatch.setattr(indexer, 'CODE_DIR', root/'code')
    monkeypatch.setattr(indexer, 'STUDIO_DIR', root/'index'/'studio')
    monkeypatch.setattr(indexer, 'set_status', lambda **kw: None)


def test_pdf_closed_return_and_resume(monkeypatch, tmp_path):
    from io import BytesIO
    from PIL import Image
    patch_roots(monkeypatch, tmp_path)
    source=tmp_path/'pdfs'/'dummy.pdf'
    source.parent.mkdir(); source.write_bytes(b'stub-pdf')

    class FakePage:
        def get_text(self, fmt):
            assert fmt == 'text'
            return 'A test PDF page with searchable content.'

        def get_pixmap(self, **kw):
            class FakePix:
                def tobytes(self, fmt):
                    bio=BytesIO()
                    Image.new('RGB',(10,10),color='white').save(bio,format='PNG')
                    return bio.getvalue()
            return FakePix()

    class FakePdf:
        def __init__(self): self.closed=False
        def __enter__(self): return self
        def __exit__(self,*args): self.closed=True
        def __len__(self):
            if self.closed: raise ValueError('document closed')
            return 2
        def __getitem__(self,i): return FakePage()

    monkeypatch.setitem(sys.modules,'pymupdf',types.SimpleNamespace(open=lambda path: FakePdf()))
    enc=StubEncoder()
    assert indexer.index_pdf(enc,source)==2
    calls=enc.calls
    assert calls==2
    assert len(list(indexer.part_dir(source).glob('pdf_*.npz')))==2
    # All checkpoints exist. A resumed attempt must not re-embed anything.
    assert indexer.index_pdf(enc,source)==2
    assert enc.calls==calls


def test_code_text_chunks_and_scan(monkeypatch, tmp_path):
    patch_roots(monkeypatch, tmp_path)
    path=tmp_path/'code'/'src'/'PlayerController.cs'
    path.parent.mkdir(parents=True)
    path.write_text('\n'.join(f'int value{i} = {i};' for i in range(103)),encoding='utf8')
    secret=tmp_path/'code'/'src'/'credentials.json'
    secret.write_text('{"api_key":"hidden"}',encoding='utf8')
    ignored=tmp_path/'code'/'bin'/'Ignore.cpp'
    ignored.parent.mkdir(); ignored.write_text('bad')
    document=tmp_path/'documents'/'notes.txt'
    document.parent.mkdir();document.write_text('Korean text 대한민국',encoding='utf8')
    discovered=indexer.scan_documents()
    assert set(discovered)=={path,document}
    enc=StubEncoder()
    assert indexer.index_document(enc,path)>=3
    assert indexer.index_document(enc,document)==1
    shards=sorted(indexer.part_dir(path).glob('doc_*.npz'))
    assert len(shards)>=3
    all_records=[]
    for shard in shards:
        _,records=indexer.read_shard(shard)
        all_records.extend(records)
    assert all(record['kind']=='source_code' and record['file']=='code/src/PlayerController.cs' for record in all_records)
    assert all(record['line_start']>=1 and record['line_end']>=record['line_start'] for record in all_records)
    assert '줄' in source_description(all_records[0])
    assert len({candidate_id(r) for r in all_records})==len(all_records)
    calls=enc.calls
    indexer.index_document(enc,path)
    assert enc.calls==calls


def test_gemini_credentials_never_echo(monkeypatch):
    data={}
    class KeyringFake:
        @staticmethod
        def get_password(service,account): return data.get((service,account))
        @staticmethod
        def set_password(service,account,value): data[(service,account)]=value
        @staticmethod
        def delete_password(service,account): data.pop((service,account),None)
        errors=types.SimpleNamespace(PasswordDeleteError=Exception)
    monkeypatch.delenv('GEMINI_API_KEY',raising=False)
    monkeypatch.setitem(sys.modules,'keyring',KeyringFake)
    monkeypatch.setitem(save_gemini_api_key.__globals__, '_vault', lambda:KeyringFake)
    save_gemini_api_key('local-secret')
    assert get_gemini_api_key()=='local-secret'
    delete_gemini_api_key()
    assert get_gemini_api_key()==''

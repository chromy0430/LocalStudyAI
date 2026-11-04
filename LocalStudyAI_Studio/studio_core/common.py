from __future__ import annotations
from pathlib import Path
import hashlib
import json
import os
import re

ROOT = Path(os.environ.get('LOCALSTUDY_ROOT') or Path(__file__).resolve().parents[2]).resolve()
INDEX_DIR = ROOT / 'index'
PDF_DIR = ROOT / 'pdfs'
VIDEO_DIR = ROOT / 'videos'
DOCUMENT_DIR = ROOT / 'documents'
CODE_DIR = ROOT / 'code'
STUDIO_DIR = INDEX_DIR / 'studio'
STUDIO_PARTS = STUDIO_DIR / 'parts'
STUDIO_PREVIEWS = STUDIO_DIR / 'previews'
STUDIO_CACHE = STUDIO_DIR / 'cache'
JOBS_DIR = ROOT / 'studio_logs'

for folder in (PDF_DIR, VIDEO_DIR, DOCUMENT_DIR, CODE_DIR, INDEX_DIR, STUDIO_DIR, JOBS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

KINDS = {
    'pdf_text': 'PDF 본문', 'pdf_image': 'PDF 그림·도표',
    'transcript': '영상 자막', 'video': '영상 화면', 'audio': '영상 음향',
    'text_document': '텍스트 문서', 'source_code': '소스코드',
}

def within_root(rel_path: str) -> Path:
    resolved = (ROOT / rel_path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f'허용되지 않는 파일 경로: {rel_path}')
    return resolved

def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default
    except (ValueError, OSError):
        return default

def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)

def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def file_key(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode('utf-8')).hexdigest()[:18]

def timecode(seconds: float | None) -> str:
    s = max(0, int(seconds or 0))
    return f'{s // 3600:02}:{(s // 60) % 60:02}:{s % 60:02}'

def safe_upload_name(name: str) -> str:
    name = Path(name.replace('\\', '/')).name
    name = re.sub(r'[^\w.()\- 가-힣]', '_', name, flags=re.UNICODE)
    if not name or name.startswith('.'):
        raise ValueError('사용할 수 없는 파일명입니다')
    return name

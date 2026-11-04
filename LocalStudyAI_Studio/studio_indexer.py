"""Incremental indexer. Keeps prior index/vectors.npy and records.json untouched.

Generated data lives in index/studio. Per PDF page and video segment checkpoints
survive a crash, and all inputs (including waveform) use EmbeddingGemma 2 740M.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
from pathlib import Path
import sys
import traceback

import numpy as np
from studio_core.common import (ROOT, PDF_DIR, VIDEO_DIR, STUDIO_DIR, STUDIO_PARTS,
    STUDIO_PREVIEWS, STUDIO_CACHE, atomic_json, file_key, file_hash)

PDF_PAGES_BATCH = 8
TEXT_BATCH_SIZE = 12
IMAGE_BATCH_SIZE = 2
VIDEO_SECONDS = 20
FRAME_STEP = 3
VIDEO_EXTS = {'.mp4', '.mkv', '.mov', '.m4v', '.webm'}
TEXT_EXTS = {'.txt', '.md', '.rst', '.csv', '.json', '.yaml', '.yml', '.xml'}
CODE_EXTS = {'.cs', '.cpp', '.c', '.cc', '.cxx', '.h', '.hpp', '.hh', '.hxx',
             '.py', '.js', '.jsx', '.ts', '.tsx', '.shader', '.hlsl', '.cginc',
             '.compute', '.glsl'}
DOCUMENT_EXTS = TEXT_EXTS | CODE_EXTS
DOCUMENT_DIR = ROOT / 'documents'
CODE_DIR = ROOT / 'code'
SKIP_DIRS = {'.git', '.venv', '__pycache__', 'node_modules', 'bin', 'obj',
             'library', 'temp', 'logs', 'build', 'builds', '.vs', '.idea'}
SECRET_NAMES = {'.env', '.env.local', '.env.production', 'id_rsa', 'id_ed25519',
                'credentials.json', 'secrets.json', 'appsettings.secrets.json'}
MAX_TEXT_FILE_BYTES = 20 * 1024 * 1024
DOC_MAX_CHARS = 2400
DOC_MAX_LINES = 45
DOC_OVERLAP_LINES = 6
VERSION = 'gemma2-740-full-v1-pdf110-video20-frame3-text700'
STATUS_FILE = STUDIO_DIR / 'progress.json'
# -- v0.4: Run only the selected media family (preserve others) --
CATEGORIES = ('pdf', 'video', 'code', 'text')
CATEGORY_LABELS = {
    'all': '전체', 'pdf': 'PDF', 'video': '동영상',
    'code': '소스코드', 'text': '일반 문서',
}


def category_of(path: Path) -> str | None:
    """Source type follows the extension even if a user placed it in the wrong folder."""
    ext = path.suffix.lower()
    if ext == '.pdf': return 'pdf'
    if ext in VIDEO_EXTS: return 'video'
    if ext in CODE_EXTS: return 'code'
    if ext in TEXT_EXTS: return 'text'
    return None


def files_for_category(pdfs, videos, documents, category: str):
    if category not in (*CATEGORIES, 'all'):
        raise ValueError(f'지원하지 않는 임베딩 종류: {category}')
    sources = pdfs + videos + documents
    return sources if category == 'all' else [p for p in sources if category_of(p) == category]



def set_status(**changes):
    from studio_core.common import read_json
    status = read_json(STATUS_FILE,{})
    status.update(changes)
    atomic_json(STATUS_FILE,status)


def fingerprint(path):
    return VERSION + ':' + file_hash(path)


def split_text(text: str, size=700, overlap=100):
    text = ' '.join(text.split())
    if not text: return []
    step=size-overlap
    return [text[i:i+size] for i in range(0,len(text),step)]


def encode(model, inputs, *, prompt_name=None, batch_size=1):
    if not inputs: return np.zeros((0,768),dtype=np.float32)
    kw = dict(normalize_embeddings=True, batch_size=batch_size, show_progress_bar=False)
    if prompt_name: kw['prompt_name']=prompt_name
    out = np.asarray(model.encode(inputs, **kw),dtype=np.float32)
    return out.reshape(1,-1) if out.ndim == 1 else out


def write_shard(file: Path, vectors, records):
    if not records: return
    data=np.asarray(vectors,dtype=np.float32)
    if data.ndim!=2 or len(records)!=len(data):
        raise ValueError(f'벡터와 레코드 수가 맞지 않습니다: {file.name}')
    file.parent.mkdir(parents=True,exist_ok=True)
    temporary = file.with_name(file.name+'.tmp')
    with temporary.open('wb') as f:
        np.savez(f,vectors=data,metadata=json.dumps(records,ensure_ascii=False))
    os.replace(temporary,file)


def read_shard(file):
    with np.load(file,allow_pickle=False) as data:
        return data['vectors'].astype(np.float32), json.loads(data['metadata'].item())


def scan_documents():
    """Scan only explicitly added document/code folders, skipping build caches/secrets."""
    files = []
    for folder in (DOCUMENT_DIR, CODE_DIR):
        if not folder.is_dir():
            continue
        for base, dirs, names in os.walk(folder):
            dirs[:] = [d for d in dirs if d.lower() not in SKIP_DIRS and not d.startswith('.')]
            for name in names:
                p = Path(base) / name
                if (p.suffix.lower() not in DOCUMENT_EXTS
                        or name.lower() in SECRET_NAMES or name.startswith('.')
                        or p.is_symlink() or not p.is_file()):
                    continue
                files.append(p)
    return sorted(files)


def scan_files():
    pdfs = sorted(p for p in PDF_DIR.rglob('*') if p.is_file() and p.suffix.lower() == '.pdf')
    videos = sorted(p for p in VIDEO_DIR.rglob('*') if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    return pdfs, videos, scan_documents()


def read_source_text(path):
    if path.stat().st_size > MAX_TEXT_FILE_BYTES:
        raise ValueError(f'텍스트/코드 파일이 20MB를 초과합니다: {path.name}')
    content = path.read_bytes()
    if content.startswith((b'\xff\xfe', b'\xfe\xff')):
        return content.decode('utf-16')
    if b'\x00' in content:
        raise ValueError(f'바이너리 파일로 판단하여 제외합니다: {path.name}')
    for encoding in ('utf-8-sig', 'utf-8', 'cp949'):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            pass
    raise ValueError(f'파일 인코딩을 확인할 수 없습니다: {path.name}')


def line_chunks(text):
    """Yield bounded code/text chunks retaining 1-indexed source line numbers."""
    lines = text.splitlines()
    if not lines:
        return
    start = 0
    while start < len(lines):
        end = start
        total = 0
        while end < len(lines) and end - start < DOC_MAX_LINES:
            next_length = len(lines[end]) + 1
            if end > start and total + next_length > DOC_MAX_CHARS:
                break
            total += next_length
            end += 1
        chunk = '\n'.join(lines[start:end]).strip()
        # Prevent accidental multi-kilobyte minified source lines exceeding context limits.
        if chunk:
            if len(chunk) > DOC_MAX_CHARS:
                for pos in range(0, len(chunk), DOC_MAX_CHARS):
                    fragment = chunk[pos:pos + DOC_MAX_CHARS]
                    if fragment.strip():
                        yield start + 1, end, fragment
            else:
                yield start + 1, end, chunk
        if end >= len(lines):
            break
        start = max(start + 1, end - DOC_OVERLAP_LINES)


def index_document(model, path):
    """Incremental indexing for txt/markdown and source code, shard by chunk."""
    folder = prepare(path)
    text = read_source_text(path)
    chunks = list(line_chunks(text))
    if not chunks:
        raise ValueError(f'빈 텍스트/코드 파일입니다: {path.name}')
    relative = path.relative_to(ROOT).as_posix()
    kind = 'source_code' if path.suffix.lower() in CODE_EXTS else 'text_document'
    print(f'FILE: {path.name} | {len(chunks)} 청크', flush=True)
    for start in range(0, len(chunks), TEXT_BATCH_SIZE):
        pending = [(i, item) for i, item in enumerate(chunks[start:start+TEXT_BATCH_SIZE], start)
                   if not (folder / f'doc_{i:06}.npz').is_file()]
        if not pending:
            continue
        payload = [f'file: {path.name} | type: {kind} | lines: {chunk[0]}-{chunk[1]}\n{chunk[2]}'
                   for _, chunk in pending]
        vecs = encode(model, payload, prompt_name='Document', batch_size=TEXT_BATCH_SIZE)
        for (i, (begin, finish, content)), vector in zip(pending, vecs):
            rec = {'kind': kind, 'file': relative, 'page': None, 'start': None, 'end': None,
                   'line_start': begin, 'line_end': finish, 'chunk_index': i,
                   'language': path.suffix.lower().lstrip('.'), 'text': content, 'preview': None}
            write_shard(folder / f'doc_{i:06}.npz', [vector], [rec])
        done = min(start + TEXT_BATCH_SIZE, len(chunks))
        set_status(stage='document', file=path.name, completed=done, total=len(chunks))
        print(f'  {done}/{len(chunks)} 청크 저장 완료', flush=True)
    return len(chunks)


def part_dir(path):
    # Preserve the original v0.4 checkpoints. Changed content gets its own revision.
    from studio_core.common import read_json
    base = STUDIO_PARTS / file_key(path)
    if not path.exists():
        return base
    marker = read_json(base / 'identity.json', {})
    if not base.exists() or marker.get('signature') == fingerprint(path):
        return base
    return base / 'revisions' / fingerprint(path).split(':')[-1]


def prepare(path):
    folder=part_dir(path)
    marker=folder / 'identity.json'
    if not marker.exists():
        folder.mkdir(parents=True,exist_ok=True)
        atomic_json(marker,{'source':str(path.relative_to(ROOT)),
                            'signature':fingerprint(path),'version':VERSION})
    return folder


def index_pdf(model,path):
    import pymupdf
    folder=prepare(path)
    print(f'PDF: {path.name}',flush=True)
    with pymupdf.open(path) as doc:
        total_pages = len(doc)  # Save before PyMuPDF closes the document.
        for batch_start in range(0,total_pages,PDF_PAGES_BATCH):
            batch_indices=[i for i in range(batch_start,min(batch_start+PDF_PAGES_BATCH,total_pages))
                           if not (folder / f'pdf_{i+1:06}.npz').exists()]
            if not batch_indices: continue
            prepared=[]; text_data=[]; image_data=[]
            for page_idx in batch_indices:
                page=doc[page_idx]
                page_text=page.get_text('text') or ''
                texts=split_text(page_text)
                # Revision-specific previews preserve images referenced by older snapshots.
                image_path=STUDIO_PREVIEWS / folder.relative_to(STUDIO_PARTS) / f'{page_idx+1:06}.jpg'
                image_path.parent.mkdir(parents=True,exist_ok=True)
                # Always rerender pending pages. The PDF may have changed even if an
                # old preview with the same file/page name still exists.
                pix=page.get_pixmap(dpi=110,alpha=False)
                from PIL import Image
                from io import BytesIO
                Image.open(BytesIO(pix.tobytes('png'))).convert('RGB').save(image_path,'JPEG',quality=82)
                relative=str(path.relative_to(ROOT))
                meta=[]
                for t in texts:
                    text_data.append(f'title: {path.stem} | text: {t}')
                    meta.append({'kind':'pdf_text','file':relative,'page':page_idx+1,'start':None,'end':None,'text':t,'preview':None})
                image_data.append({'image':str(image_path)})
                preview_record={'kind':'pdf_image','file':relative,'page':page_idx+1,'start':None,'end':None,'text':page_text[:3000], 'preview':str(image_path.relative_to(ROOT))}
                prepared.append((page_idx,meta,preview_record))
            text_vectors=encode(model,text_data,prompt_name='Document',batch_size=TEXT_BATCH_SIZE) if text_data else []
            image_vectors=encode(model,image_data,batch_size=IMAGE_BATCH_SIZE)
            text_pos=0
            for j,(page_idx,meta,preview_record) in enumerate(prepared):
                n=len(meta)
                vs=[text_vectors[text_pos+k] for k in range(n)]
                text_pos+=n
                vs.append(image_vectors[j])
                write_shard(folder / f'pdf_{page_idx+1:06}.npz',vs,meta+[preview_record])
            current=min(batch_start+PDF_PAGES_BATCH,total_pages)
            print(f'  {current}/{total_pages} 페이지 저장 완료',flush=True)
            set_status(stage='pdf',file=path.name,completed=current,total=total_pages)
    return total_pages


def read_frames(cap,start,end):
    import cv2
    from PIL import Image
    frames=[]
    for sec in np.arange(start,end,FRAME_STEP):
        cap.set(cv2.CAP_PROP_POS_MSEC,float(sec*1000))
        good,frame=cap.read()
        if not good: continue
        image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        image.thumbnail((640,640))
        frames.append(image)
    return frames


def extract_audio(path):
    STUDIO_CACHE.mkdir(parents=True,exist_ok=True)
    file_stat=path.stat()
    wav=STUDIO_CACHE / (f'{file_key(path)}_{file_stat.st_mtime_ns}_{file_stat.st_size}.wav')
    if wav.exists() and wav.stat().st_size>1000: return wav
    tmp=wav.with_suffix('.working.wav')
    result=subprocess.run(['ffmpeg','-y','-nostdin','-i',str(path),'-vn','-ac','1','-ar','16000',
        '-c:a','pcm_s16le',str(tmp)],capture_output=True,text=True)
    if result.returncode!=0 or not tmp.exists() or tmp.stat().st_size<1000:
        print('  음성 트랙 없음 또는 ffmpeg 오류 (화면 임베딩은 계속 진행)',flush=True)
        return None
    os.replace(tmp,wav)
    return wav


def index_video(model,path,whisper):
    import cv2
    import soundfile as sf
    folder=prepare(path)
    print(f'VIDEO: {path.name}',flush=True)
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise RuntimeError(f'동영상을 열 수 없음: {path}')
    fps=cap.get(cv2.CAP_PROP_FPS)
    if fps<=0:
        cap.release()
        raise RuntimeError('FPS 정보를 찾지 못했습니다')
    duration=cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps
    audio_path=extract_audio(path)
    transcript_file=folder/'transcript.json'
    from studio_core.common import read_json
    if audio_path and transcript_file.exists():
        subtitles=read_json(transcript_file,[])
    elif audio_path:
        # Pass ndarray to faster-whisper to avoid PyAV av.open path issues
        audio,sr=sf.read(str(audio_path),dtype='float32',always_2d=False)
        if sr!=16000 or audio.ndim!=1: raise ValueError('16 kHz mono audio expected')
        print('  Whisper 전체 오디오 자막 생성 중',flush=True)
        parts,_=whisper.transcribe(audio,beam_size=3,vad_filter=True)
        subtitles=[{'start':s.start,'end':s.end,'text':s.text.strip()} for s in parts]
        atomic_json(transcript_file,subtitles)
        print(f'  자막 {len(subtitles)}개 저장',flush=True)
        del audio
    else:
        subtitles=[]
    wav=sf.SoundFile(str(audio_path)) if audio_path else None
    try:
        for start in np.arange(0,duration,VIDEO_SECONDS):
            end=min(float(start)+VIDEO_SECONDS,float(duration))
            start=float(start)
            seg_id=int(start/VIDEO_SECONDS)
            output=folder/f'video_{seg_id:06}.npz'
            if output.exists(): continue
            meta=[];vectors=[]
            relative=str(path.relative_to(ROOT))
            spoken=' '.join(s['text'] for s in subtitles if s['start']<end and s['end']>start)
            frames=read_frames(cap,start,end)
            if frames:
                vs=encode(model,[{'video':frames}])
                vectors.append(vs[0]); meta.append({'kind':'video','file':relative,'page':None,'start':start,'end':end,'text':spoken,'preview':None})
            if wav is not None:
                wav.seek(min(int(start*wav.samplerate),len(wav)))
                audio=wav.read(int((end-start)*wav.samplerate),dtype='float32')
                if len(audio)>=wav.samplerate:
                    vs=encode(model,[{'audio':audio}])
                    vectors.append(vs[0]);meta.append({'kind':'audio','file':relative,'page':None,'start':start,'end':end,'text':spoken,'preview':None})
            if spoken.strip():
                vs=encode(model,[f'title: {path.stem} | text: {spoken}'],prompt_name='Document')
                vectors.append(vs[0]);meta.append({'kind':'transcript','file':relative,'page':None,'start':start,'end':end,'text':spoken,'preview':None})
            if vectors: write_shard(output,vectors,meta)
            print(f'  {start:.0f}–{end:.0f}초 저장 완료',flush=True)
            set_status(stage='video',file=path.name,completed=end,total=duration)
    finally:
        if wav: wav.close()
        cap.release()


def compact(files):
    """Publish only completed changes, retaining every prior snapshot and shard."""
    from studio_core.storage import publish_update
    combined=[]; records=[]
    for source in files:
        for file in sorted(part_dir(source).glob('*.npz')):
            vec,rec=read_shard(file)
            combined.append(vec); records.extend(rec)
    if not combined:
        return 0
    matrix=np.concatenate(combined,axis=0).astype(np.float32)
    matrix/=np.maximum(np.linalg.norm(matrix,axis=1,keepdims=True),1e-8)
    count=publish_update(STUDIO_DIR,matrix,records)
    print(f'Studio 증분 인덱스 저장 완료: {count}개 (기존 인덱스 보존)',flush=True)
    return count


def legacy_paths():
    old=ROOT/'index'/'records.json'
    if not old.is_file():return set()
    try:
        data=json.loads(old.read_text(encoding='utf-8'))
        return {str(r.get('file','')).replace('\\','/') for r in data if r.get('file')}
    except Exception:
        raise ValueError('기존 인덱스 메타데이터를 읽을 수 없어 자동 인덱싱을 중단합니다.') from None


def source_file_info(path):
    stat=path.stat()
    return {'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}


def update_manifest(manifest):
    atomic_json(STUDIO_DIR/'manifest.json',manifest)


def main(category='all'):
    """Incrementally index only selected category, retaining all other families."""
    if category not in (*CATEGORIES, 'all'):
        raise ValueError(f'지원하지 않는 임베딩 종류: {category}')
    os.environ.setdefault('HF_HOME',str(ROOT/'models'/'huggingface'))
    os.environ.setdefault('HF_HUB_CACHE',str(ROOT/'models'/'huggingface'/'hub'))
    os.environ.setdefault('TORCH_HOME',str(ROOT/'models'/'torch'))
    from studio_core.common import read_json
    set_status(stage='scan',category=category,file='',completed=0,total=0,error='',failed=[])
    pdfs,videos,documents=scan_files()
    all_files=pdfs+videos+documents
    selected=files_for_category(pdfs,videos,documents,category)
    # Missing files are reported while all their persisted data is retained.
    baseline=legacy_paths()
    manifest_path=STUDIO_DIR/'manifest.json'
    manifest=(json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists()
              else {'version':VERSION,'files':{},'failed':{}})
    if not manifest_path.exists() and (STUDIO_DIR/'records.json').exists():
        raise ValueError('기존 Studio 인덱스의 manifest가 없습니다. 재구축하지 않고 중단합니다.')
    if manifest.get('version')!=VERSION:
        raise ValueError('인덱스 버전이 다릅니다. 기존 데이터 보존을 위해 자동 재구축하지 않습니다.')
    entries=manifest.setdefault('files',{})
    failed=manifest.setdefault('failed',{})
    pending=[];unchanged=[]
    active={p.relative_to(ROOT).as_posix() for p in all_files}
    # Missing sources are hidden by the reader, but their manifests and shards stay.
    removed=[rel for rel in entries if rel not in active
             and (category=='all' or category_of(Path(rel))==category)]
    for path in selected:
        rel=path.relative_to(ROOT).as_posix()
        stat=source_file_info(path)
        existing=entries.get(rel)
        if existing and all(existing.get(k)==v for k,v in stat.items()) and existing.get('mode') in ('legacy','studio') and rel not in failed:
            unchanged.append(path);continue
        signature=fingerprint(path)
        if existing and existing.get('signature')==signature and existing.get('mode') in ('legacy','studio') and rel not in failed:
            entries[rel].update(stat);unchanged.append(path);continue
        if existing is None and rel in baseline:
            # Adopt already-indexed legacy data, without re-embedding.
            entries[rel]={'mode':'legacy','signature':signature,**stat}
            unchanged.append(path);continue
        pending.append((path,rel,signature,stat))
    update_manifest(manifest)
    print(f'선택 유형: {CATEGORY_LABELS[category]}',flush=True)
    print(f'대상 파일 {len(selected)}개 / 기존 재사용 {len(unchanged)}개 / 신규·수정 {len(pending)}개 / 원본 미발견 {len(removed)}개 (데이터 보존)',flush=True)
    for f in unchanged:print('건너뛰기:',f.name,flush=True)
    if not pending:
        set_status(stage='completed',category=category,file='',completed=0,total=0,
                   sources=0,new_files=0,skipped=len(unchanged),removed=0,failed=[])
        print('변경 사항 없음: 모든 기존 임베딩 재사용',flush=True)
        return
    model=None
    whisper=None;successful=[];errors=[]
    for path,rel,signature,stat in pending:
        try:
            if model is None:
                import torch
                from sentence_transformers import SentenceTransformer
                device='cuda' if torch.cuda.is_available() else 'cpu'
                print(f'EmbeddingGemma 2 740M: {device}',flush=True)
                model=SentenceTransformer('google/embeddinggemma-2',device=device)
            kind=category_of(path)
            if kind=='pdf':
                index_pdf(model,path)
            elif kind in ('text','code'):
                index_document(model,path)
            elif kind=='video':
                if whisper is None:
                    from faster_whisper import WhisperModel
                    whisper=WhisperModel('small',device='cpu',compute_type='int8')
                index_video(model,path,whisper)
            else:
                raise ValueError(f'지원하지 않는 파일: {path.name}')
            entries[rel]={'mode':'studio','signature':signature,**stat}
            failed[rel]='검색 인덱스 게시 대기'
            successful.append(path)
            update_manifest(manifest)
            print(f'완료: {path.name}',flush=True)
        except Exception as exc:
            errors.append({'file':rel,'error':str(exc)})
            failed[rel]=str(exc)
            update_manifest(manifest)
            print(traceback.format_exc(),file=sys.stderr,flush=True)
    # Publish successful files only; failed replacements keep the prior search data.
    count=compact(successful)
    for path in successful:
        failed.pop(path.relative_to(ROOT).as_posix(),None)
    update_manifest(manifest)
    set_status(stage='completed' if not errors else 'partial_failure',category=category,
        file='',completed=0,total=0,error='',failed=errors,
        sources=len(successful),new_files=len(successful),skipped=len(unchanged),
        removed=len(removed),items=count)
    print(f'{CATEGORY_LABELS[category]} 증분 임베딩 완료!',flush=True)
    if errors:sys.exit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='LocalStudyAI selective incremental embedding')
    parser.add_argument('--job-id',default='manual')
    parser.add_argument('--type','--category',dest='category',default='all',
                        choices=['all',*CATEGORIES],help='인덱싱할 자료 타입만 선택')
    args=parser.parse_args()
    from studio_core.index_lock import index_lock
    with index_lock(STUDIO_DIR):
        set_status(job_id=args.job_id)
        try:main(category=args.category)
        except Exception as exc:
            set_status(stage='error',category=args.category,error=str(exc))
            raise

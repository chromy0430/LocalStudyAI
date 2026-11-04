from __future__ import annotations
import json
import os
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .common import INDEX_DIR, STUDIO_DIR, ROOT, KINDS, read_json, within_root
from .storage import index_folders

@dataclass
class Index:
    vectors: np.ndarray
    records: list[dict]
    sources: list[str]


def candidate_id(record: dict):
    if record.get('kind','').startswith('pdf_'):
        return (record.get('file'),record.get('page'))
    if record.get('kind') in ('text_document','source_code'):
        return (record.get('file'),record.get('chunk_index'),record.get('line_start'),record.get('line_end'))
    return (record.get('file'),record.get('start'),record.get('end'))


def _load_one(folder:Path):
    present=[(folder/name).is_file() for name in ('vectors.npy','records.json')]
    if not any(present):
        return None
    if not all(present):
        raise ValueError(f'{folder}: 벡터와 메타데이터 파일 중 하나가 없습니다. 기존 파일을 보존하고 중단합니다.')
    records=json.loads((folder/'records.json').read_text(encoding='utf-8'))
    # mmap allows us to avoid copying a large, already-normalized float32 matrix.
    vectors=np.load(folder/'vectors.npy',mmap_mode='r',allow_pickle=False)
    if vectors.ndim!=2 or vectors.shape[0]!=len(records):
        raise ValueError(f'{folder}: 임베딩과 메타데이터의 개수가 맞지 않습니다')
    return (vectors if vectors.dtype==np.float32 else vectors.astype(np.float32)),records


class VectorBlocks:
    """Read-only mmap blocks; filtering/merging never copies the full matrix."""
    def __init__(self, blocks):
        self.blocks = blocks
        dimensions = {vec.shape[1] for vec, keep in blocks}
        if len(dimensions) != 1:
            raise ValueError('기존 인덱스와 Studio 임베딩의 벡터 차원이 다릅니다')
        self.shape = (sum(len(keep) for _, keep in blocks), dimensions.pop())

    def __matmul__(self, query):
        results = []
        for vectors, keep in self.blocks:
            scores = np.empty(len(vectors), dtype=np.float32)
            for start in range(0, len(vectors), 8192):
                block = vectors[start:start+8192]
                norms = np.sqrt(np.einsum('ij,ij->i', block, block))
                scores[start:start+len(block)] = (block @ query) / np.maximum(norms, 1e-8)
            results.append(scores[keep])
        return np.concatenate(results)


def load_combined() -> Index:
    layers=[]
    for folder in index_folders(INDEX_DIR, STUDIO_DIR):
        item=_load_one(folder)
        if item is not None:
            layers.append(('legacy' if folder==INDEX_DIR else 'studio',item))
    if not layers:
        raise FileNotFoundError('저장된 검색 인덱스가 없습니다')
    # Work backwards so the latest complete file update shadows all older chunks.
    newer_files=set(); chosen=[]; exists_cache={}
    root=os.path.normcase(os.path.abspath(ROOT))
    for origin,(vectors,rows) in reversed(layers):
        keep=[]; layer_files=set()
        for i,row in enumerate(rows):
            rel=str(row.get('file','')).replace('\\','/')
            if not rel:
                continue
            identity=os.path.normcase(rel)
            layer_files.add(identity)
            if identity in newer_files:
                continue
            if identity not in exists_cache:
                path=os.path.normcase(os.path.abspath(os.path.join(root,rel)))
                try:
                    exists_cache[identity]=(os.path.commonpath([root,path])==root and os.path.isfile(path))
                except (ValueError,OSError):
                    exists_cache[identity]=False
            if exists_cache[identity]: keep.append(i)
        newer_files.update(layer_files)
        if keep: chosen.append((origin,vectors,rows,np.asarray(keep,dtype=np.int64)))
    records=[]; sources=[]; blocks=[]
    for origin,vectors,rows,keep in reversed(chosen):
        blocks.append((vectors,keep))
        records.extend(rows[i] for i in keep)
        sources.extend([origin]*len(keep))
    if not blocks:
        raise ValueError('인덱스는 있지만 원본 파일을 찾을 수 없습니다')
    return Index(vectors=VectorBlocks(blocks),records=records,sources=sources)


def search_index(index:Index,query_vec,allowed_kinds,k=6,source=None):
    q=np.array(query_vec,dtype=np.float32,copy=True).reshape(-1)
    if q.size!=index.vectors.shape[1]:
        raise ValueError(f'질문 벡터 차원 {q.size}, 검색 데이터 {index.vectors.shape[1]}')
    q/=max(np.linalg.norm(q),1e-8)
    scores = index.vectors @ q
    n = len(scores)
    if n == 0 or k <= 0:
        return []
    # Top-k partial selection instead of sorting every embedding on every query.
    take = min(n,max(k*8,64))
    while True:
        if take == n:
            ranked = np.argsort(-scores)
        else:
            best = np.argpartition(scores, n-take)[-take:]
            ranked = best[np.argsort(-scores[best])]
        hits=[];seen=set()
        for idx in ranked:
            rec=index.records[int(idx)]
            if rec.get('kind') not in allowed_kinds:continue
            if source and rec.get('file')!=source:continue
            key=candidate_id(rec)
            if key in seen:continue
            seen.add(key)
            hits.append({**rec,'score':float(scores[idx]),'origin':index.sources[int(idx)]})
            if len(hits)>=k:
                return hits
        if take == n:
            return hits
        take = min(n,take*2)


def related_text(record,all_records):
    if record.get('text'):return str(record['text'])
    key=candidate_id(record)
    blocks=[]
    for r in all_records:
        if candidate_id(r)==key and r.get('kind') in ('pdf_text','transcript') and r.get('text'):
            blocks.append(str(r['text']))
            if len(blocks)>=3:break
    return '\n\n'.join(blocks)


def source_description(r):
    name=Path(str(r.get('file','알 수 없는 파일'))).name
    if r.get('kind','').startswith('pdf_'):
        return f'{name} · PDF {r.get("page", "?")}페이지'
    if r.get('kind') in ('text_document','source_code'):
        return f'{name} · {r.get("line_start", "?")}–{r.get("line_end", "?")}줄'
    return f'{name} · {float(r.get("start") or 0):.0f}–{float(r.get("end") or 0):.0f}초'


def summarize_index(index):
    docs={r['file'] for r in index.records if r.get('file')}
    pdf={r['file'] for r in index.records if r.get('kind','').startswith('pdf_')}
    vid={r['file'] for r in index.records if r.get('kind') in ('video','audio','transcript')}
    sources={r['file'] for r in index.records if r.get('kind') in ('text_document','source_code')}
    return {'items':len(index.records),'files':len(docs),'pdf':len(pdf),'video':len(vid),'text_code':len(sources)}

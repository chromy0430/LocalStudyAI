"""Append-only index publication. Existing snapshots are never rewritten."""
from pathlib import Path
import json
import os
import uuid
import numpy as np
from .common import atomic_json


def index_folders(index_dir, studio_dir):
    folders = [index_dir, studio_dir]
    pointer = studio_dir / 'updates.json'
    if pointer.exists():
        names = json.loads(pointer.read_text(encoding='utf-8'))
        for name in names:
            if not isinstance(name, str) or len(name) != 32 or any(c not in '0123456789abcdef' for c in name):
                raise ValueError('잘못된 증분 인덱스 경로입니다')
            folder = studio_dir / 'updates' / name
            if not (folder/'vectors.npy').is_file() or not (folder/'records.json').is_file():
                raise ValueError('증분 인덱스가 불완전합니다. 기존 데이터를 보존하고 중단합니다.')
            folders.append(folder)
    return folders


def publish_update(studio_dir, vectors, records):
    if not records:
        return 0
    vectors=np.asarray(vectors,dtype=np.float32)
    if vectors.ndim!=2 or len(vectors)!=len(records) or not np.isfinite(vectors).all():
        raise ValueError('증분 벡터의 형태 또는 값이 올바르지 않습니다.')
    folders=index_folders(studio_dir.parent,studio_dir)
    for base in folders:
        if (base/'vectors.npy').is_file():
            previous=np.load(base/'vectors.npy',mmap_mode='r',allow_pickle=False)
            if previous.ndim!=2 or previous.shape[1]!=vectors.shape[1]:
                raise ValueError('기존 임베딩과 차원이 달라 증분 게시를 중단합니다.')
    pointer = studio_dir / 'updates.json'
    prior = json.loads(pointer.read_text(encoding='utf-8')) if pointer.exists() else []
    name = uuid.uuid4().hex
    folder = studio_dir / 'updates' / name
    folder.mkdir(parents=True)
    with (folder/'vectors.npy').open('wb') as f:
        np.save(f, vectors, allow_pickle=False)
        f.flush(); os.fsync(f.fileno())
    atomic_json(folder/'records.json', records)
    # A reader sees the complete old or new generation, never half a pair.
    atomic_json(pointer, prior + [name])
    return len(records)

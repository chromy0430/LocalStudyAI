from __future__ import annotations
from io import BytesIO
from pathlib import Path
from .common import within_root
from .search import candidate_id


def find_preview(record: dict, all_records: list[dict]) -> Path | None:
    path = record.get('preview')
    if path:
        try:
            p = within_root(path)
            if p.is_file(): return p
        except ValueError: pass
    if record.get('kind', '').startswith('pdf_'):
        for x in all_records:
            if x.get('kind') == 'pdf_image' and candidate_id(x) == candidate_id(record) and x.get('preview'):
                try:
                    p = within_root(x['preview'])
                    if p.is_file(): return p
                except ValueError: pass
    return None


def compress_image(path_or_bytes, max_edge=1100, quality=76) -> bytes:
    from PIL import Image
    with Image.open(BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes) as im:
        im = im.convert('RGB')
        im.thumbnail((max_edge, max_edge))
        buff = BytesIO()
        im.save(buff, format='JPEG', quality=quality)
        return buff.getvalue()


def video_frame(record: dict) -> bytes | None:
    import cv2
    try:
        file = within_root(str(record['file']))
    except (ValueError, KeyError): return None
    if not file.is_file(): return None
    cap = cv2.VideoCapture(str(file))
    try:
        if not cap.isOpened(): return None
        mid = (float(record.get('start') or 0) + float(record.get('end') or record.get('start') or 0)) / 2
        cap.set(cv2.CAP_PROP_POS_MSEC, mid * 1000)
        ok, frame = cap.read()
        if not ok: return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        from PIL import Image
        image = Image.fromarray(frame)
        image.thumbnail((1100, 1100))
        buf = BytesIO()
        image.save(buf, format='JPEG', quality=75)
        return buf.getvalue()
    finally:
        cap.release()


def reference_images(results, all_records, maximum=2):
    images, labels, used = [], [], set()
    for i,r in enumerate(results,1):
        if len(images)>=maximum: break
        key = candidate_id(r)
        if key in used: continue
        used.add(key)
        try:
            if r.get('kind', '').startswith('pdf_'):
                path = find_preview(r, all_records)
                payload = compress_image(path) if path else None
            elif r.get('kind') in ('video', 'audio', 'transcript'):
                payload = video_frame(r)
            else:
                payload = None
            if payload:
                images.append(payload)
                labels.append(f'이미지 {len(images)}는 [자료 {i}]의 원본 화면입니다.')
        except Exception:
            continue
    return images, labels

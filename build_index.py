
from pathlib import Path
import hashlib
import json
import subprocess

import cv2
import numpy as np
import pymupdf
import soundfile as sf
import torch

from PIL import Image
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. 기본 설정
# ==========================================

ROOT = Path(__file__).resolve().parent

PDF_DIR = ROOT / "pdfs"
VIDEO_DIR = ROOT / "videos"
INDEX_DIR = ROOT / "index"
CACHE_DIR = INDEX_DIR / "cache"

for folder in [PDF_DIR, VIDEO_DIR, INDEX_DIR, CACHE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".m4v"}

# 영상 하나를 20초씩 나누어 분석
VIDEO_CHUNK_SECONDS = 20

# 영상에서는 3초마다 프레임 한 장 사용
FRAME_INTERVAL = 3

records = []
vectors = []


# ==========================================
# 2. 로컬 AI 모델 준비
# ==========================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"EmbeddingGemma 2 실행 장치: {device}")
print("EmbeddingGemma 2 740M 로딩 중...")

# 모든 인코더를 사용한다. 740M 구성.
embedder = SentenceTransformer(
    "google/embeddinggemma-2",
    device=device
)

print("Whisper 로딩 중...")

# Whisper는 CPU에서 실행하여 GPU 메모리 절약
whisper = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)


# ==========================================
# 3. 공통 유틸리티
# ==========================================

def relative(path):
    return str(Path(path).relative_to(ROOT))


def cache_name(path):
    return hashlib.sha1(
        str(path.resolve()).encode("utf-8")
    ).hexdigest()[:16]


def save_vector(kind, source, data,
                page=None, start=None, end=None,
                text="", preview=None):

    # 글일 때는 Document 프롬프트 사용
    kwargs = {"normalize_embeddings": True}

    if kind in ("pdf_text", "transcript"):
        kwargs["prompt_name"] = "Document"

    vector = embedder.encode(data, **kwargs)

    vectors.append(np.asarray(vector, dtype=np.float32))

    records.append({
        "kind": kind,
        "file": relative(source),
        "page": page,
        "start": start,
        "end": end,
        "text": text,
        "preview": relative(preview) if preview else None
    })


def split_text(text, length=700, overlap=100):
    text = " ".join(text.split())

    if not text:
        return []

    result = []
    step = length - overlap

    for pos in range(0, len(text), step):
        result.append(text[pos:pos + length])

        if pos + length >= len(text):
            break

    return result


# ==========================================
# 4. 모든 PDF 자동 분석
# ==========================================


# ==========================================
# PDF 임베딩 성능 최적화 설정
# ==========================================

PDF_BATCH_PAGES = 8
TEXT_BATCH_SIZE = 16
IMAGE_BATCH_SIZE = 2


def index_pdfs(skip_files=None):
    skip_files = set(skip_files or [])
    files = sorted(PDF_DIR.rglob("*.pdf"))

    print(f"\nPDF {len(files)}개 발견")
    print("GPU 배치 임베딩 사용")

    for pdf_path in files:
    
        if relative(pdf_path) in skip_files:
            print(f"⏭ 기존 PDF 건너뛰기: {pdf_path.name}")
            continue
        
        print(f"\nPDF 처리: {pdf_path.name}")

        with pymupdf.open(pdf_path) as doc:
            total_pages = len(doc)
            page_id = cache_name(pdf_path)

            # 8페이지씩 묶어서 처리
            for batch_start in range(
                0, total_pages, PDF_BATCH_PAGES
            ):
                batch_end = min(
                    batch_start + PDF_BATCH_PAGES,
                    total_pages
                )

                text_inputs = []
                text_records = []

                image_inputs = []
                image_records = []

                # --------------------------
                # 1. CPU에서 페이지 준비
                # --------------------------

                for page_index in range(
                    batch_start, batch_end
                ):
                    page = doc[page_index]
                    page_number = page_index + 1

                    text = page.get_text("text")

                    # 텍스트 문단 모으기
                    for chunk in split_text(text):
                        text_inputs.append(
                            f"{pdf_path.stem}: {chunk}"
                        )

                        text_records.append({
                            "kind": "pdf_text",
                            "file": relative(pdf_path),
                            "page": page_number,
                            "start": None,
                            "end": None,
                            "text": chunk,
                            "preview": None
                        })

                    # PDF 전체 페이지 이미지 생성
                    image_path = CACHE_DIR / (
                        f"pdf_{page_id}_{page_number}.png"
                    )

                    if not image_path.exists():
                        pix = page.get_pixmap(
                            dpi=110,
                            alpha=False
                        )
                        pix.save(str(image_path))

                    image_inputs.append({
                        "image": str(image_path)
                    })

                    image_records.append({
                        "kind": "pdf_image",
                        "file": relative(pdf_path),
                        "page": page_number,
                        "start": None,
                        "end": None,
                        "text": text[:3000],
                        "preview": relative(image_path)
                    })

                # --------------------------
                # 2. GPU에서 텍스트 배치 처리
                # --------------------------

                if text_inputs:
                    text_embeddings = embedder.encode(
                        text_inputs,
                        prompt_name="Document",
                        batch_size=TEXT_BATCH_SIZE,
                        normalize_embeddings=True,
                        show_progress_bar=False
                    )

                    vectors.extend(
                        np.asarray(
                            text_embeddings,
                            dtype=np.float32
                        )
                    )

                    records.extend(text_records)

                # --------------------------
                # 3. GPU에서 이미지 배치 처리
                # --------------------------

                if image_inputs:
                    image_embeddings = embedder.encode(
                        image_inputs,
                        batch_size=IMAGE_BATCH_SIZE,
                        normalize_embeddings=True,
                        show_progress_bar=False
                    )

                    vectors.extend(
                        np.asarray(
                            image_embeddings,
                            dtype=np.float32
                        )
                    )

                    records.extend(image_records)

                print(
                    f"  {batch_end}/{total_pages} 페이지 완료"
                    f" | 텍스트 {len(text_inputs)}개"
                    f" | 이미지 {len(image_inputs)}개"
                )

        print(f"PDF 처리 완료: {pdf_path.name}")

    files = sorted(PDF_DIR.rglob("*.pdf"))

    print(f"\nPDF {len(files)}개 발견")

    for pdf_path in files:
        print(f"\nPDF 처리: {pdf_path.name}")

        with pymupdf.open(pdf_path) as doc:
            for number, page in enumerate(doc, start=1):

                # A. 텍스트 임베딩
                text = page.get_text("text")

                for chunk in split_text(text):
                    save_vector(
                        kind="pdf_text",
                        source=pdf_path,
                        data=f"title: {pdf_path.stem} | text: {chunk}",
                        page=number,
                        text=chunk
                    )

                # B. 페이지 전체 이미지 추출
                # 그림, 표, 그래프를 보존한다.
                page_id = cache_name(pdf_path)

                image_path = CACHE_DIR / (
                    f"pdf_{page_id}_{number}.png"
                )

                pix = page.get_pixmap(
                    dpi=110,
                    alpha=False
                )
                pix.save(str(image_path))

                # C. 이미지 자체 임베딩
                save_vector(
                    kind="pdf_image",
                    source=pdf_path,
                    data={"image": str(image_path)},
                    page=number,
                    text=text[:3000],
                    preview=image_path
                )

                print(f"  {number}/{len(doc)} 페이지 완료")


# ==========================================
# 5. 영상 속 프레임 읽기
# ==========================================

def read_video_frames(cap, start, end):
    frames = []

    for second in np.arange(start, end, FRAME_INTERVAL):

        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            float(second * 1000)
        )

        ok, frame = cap.read()

        if not ok:
            continue

        frame = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(frame)
        image.thumbnail((640, 640))

        frames.append(image)

    return frames


# ==========================================
# 6. 동영상 음성을 16kHz WAV로 추출
# ==========================================

def extract_audio(video_path):
    audio_path = CACHE_DIR / (
        cache_name(video_path) + ".wav"
    )

    command = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(audio_path)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("  오디오 추출 불가: 음성이 없거나 파일 문제")
        return None

    return audio_path


# ==========================================
# 7. 모든 동영상 자동 분석
# ==========================================

def index_videos(skip_files=None):
    skip_files = set(skip_files or [])
    files = sorted(
        p for p in VIDEO_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )

    print(f"\n동영상 {len(files)}개 발견")

    for video_path in files:
        if relative(video_path) in skip_files:
            print(f"⏭ 기존 동영상 건너뛰기: {video_path.name}")
            continue
        
        print(f"\n영상 처리: {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print("  영상을 열 수 없습니다.")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        if fps <= 0 or frame_count <= 0:
            print("  영상 길이를 확인할 수 없습니다.")
            cap.release()
            continue

        duration = frame_count / fps

        # A. 영상 전체에서 음성 자동 추출
        audio_path = extract_audio(video_path)

        # B. 음성 전체 자동 받아쓰기
        subtitles = []

        if audio_path:
            print("  전체 영상 음성을 텍스트로 변환 중...")

            # WAV 파일이 정상적으로 존재하는지 확인
            if not audio_path.is_file():
                raise FileNotFoundError(
                    f"오디오 파일을 찾을 수 없습니다: {audio_path}"
                )

            print(f"  오디오 파일: {audio_path}")
            print(f"  오디오 크기: {audio_path.stat().st_size:,} bytes")

            # PyAV 대신 soundfile로 직접 음성 읽기
            audio_samples, sample_rate = sf.read(
                str(audio_path),
                dtype="float32",
                always_2d=False
            )

            print(f"  샘플레이트: {sample_rate}Hz")

            if sample_rate != 16000:
                raise ValueError("오디오 샘플레이트가 16000Hz가 아닙니다.")

            if audio_samples.ndim != 1:
                raise ValueError("오디오가 모노 채널이 아닙니다.")

            # 파일 경로가 아닌 음성 배열을 Whisper에 전달
            segments, _ = whisper.transcribe(
                audio_samples,
                beam_size=3,
                vad_filter=True
            )


            subtitles = [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip()
                }
                for seg in segments
            ]

            print(f"  자막 {len(subtitles)}개 생성 완료")

        # C. 음성 파일은 한 번 열어 재사용
        wav = sf.SoundFile(str(audio_path)) if audio_path else None

        try:
            # D. 20초씩 자동 반복
            for start in np.arange(
                0, duration, VIDEO_CHUNK_SECONDS
            ):
                end = min(
                    start + VIDEO_CHUNK_SECONDS,
                    duration
                )

                start = float(start)
                end = float(end)

                print(f"  {start:.0f}초 ~ {end:.0f}초 분석")

                # D-1. 동영상 프레임 묶음 임베딩
                frames = read_video_frames(
                    cap, start, end
                )

                if frames:
                    save_vector(
                        kind="video",
                        source=video_path,
                        data={"video": frames},
                        start=start,
                        end=end
                    )

                # 해당 구간에서 말한 문장 모으기
                spoken = " ".join(
                    sub["text"]
                    for sub in subtitles
                    if sub["start"] < end
                    and sub["end"] > start
                )

                # D-2. 실제 오디오 파형 임베딩
                # Whisper 결과가 아닌 오디오 자체를 입력!
                if wav is not None:
                    rate = wav.samplerate

                    wav.seek(int(start * rate))

                    audio_chunk = wav.read(
                        int((end - start) * rate),
                        dtype="float32",
                        always_2d=False
                    )

                    if len(audio_chunk) >= rate:
                        save_vector(
                            kind="audio",
                            source=video_path,
                            data={"audio": audio_chunk},
                            start=start,
                            end=end,
                            text=spoken
                        )

                # D-3. Whisper 자막도 별도 임베딩
                if spoken.strip():
                    save_vector(
                        kind="transcript",
                        source=video_path,
                        data=(
                            f"title: {video_path.stem} | "
                            f"text: {spoken}"
                        ),
                        start=start,
                        end=end,
                        text=spoken
                    )

        finally:
            if wav is not None:
                wav.close()

            cap.release()


# ==========================================
# 8. 인덱스 저장
# ==========================================


def main():
    global records, vectors

    final_vectors = INDEX_DIR / "vectors.npy"
    final_records = INDEX_DIR / "records.json"

    pdf_vectors = INDEX_DIR / "pdf_vectors.npy"
    pdf_records = INDEX_DIR / "pdf_records.json"

    records = []
    vectors = []

    # 1. PDF와 영상이 모두 포함된 통합 인덱스 우선 사용
    if final_vectors.exists() and final_records.exists():
        print("✅ 기존 통합 인덱스를 불러옵니다.")

        matrix = np.load(final_vectors, allow_pickle=False)
        records = json.loads(
            final_records.read_text(encoding="utf-8")
        )

        if len(matrix) != len(records):
            raise ValueError(
                "기존 벡터와 메타데이터 개수가 일치하지 않습니다."
            )

        vectors = list(matrix)

    # 2. 통합 인덱스가 없다면 PDF 체크포인트 사용
    elif pdf_vectors.exists() and pdf_records.exists():
        print("✅ 기존 PDF 체크포인트를 불러옵니다.")

        matrix = np.load(pdf_vectors, allow_pickle=False)
        records = json.loads(
            pdf_records.read_text(encoding="utf-8")
        )

        if len(matrix) != len(records):
            raise ValueError(
                "PDF 체크포인트의 데이터 개수가 일치하지 않습니다."
            )

        vectors = list(matrix)

    # 3. 등록 완료된 PDF와 동영상 경로 수집
    indexed_pdfs = {
        r["file"]
        for r in records
        if r["kind"] in ("pdf_text", "pdf_image")
    }

    indexed_videos = {
        r["file"]
        for r in records
        if r["kind"] in ("video", "audio", "transcript")
    }

    print(f"등록된 PDF: {len(indexed_pdfs)}개")
    print(f"등록된 동영상: {len(indexed_videos)}개")

    previous_count = len(records)

    # 4. 새 PDF만 처리
    print("\n📄 신규 PDF 검사")
    index_pdfs(skip_files=indexed_pdfs)

    # 5. 새 동영상만 처리
    print("\n🎬 신규 동영상 검사")
    index_videos(skip_files=indexed_videos)

    # 6. 변경 사항이 없다면 종료
    if len(records) == previous_count:
        print("\n✅ 새롭게 임베딩할 자료가 없습니다.")
        return

    # 7. 모든 벡터 통합
    matrix = np.stack(vectors).astype(np.float32)

    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True
    )

    matrix = matrix / np.maximum(norms, 1e-8)

    # 8. 임시 파일에 먼저 저장
    # 저장 중 오류가 발생해도 기존 파일 보호
    temp_vectors = INDEX_DIR / "vectors.tmp"
    temp_records = INDEX_DIR / "records.tmp"

    with open(temp_vectors, "wb") as f:
        np.save(f, matrix)

    temp_records.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # 9. 저장이 완료되면 기존 인덱스 교체
    import os
    os.replace(temp_vectors, final_vectors)
    os.replace(temp_records, final_records)

    print("\n==============================")
    print("✅ 증분 임베딩 완료!")
    print(f"기존 항목: {previous_count}개")
    print(f"추가 항목: {len(records) - previous_count}개")
    print(f"전체 항목: {len(records)}개")
    print("==============================")


if __name__ == "__main__":
    main()

# LocalStudyAI Studio 0.4 — 종류별 임베딩 버튼·증분 작업 분리

## 무엇이 들어 있나요?

- **동일한 임베딩 DB로 답변 모델만 변경:** Qwen3-VL (Ollama), Gemma 4 (Ollama), Gemini API 드롭다운 선택
- 기존 `index/vectors.npy`, `index/records.json`을 **읽기 전용으로 재사용**
- 새로 넣은 PDF/영상/TXT/C#/C++/Python/Shader 파일의 임베딩은 `index/studio/`에 따로 저장해 기존 DB 손상 방지
- 기존 자료의 SHA-256 파일 변경 감지, 페이지·영상 구간 체크포인트
- PDF 본문 + 페이지 그림, 영상 프레임 + 실제 음성 파형 + Whisper 자막의 740M 임베딩
- 등록 자료/검색 통계, 소스 미리보기, 인덱싱 실행·로그, 대화 JSON 내보내기
- 예전 에이전트 기능의 **핵심 역할 프롬프트**를 전문가 모드로 정리

## 설치 위치: 중요

ZIP 파일의 `LocalStudyAI_Studio` 폴더 전체를 다음 위치로 복사합니다.

```
E:\LocalStudyAI\
  .venv\
  pdfs\
  videos\
  index\
    vectors.npy
    records.json
  build_index.py              <-- 기존 파일, 유지
  app.py                      <-- 기존 파일, 유지
  LocalStudyAI_Studio\
    START_STUDIO.bat
    studio_app.py
    studio_indexer.py
    studio_core\
```

> 기존 `build_index.py`나 `app.py`를 **덮어쓰지 마세요.**

### 1. 필요한 라이브러리

PowerShell에서:

```powershell
cd E:\LocalStudyAI
.\.venv\Scripts\python.exe -m pip install -r .\LocalStudyAI_Studio\requirements-studio.txt
```

이미 설치된 패키지는 다시 다운로드하지 않습니다. PyTorch CUDA 버전은 기존 환경을 유지하세요.

### 2. 답변 모델 목록 (자동 조회)

- **로컬:** Ollama에서 실제로 설치된 Qwen, Gemma 4 모델만 표시합니다. 모델이 없으면 해당 메뉴에 빈 목록과 안내가 표시됩니다. 설치 여부는 `ollama list`로 확인하세요. 모델을 새로 다운받을 필요는 없습니다.
- **Gemini:** 사이드바에서 API 키를 넣으면 `client.models.list()`를 호출해 **본인 키에서 사용 가능한 3.5 ~ 4의 `generateContent` 모델만 조회**합니다. `gemini-4-argon`, `gemini-3.8-flash`, `gemini-3.6-flash`, `gemini-3.5-flash-lite` 등이 접근 가능할 경우 표시됩니다. 2.5와 TTS/Live/이미지 생성 전용 모델은 제외합니다.
- **모델 목록 갱신:** Gemini 드롭다운 위의 `Gemini 모델 목록 새로고침` 버튼을 누르세요. 조회 결과는 10분간 캐시됩니다.
- AI 모델을 바꿔도 검색에 사용하는 EmbeddingGemma 2 벡터는 재사용합니다.
- **Streamlit:** 이전 `use_container_width` 사용을 `width="stretch"`로 업데이트했으며, `streamlit>=1.49`가 필요합니다.

Gemini를 사용하려면 사이드바에서 API 키를 입력한 뒤 **API 키 안전하게 저장** 버튼을 누르세요. 키는 Windows 자격 증명 관리자에 저장하며 일반 설정 JSON에 평문으로 기록하지 않습니다. 앱 재시작 후에도 재사용됩니다. 환경 변수 `GEMINI_API_KEY`가 있는 경우 환경 변수 키가 우선됩니다. Google API 답변을 선택하면 질문·검색 문단·이미지 일부가 Google 서버로 전송됩니다.

### 3. 실행

`E:\LocalStudyAI\LocalStudyAI_Studio\START_STUDIO.bat` 더블클릭. 또는:

```powershell
cd E:\LocalStudyAI
.\.venv\Scripts\python.exe -m streamlit run .\LocalStudyAI_Studio\studio_app.py
```

브라우저에서 `http://localhost:8501`로 열립니다. 기존 Streamlit이 같은 포트를 사용 중이면 먼저 종료하거나 `--server.port 8502` 옵션을 붙이세요.

### 4. 기존 자료를 먼저 테스트

첫 시작 시 `index/vectors.npy`와 `index/records.json`을 그대로 읽습니다. 새로운 임베딩은 필요 없습니다.

1. 왼쪽 **답변 제공자**에서 Qwen / Gemma 4 / Gemini 선택
2. 왼쪽 **답변 모델** 드롭다운에서 실제 설치/허용된 모델 선택
3. **질문** 탭에서 "Unity Mesh를 사용하는 장면을 찾아줘" 입력
4. 검색 결과 타임스탬프, 원본 영상 재생과 AI 답변 확인
5. 모델만 바꾸고 같은 질문을 다시 입력하여 결과 비교

### 5. 새로운 PDF·영상·텍스트·코드 추가 후 임베딩

- **자료실** 탭에 업로드하거나 기존 `E:\LocalStudyAI\pdfs`, `videos`, `documents`, `code` 폴더에 수동 복사. 일반 문서(.txt/.md/.csv/.json/.yaml)는 `documents`, C#/.cpp/.h/.hpp/.py/Unity 셰이더 등 코드는 `code` 아래에 넣으세요.
- **임베딩** 탭에서 **PDF 임베딩 / 동영상 임베딩 / 코드 임베딩 / 일반 문서 임베딩** 중 필요한 버튼만 클릭. 여러 종류를 한 번에 처리하려면 **전체 종류 증분 임베딩**을 클릭
- 선택한 종류의 신규·수정 파일만 백그라운드로 처리하며 로그는 `E:\LocalStudyAI\studio_logs`에 `index-studio-pdf-...log` 등으로 저장
- 완료 후 `index\studio\vectors.npy`, `records.json`, `manifest.json`이 생성. 코드/문서 검색은 줄 번호와 파일 이름을 보여줍니다.
- 작업 도중 실패 시 PDF의 **완료된 페이지 체크포인트**, 영상의 **완료된 구간 체크포인트**, 텍스트/코드의 **완료된 청크 체크포인트**를 다음 실행에 재사용합니다.
- Studio 인덱스는 예전 인덱스와 병합하여 검색하므로, 새 자료를 추가해도 기존 자료 검색은 유지됩니다.

**중요:** 최초 실행 시 기존 `index/records.json`에 이미 있는 파일은 재임베딩하지 않고 '기존'으로 등록합니다. 그 뒤 내용이 바뀐 파일은 SHA-256 비교로 재처리합니다. 기존 인덱스에 들어간 파일이 Studio 도입 *전에* 이미 내용이 변경된 경우, 첫 실행만으로는 변경 시점을 알 수 없으므로 해당 파일을 별도 재구축할 필요가 있습니다.

### 6. v0.4 종류별 버튼 / CLI 사용법

임베딩 화면에 네 가지 개별 버튼을 추가했습니다. **PDF 버튼으로 동영상이나 C# 파일이 재임베딩되지 않으며**, 각 종류의 SHA-256 변경 확인과 실패 재시도만 수행합니다. 변경되지 않은 PDF 페이지/영상 구간/코드 청크는 기존 체크포인트를 사용합니다. 검색 결과는 계속 PDF·영상·코드·텍스트를 통합해 제공합니다.

| 버튼 | 대상 파일 | 해당 폴더 |
|---|---|---|
| PDF 임베딩 | `.pdf` | `pdfs/` |
| 동영상 임베딩 | `.mp4`, `.mkv`, `.mov`, `.webm`, `.m4v` | `videos/` |
| 코드 임베딩 | `.cs`, `.cpp`, `.h`, `.hpp`, `.py`, `.shader` 등 | `code/`, `documents/` (확장자 기준) |
| 일반 문서 임베딩 | `.txt`, `.md`, `.csv`, `.json`, `.yaml` 등 | `documents/`, `code/` (확장자 기준) |

Windows PowerShell에서 UI 없이 직접 실행할 수도 있습니다. (한 번에 한 작업만 실행하세요.)

```powershell
cd E:\LocalStudyAI
.\.venv\Scripts\python.exe -u .\LocalStudyAI_Studio\studio_indexer.py --type pdf
.\.venv\Scripts\python.exe -u .\LocalStudyAI_Studio\studio_indexer.py --type video
.\.venv\Scripts\python.exe -u .\LocalStudyAI_Studio\studio_indexer.py --type code
.\.venv\Scripts\python.exe -u .\LocalStudyAI_Studio\studio_indexer.py --type text
# 전체 처리(기존 방식과 호환)
.\.venv\Scripts\python.exe -u .\LocalStudyAI_Studio\studio_indexer.py --type all
```

**중요:** 선택한 종류 외의 `manifest.json` 파일 기록과 Studio 검색 벡터는 유지합니다. 특정 종류의 파일을 지웠을 때는 **그 종류의 버튼**을 눌러 삭제 변경을 동기화하세요. 원래 `index/vectors.npy`의 *기존 레거시 벡터*는 읽기 전용이므로 현재 기능으로 삭제하지 않습니다. 별도 유형 선택 작업은 다른 타입의 상태를 초기화하지 않습니다.

관리 편의상 유형별 **발견 파일 수 / 등록 상태 / 확인 필요 수**가 표시됩니다. 표시는 크기와 수정 시간을 비교한 빠른 값이며 실제 인덱싱은 SHA-256으로 파일을 검사합니다. 동시 GPU 사용을 방지하기 위해 임베딩 작업 중에는 다른 버튼을 누를 수 없습니다.

### 7. v0.3에서 유지되는 수정 및 주의점

- 기존 `studio_indexer.py`의 PDF 마지막 줄 `return len(doc)`에서 PyMuPDF 문서 종료 오류가 발생하던 버그를 수정했습니다. 이미 처리한 `.npz` 페이지 체크포인트는 유지하므로 동일 PDF라면 다시 임베딩할 필요가 없습니다.
- `index/studio/parts`의 기존 체크포인트를 보존하려고 임베딩 `VERSION` 값을 변경하지 않았습니다.
- 텍스트/코드는 `documents/`, `code/` 폴더에서 탐색합니다. `bin`, `obj`, `.git`, `.venv`, `node_modules`, `Library`, `Temp` 등 생성 폴더를 무시합니다.
- 파일당 20MB 이하의 텍스트/코드만 처리하며, UTF-8, CP949, BOM 포함 UTF-16을 지원합니다.
- 텍스트/코드 파일에 토큰, 비밀번호, 개인 정보가 있으면 임베딩되기 전에 제거하세요. 키가 있는 설정 파일이나 `.env` 등을 임베딩 대상으로 등록하지 마세요.
- Gemini API 키를 **Windows Credential Manager**에 저장하려면 `keyring` 패키지가 필요합니다. `requirements-studio.txt`로 설치하세요.
- 기존 파일을 덮어쓰기 전에 `studio_indexer.py`, `index/studio`를 백업하면 더 안전합니다. **`index/vectors.npy`, `index/records.json`은 삭제하지 마세요.**
- 이번 수정은 Python 테스트용 더미 모델로 검증했고, 네 Windows GPU에서 실제 PDF 전체를 돌려보지는 못했습니다.

### 8. 기존 주의점

- 지금 버전은 **로컬 Windows 실제 GPU·Ollama 연결까지 검증한 제품은 아니며**, Python 문법과 가상 데이터의 인덱스 병합을 테스트한 첫 통합 버전입니다.
- PDF가 이미지로만 구성된 스캔본이라면 텍스트 OCR은 아직 자동으로 하지 않습니다. 페이지 이미지는 임베딩합니다.
- `Gemma 4` 또는 `Qwen3-VL` 모델이 '최종 답변 없음' 오류를 내면 오류가 표시됩니다. 모델 인스톨, 메모리, 컨텍스트 길이를 확인하세요.
- Studio 인덱싱은 **별도 프로세스**로 실행됩니다. 기존 `build_index.py`와 동시에 실행하지 마세요.
- 전문가 모드는 *프롬프트 기반* 1차 통합입니다. 예전 오케스트레이터의 단계별 도구 실행, 실시간 시장 검색, Graphviz 그래프 생성, 코드베이스 자동 수정 기능은 이 버전에 들어 있지 않습니다.
- 단일 PC·개인용 전제입니다. Streamlit 서버는 `127.0.0.1`에 바인딩하고 외부 공개하지 않는 것을 권장합니다.

## 장애 점검

- 화면이 안 열림: `streamlit run --server.port 8502` 이용
- 자료를 못 읽음: Studio 폴더가 반드시 **프로젝트 폴더 바로 아래**에 있는지 확인
- Qwen 답변이 비어 있음: Instruct 모델 설치·선택, 최신 Ollama, 로그 확인
- Gemini 사용 불가: API 키와 Google `google-genai` 설치 확인
- 인덱서가 실패함: `studio_logs\index-studio-<타입>-*.log` 마지막 부분 확인
- 동영상 음성 변환 오류: FFmpeg 설치 후 확인 (`ffmpeg -version`)

## v0.4 검증 범위

Python 테스트 14개 통과: 기존 v0.3 PDF 종료 오류/체크포인트, 증분 인덱스 병합, 키 관리 테스트 + 신규 타입별 선택, 다른 종류 데이터 유지, 선택한 종류의 삭제 동기화, 작업 실행 명령 테스트. 실제 Windows RTX 3060 Ti GPU에서 전체 임베딩 시간과 모든 설치 모델에 대한 통합 동작은 별도 확인이 필요합니다.

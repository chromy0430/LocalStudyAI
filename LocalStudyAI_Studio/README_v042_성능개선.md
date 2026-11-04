# LocalStudyAI Studio v0.4.2 — 10,000개 소스코드 환경용 성능 개선

## 수정한 파일 전체 목록

| 수정된 파일 | Windows에 덮어쓸 위치 |
| --- | --- |
| `studio_app.py` | `E:\LocalStudyAI\LocalStudyAI_Studio\studio_app.py` |
| `studio_core\search.py` | `E:\LocalStudyAI\LocalStudyAI_Studio\studio_core\search.py` |
| `studio_core\jobs.py` | `E:\LocalStudyAI\LocalStudyAI_Studio\studio_core\jobs.py` |
| `START_STUDIO.bat` | `E:\LocalStudyAI\LocalStudyAI_Studio\START_STUDIO.bat` |

**기존 임베딩·체크포인트·소스파일은 삭제하지 마세요.** `studio_indexer.py`, `index` 폴더, `pdfs`, `videos`, `documents`, `code`는 이번 패치의 변경 대상이 아닙니다.

## 추천 설치: 패치 ZIP (4개 파일만 교체)

1. 실행 중인 소스코드 임베딩이 있다면 완료까지 기다리세요. 실행 중인 Streamlit도 종료하세요.
2. PowerShell에서 백업부터 만드세요:

```powershell
cd E:\LocalStudyAI
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$target = '.\LocalStudyAI_Studio'
New-Item -ItemType Directory -Force ".\studio_backup_$stamp" | Out-Null
Copy-Item "$target\studio_app.py" ".\studio_backup_$stamp\studio_app.py"
Copy-Item "$target\studio_core\search.py" ".\studio_backup_$stamp\search.py"
Copy-Item "$target\studio_core\jobs.py" ".\studio_backup_$stamp\jobs.py"
Copy-Item "$target\START_STUDIO.bat" ".\studio_backup_$stamp\START_STUDIO.bat"
```

3. `LocalStudyAI_Studio_v0.4.2_PATCH.zip`을 열고 **그 안의 `LocalStudyAI_Studio` 폴더를** `E:\LocalStudyAI`에 풀어 **기존 파일을 덮어쓰세요**. ZIP 내부는 대상 경로와 같은 구조입니다. `E:\LocalStudyAI\LocalStudyAI_Studio\LocalStudyAI_Studio`처럼 이중 폴더가 생기지 않게 하세요.
4. 문법 확인:

```powershell
cd E:\LocalStudyAI
.\.venv\Scripts\python.exe -m py_compile .\LocalStudyAI_Studio\studio_app.py .\LocalStudyAI_Studio\studio_core\search.py .\LocalStudyAI_Studio\studio_core\jobs.py
```

5. `E:\LocalStudyAI\LocalStudyAI_Studio\START_STUDIO.bat` 실행. 기본 주소는 `http://127.0.0.1:8501`.

## 체감 변화

- **처음 접속**: 가벼운 `🏠 홈`에서 시작합니다. 벡터와 메타데이터는 아직 로딩하지 않습니다.
- **질문·자료실 접근**: 검색 인덱스를 1회 읽어 캐싱합니다. 대용량 DB는 이때 초기 로딩이 있을 수 있습니다.
- **모델·작업 모드 변경**: 질문 페이지의 컨트롤만 갱신하고 전체 인덱스를 디스크에서 재로딩하지 않습니다(인덱스 파일 변경이 없는 경우).
- **자료실**: 50개 파일씩 페이지로 표시합니다.
- **질문 파일 필터**: 10,000개 항목 드롭다운 대신, 이름 일부를 입력할 때에만 후보 최대 100개를 보여줍니다.
- **임베딩 관리**: 실제 해당 페이지에서만 원본 폴더를 검사하고, 결과는 45초 동안 캐싱합니다. 필요하면 `진행 상태 새로고침`으로 갱신합니다.
- **검색 로더**: 원본 경로 체크를 청크가 아닌 파일별 1회 수행합니다. 정규화된 하나의 인덱스만 사용할 때 메모리 매핑으로 배열 복사를 피합니다. 이중 인덱스 병합은 여전히 시간이 소요될 수 있습니다.
- **검색**: 후보를 부분 선택해, 20만 청크를 질문마다 전부 정렬하지 않습니다.
- **로그**: 새 인덱싱 프로세스는 UTF-8, 로그창은 높이 350px / 최근 150줄입니다.

## 참고 및 주의

- 이번 패치는 UI·검색 로딩 최적화입니다. 임베딩을 다시 계산하지 않습니다.
- 원본 `index/vectors.npy` 및 `index/studio/vectors.npy` 등은 교체하지 않습니다.
- `st.cache_resource`는 Streamlit 서버 프로세스가 살아 있는 동안만 유지됩니다. 서버를 완전히 종료하면 캐시가 사라지고 다음 질문 화면 접근 시 다시 로딩합니다.
- 기존 v0.4 코드 기반으로 수정했습니다. 사용자 PC에서 별도 커스텀한 코드가 있다면 백업과 비교하세요.
- Windows/Ollama/GPU 전체 실사용 성능은 사용자 PC에서 확인해야 합니다.
- PDF, 영상, 소스코드가 수십만~수백만 청크로 더 늘어나면 파일 분할 인덱스/FAISS 등 2차 구조 개편이 필요할 수 있습니다.

## 테스트

Python 구문 검사 및 로컬 단위 테스트 17개가 통과했습니다. Windows 로컬 실행 자체는 검증하지 않았습니다.

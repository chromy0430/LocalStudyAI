# LocalStudyAI Studio v0.4.3

기존 `E:\LocalStudyAI` 프로젝트에 적용하는 성능·데이터 보존 패치입니다.
`START_STUDIO.bat`를 실행하면 http://127.0.0.1:8501 에서 앱을 엽니다.

## 동작

- 홈, 설정, 전문가 자료, 임베딩 화면은 대용량 검색 데이터를 읽지 않습니다.
- 질문과 자료실에서 처음 필요할 때 검색 메타데이터와 읽기 전용 mmap을 엽니다.
- 모델·작업 모드·페이지 변경에는 같은 캐시를 재사용합니다.
- 자료실은 50개씩 표시하며 파일명 필터를 제공합니다.
- 로그는 UTF-8로 기록하고, 기존 CP949 로그도 읽습니다. 최근 150줄만 높이 350 영역에 표시합니다.
- 유형별 신규·수정 파일은 기존 체크포인트를 재사용합니다. 수정 파일의 체크포인트와 PDF 미리보기는 별도 revision에 기록합니다.
- 새 검색 데이터는 `index/studio/updates/<id>`에 추가합니다. 기존 `vectors.npy`, `records.json`, 체크포인트를 삭제하거나 전체 재구축하지 않습니다.
- 원본이 사라지면 검색에서 숨기고 저장된 벡터·체크포인트·등록 정보는 유지합니다.
- 인덱스 버전 불일치, 손상된 manifest, 벡터 차원 불일치는 자동 재구축 없이 오류로 표시합니다.
- Gemini 키는 Windows Credential Manager에 저장합니다. 환경 변수 `GEMINI_API_KEY`가 있으면 이를 우선합니다.
- Ollama는 API에서 `completion` 기능을 보고한 설치 모델만 표시합니다. Gemini는 실제 API의 `generateContent` 목록을 표시합니다. 이 목록은 음성·이미지 전용 모델도 포함할 수 있으며, 목록 조회가 모든 입력 형식의 응답 성공을 보장하지는 않습니다.

## Xevn 선택 기능

모델: https://huggingface.co/quevirus/Xevn-4B

공식 revision `ba11f0c2d8100a1e8b7789a1174ddee7c24dcab0`의 모델을
`E:\LocalStudyAI\models\Xevn-4B`에 설치했습니다. 추가 의존성은 `requirements-xevn.txt`에 있습니다.

질문 화면의 **Xevn 근거 검토 (실험)**를 켜면 검색 상위 3개 결과의 발췌 앞 900자를 CPU에서
직접 근거·부분 근거·무관·판단 불가로 검토합니다. 4분이 지나면 작업 프로세스를 종료하고 원래 검색 결과를 유지합니다.
검토 결과는 기존 검색 항목을 삭제하지 않으며 원래 점수를 바꾸지 않습니다.
모델은 요청할 때만 별도 프로세스에 로딩하고 종료 시 메모리를 반환합니다.

기본값은 꺼짐입니다. Windows CPU에서 지연이 크고, 사용자 자료에 대한 정확도 향상은 아직 검증되지 않았습니다.
이 모델의 출력 확률을 자료의 정확도나 정답 확률로 표시하지 않습니다.

## 실행

```powershell
Set-Location E:\LocalStudyAI
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
$env:HF_HOME='E:\LocalStudyAI\models\huggingface'
$env:HF_HUB_CACHE="$env:HF_HOME\hub"
$env:TORCH_HOME='E:\LocalStudyAI\models\torch'
.\.venv\Scripts\python.exe -m streamlit run LocalStudyAI_Studio\studio_app.py --server.address 127.0.0.1 --server.port 8501
```

임베딩 버튼은 명시적으로 누를 때만 증분 작업을 시작합니다. 앱 시작·질문·모델 변경은 인덱서를 실행하지 않습니다.
이전 버전의 `build_index.py`나 기존 v0.4.2 인덱서를 실행하면 보존 정책이 적용되지 않으므로 이 패치의 Studio를 사용하세요.

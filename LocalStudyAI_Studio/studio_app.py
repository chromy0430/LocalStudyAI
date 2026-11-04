"""Local Study AI Studio — multiprovider Streamlit frontend.
Install in E:\\LocalStudyAI\\LocalStudyAI_Studio then launch this file.
"""
from __future__ import annotations
from collections import Counter
import math
import json
import os
from pathlib import Path
import traceback

import numpy as np
import streamlit as st
from studio_core.common import (
    ROOT, PDF_DIR, VIDEO_DIR, DOCUMENT_DIR, CODE_DIR, INDEX_DIR, STUDIO_DIR, KINDS,
    safe_upload_name, within_root, read_json, timecode,
)
from studio_core.storage import index_folders
from studio_core.search import load_combined, search_index, summarize_index, source_description
from studio_core.media import find_preview, video_frame, reference_images
from studio_core.providers import (
    PROVIDERS, installed_models, local_model_options, available_gemini_models, make_prompt, stream_answer,
)
from studio_core.experts import EXPERTS
from studio_core.credentials import get_gemini_api_key, save_gemini_api_key, delete_gemini_api_key
from studio_core.evidence import available as xevn_available, review_evidence
from studio_indexer import (TEXT_EXTS, CODE_EXTS, SECRET_NAMES, MAX_TEXT_FILE_BYTES,
    scan_files, category_of, CATEGORY_LABELS)
from studio_core.jobs import start_job, status, log_tail

st.set_page_config(page_title='Local Study AI Studio',page_icon='📚',layout='wide',initial_sidebar_state='expanded')
st.markdown('''<style>
.block-container{max-width:1500px;padding-top:1.4rem;padding-bottom:3rem}
[data-testid="stMetric"]{border:1px solid rgba(140,150,180,.2);border-radius:14px;padding:13px 16px;background:rgba(125,125,140,.045)}
[data-testid="stSidebar"]{border-right:1px solid rgba(140,150,180,.16)}
[data-testid="stChatMessage"]{border-radius:16px;border:1px solid rgba(140,150,180,.14)}
div[data-testid="stExpander"]{border:1px solid rgba(140,150,180,.18);border-radius:10px}
.stButton>button{border-radius:10px}
</style>''',unsafe_allow_html=True)


@st.cache_resource(show_spinner=False, max_entries=1)
def cached_corpus(token):
    """Cache decoded records and vectors across Streamlit widget reruns."""
    return load_combined()


@st.cache_resource(show_spinner=False, max_entries=1)
def cached_catalog(token):
    """Compute the file catalog only once, not after every model change."""
    index = cached_corpus(token)
    by_source = {}
    for row in index.records:
        rel = row.get('file')
        if not rel:
            continue
        info = by_source.setdefault(rel, {'kinds': Counter(), 'entries': 0})
        info['entries'] += 1
        kind = row.get('kind', '')
        info['kinds'][KINDS.get(kind, kind)] += 1
    return {'by_source': by_source,
            'files': sorted(by_source, key=str.casefold),
            'stats': summarize_index(index)}


def corpus_token():
    locations = [p for folder in index_folders(INDEX_DIR, STUDIO_DIR)
                 for p in (folder/'vectors.npy', folder/'records.json')]
    return tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size)
                 for p in locations if p.is_file())



@st.cache_data(ttl=45, show_spinner=False)
def local_models_cached():
    """Avoid querying Ollama for every widget interaction."""
    return installed_models()


@st.cache_data(ttl=45, show_spinner=False)
def indexing_inventory():
    """Only scan the source tree while on the indexing page."""
    manifest = read_json(STUDIO_DIR/'manifest.json', {'files': {}, 'failed': {}})
    entries = manifest.get('files', {})
    failures = manifest.get('failed', {})
    pdfs, videos, documents = scan_files()
    sources = pdfs + videos + documents
    counts = Counter(category_of(p) for p in sources)
    completed, waiting = Counter(), Counter()
    for path in sources:
        rel = path.relative_to(ROOT).as_posix()
        cat = category_of(path)
        existing = entries.get(rel, {})
        try:
            stat = path.stat()
            same = (existing.get('size'), existing.get('mtime_ns')) == (stat.st_size, stat.st_mtime_ns)
        except OSError:
            same = False
        if existing and same and rel not in failures:
            completed[cat] += 1
        else:
            waiting[cat] += 1
    return manifest, counts, completed, waiting


@st.cache_resource(show_spinner='질문용 EmbeddingGemma 2 모델을 CPU에 로딩하는 중...')
def query_model():
    from sentence_transformers import SentenceTransformer
    # Text only: no second copy of the vision/audio branches in RAM.
    return SentenceTransformer('google/embeddinggemma-2',device='cpu',
                               config_kwargs={'vision_config':None,'audio_config':None})


@st.cache_data(ttl=600, show_spinner=False)
def cached_gemini_models(api_key: str):
    return available_gemini_models(api_key)


def query_embedding(question):
    model=query_model()
    return np.asarray(model.encode(question,prompt_name='SearchQuery',normalize_embeddings=True),dtype=np.float32)


def save_extra_context():
    st.session_state.saved_extra_context=st.session_state.extra_context


def sync_chat():
    if 'messages' not in st.session_state:st.session_state.messages=[]


def upload_files(files):
    saved=[]
    for file in files:
        clean=safe_upload_name(file.name)
        if clean.lower() in SECRET_NAMES:
            raise ValueError(f'비밀정보가 포함될 수 있는 파일명입니다: {clean}')
        ext=Path(clean).suffix.lower()
        if ext in (TEXT_EXTS | CODE_EXTS) and getattr(file, 'size', 0) > MAX_TEXT_FILE_BYTES:
            raise ValueError(f'20MB보다 큰 텍스트/코드 파일은 등록할 수 없습니다: {clean}')
        if ext=='.pdf':folder=PDF_DIR
        elif ext in {'.mp4','.mkv','.mov','.m4v','.webm'}:folder=VIDEO_DIR
        elif ext in TEXT_EXTS:folder=DOCUMENT_DIR
        elif ext in CODE_EXTS:folder=CODE_DIR
        else:raise ValueError(f'지원되지 않는 확장자: {clean}')
        folder.mkdir(parents=True,exist_ok=True)
        target=folder/clean
        stem=target.stem
        n=2
        while target.exists():
            target=folder/f'{stem}_{n}{ext}'
            n+=1
        with target.open('wb') as out:
            while True:
                data=file.read(4*1024*1024)
                if not data:break
                out.write(data)
        saved.append(target.name)
    return saved


def format_source(record):
    return source_description(record)


def reference_view(record,all_records):
    kind=record.get('kind','')
    st.caption(f'{KINDS.get(kind,kind)} · 검색 유사도 {record.get("score",0):.3f} (확률 아님)')
    if record.get('evidence_label'):
        st.caption('Xevn 근거 검토: '+record['evidence_label']+' · 검증된 정답 판정이 아닙니다.')
    content=record.get('text','')
    if content:
        st.markdown('**참고 원문 / 자막**')
        st.text(content[:4000])
    if kind in ('text_document', 'source_code'):
        st.caption(f'줄 번호: {record.get("line_start", "?")}–{record.get("line_end", "?")} · {record.get("language", "text")}')
    elif kind.startswith('pdf_'):
        image=find_preview(record,all_records)
        if image:
            st.image(str(image),caption=f'PDF {record.get("page")}페이지',width="stretch")
        else:
            st.info('원본 페이지 미리보기가 없습니다. 기존 인덱스의 preview 경로를 확인해 주세요.')
    elif kind in ('video','audio','transcript'):
        try:
            path=within_root(record.get('file',''))
            if path.exists():
                if path.suffix.lower() in ('.mp4','.webm','.m4v'):
                    st.video(str(path),start_time=int(float(record.get('start') or 0)))
                else:
                    st.caption(f'브라우저에서 직접 재생하기 어려운 파일 형식: {path.name}')
        except ValueError:
            st.warning('유효하지 않은 원본 경로입니다.')


def stats_panel(index, catalog=None):
    stats = catalog['stats'] if catalog else summarize_index(index)
    cols=st.columns(5)
    for col,(name,key) in zip(cols,[('검색 항목','items'),('등록 자료','files'),('PDF','pdf'),('영상','video'),('텍스트·코드','text_code')]):
        col.metric(name,f'{stats[key]:,}')


def chat_tab(index,provider,model,api_key,expert,k,types,filtered_source,answer_enabled,attach_text,review=False):
    st.subheader('질문 · 답변')
    st.caption('검색은 로컬 EmbeddingGemma 2가 담당합니다. 드롭다운 선택은 답변 모델만 바꿉니다.')
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            if message.get('sources'):
                with st.expander(f'참고한 자료 {len(message["sources"])}개 보기'):
                    for r in message['sources']:
                        with st.expander(format_source(r)):
                            reference_view(r,index.records)
    question=st.chat_input('PDF, 영상, VFX, Unity 코드 등에 대해 질문하세요...')
    if not question:return
    st.session_state.messages.append({'role':'user','content':question})
    with st.chat_message('user'):st.markdown(question)
    matches=[]
    try:
        if types:
            with st.spinner('PDF·영상·텍스트·코드 의미 검색 중...'):
                vec=query_embedding(question)
                matches=search_index(index,vec,set(types),k=k,source=filtered_source or None)
            if review and matches:
                try:
                    with st.spinner('Xevn이 검색 근거를 검토하는 중... (CPU, 최대 4분)'):
                        matches=review_evidence(question,matches,index.records)
                except Exception as exc:
                    st.warning(f'근거 검토를 완료하지 못했습니다. 기존 검색 결과를 유지합니다. {exc}')
        with st.chat_message('assistant'):
            if matches:
                with st.expander(f'검색 근거 {len(matches)}개 (클릭하여 미리보기)',expanded=False):
                    for i,r in enumerate(matches,1):
                        with st.expander(f'{i}. {format_source(r)}'):
                            reference_view(r,index.records)
            elif not attach_text:
                st.info('검색 근거가 없습니다. 답변 모델은 추가 지식 없이 질문에 답하는 대신, 관련 자료가 없음을 알리도록 설정되어 있습니다.')
            response=''
            if answer_enabled:
                system,user=make_prompt(question,matches,index.records,expert,attach_text)
                images,labels=reference_images(matches,index.records,maximum=2)
                user+='\n\n'+'\n'.join(labels)
                if provider=='Gemini API':
                    st.caption('클라우드 전송: 질문, 선택된 검색 문단과 이미지 최대 2장. 원본 파일 전체는 전송하지 않습니다.')
                try:
                    response=st.write_stream(stream_answer(provider,model,system,user,images,gemini_key=api_key))
                    if not response or not str(response).strip():
                        raise RuntimeError(f'{model}: 최종 답변이 비어 있습니다')
                except Exception as exc:
                    st.error(f'답변 생성 오류: {exc}')
                    response=f'⚠️ 답변 생성 실패: {exc}'
            else:
                response='검색 결과만 표시했습니다. 사이드바에서 답변 생성을 켤 수 있습니다.'
                st.info(response)
            st.session_state.messages.append({'role':'assistant','content':str(response),'sources':matches})
    except Exception as exc:
        with st.chat_message('assistant'):
            st.error(f'검색 오류: {exc}')
        st.session_state.messages.append({'role':'assistant','content':f'⚠️ 검색 오류: {exc}'})


def library_tab(index, catalog):
    st.subheader('자료실')
    stats_panel(index, catalog)
    st.markdown('#### 새로운 PDF·영상·텍스트·소스코드 등록')
    up = st.file_uploader('원본 자료 추가 (여러 파일 선택)',
        type=['pdf','mp4','mkv','mov','m4v','webm']
             + sorted(e.lstrip('.') for e in (TEXT_EXTS | CODE_EXTS)),
        accept_multiple_files=True)
    if st.button('파일 등록', disabled=not up, width='stretch'):
        try:
            names = upload_files(up)
            indexing_inventory.clear()
            st.success(f'{len(names)}개 파일을 저장했습니다. 임베딩 페이지에서 증분 업데이트를 실행하세요.')
            st.write(', '.join(names))
        except Exception as exc:
            st.error(str(exc))

    st.markdown('#### 검색 가능한 자료')
    keyword = st.text_input('자료 필터', placeholder='파일명으로 찾기', key='library_filter').casefold()
    filtered = [rel for rel in catalog['files'] if keyword in rel.casefold()]
    count = len(filtered)
    page_size = 50
    pages = max(1, math.ceil(count/page_size))
    # Widget max_value changes when filtering. Reset old page first.
    if st.session_state.get('library_page', 1) > pages:
        st.session_state.library_page = 1
    left, right = st.columns([3, 1])
    with left:
        st.caption(f'검색 결과 {count:,}개 / 전체 {len(catalog["files"]):,}개 · 페이지당 {page_size}개')
    with right:
        page = st.number_input('페이지', min_value=1, max_value=pages,
                               step=1, key='library_page')
    start = (page-1)*page_size
    for rel in filtered[start:start+page_size]:
        info = catalog['by_source'][rel]
        with st.expander(Path(rel).name):
            st.caption(rel)
            st.write('검색 항목:', info['entries'])
            st.write('유형:', ', '.join(f'{k} {v}' for k,v in info['kinds'].items()))
    if not count:
        st.info('검색한 이름에 해당하는 자료가 없습니다.')
    st.caption('기존 인덱스는 읽기 전용입니다. 삭제/재색인도 자동으로 기존 파일을 덮어쓰지 않습니다.')


def index_tab(index):
    st.subheader('유형별 증분 임베딩')
    st.caption('필요한 종류만 선택해서 임베딩하세요. 다른 타입의 저장된 벡터와 체크포인트는 유지됩니다.')
    info=status()
    busy=info.get('status')=='running'
    manifest, counts, completed, waiting = indexing_inventory()
    entries=manifest.get('files',{})
    failures=manifest.get('failed',{})
    groups=[
        ('pdf','PDF 임베딩','PDF 본문 · 페이지 이미지'),
        ('video','동영상 임베딩','영상 프레임 · 음성 · 자막'),
        ('code','코드 임베딩','C# · C++ · Python · Shader'),
        ('text','일반 문서 임베딩','TXT · MD · CSV · JSON'),
    ]
    st.markdown('#### 임베딩 종류 선택')
    for i in range(0,len(groups),2):
        columns=st.columns(2,gap='medium')
        for col,(category,title,description) in zip(columns,groups[i:i+2]):
            with col:
                with st.container(border=True):
                    st.markdown(f'**{title}**')
                    st.caption(description)
                    a,b,c=st.columns(3)
                    a.metric('발견',counts[category])
                    b.metric('등록 상태',completed[category])
                    c.metric('확인 필요',waiting[category])
                    if st.button(f'▶ {title} 시작',key=f'start_embedding_{category}',
                                 width='stretch',disabled=busy):
                        try:
                            start_job('studio',category=category)
                            indexing_inventory.clear()
                            st.success(f'{title} 작업을 시작했습니다.')
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))

    st.caption('등록 상태는 크기·수정 시간 기준의 간단한 표시입니다. 실제 인덱싱은 SHA-256으로 변경 여부를 추가 확인합니다.')
    col_all,col_refresh=st.columns(2)
    with col_all:
        if st.button('▶ 전체 종류 증분 임베딩',width='stretch',type='primary',disabled=busy):
            try:
                start_job('studio',category='all')
                indexing_inventory.clear()
                st.success('전체 종류의 신규·수정 파일 확인을 시작했습니다.')
                st.rerun()
            except Exception as exc:st.error(str(exc))
    with col_refresh:
        if st.button('↻ 진행 상태 새로고침',width='stretch'):
            indexing_inventory.clear()
            st.rerun()
    st.divider()
    st.markdown('#### 실행 상태 및 로그')
    st.write('작업 상태:',info.get('status','idle'))
    selected=info.get('category')
    if selected in CATEGORY_LABELS:
        st.write('마지막 실행 유형:',CATEGORY_LABELS[selected])
    progress=read_json(STUDIO_DIR/'progress.json',{})
    if progress:
        st.write('처리 단계:',progress.get('stage','-'))
        if progress.get('file'):st.write('처리 파일:',progress['file'])
        total=float(progress.get('total') or 0)
        done=float(progress.get('completed') or 0)
        if total:st.progress(max(0,min(done/total,1)),text=f'{done:.0f}/{total:.0f}')
        if progress.get('error'):st.error(progress['error'])
    if info.get('log'):
        with st.expander('실행 로그 · 최근 150줄',expanded=busy):
            with st.container(height=350, border=True):
                st.code(log_tail(info,limit=32768,max_lines=150),language='text')
    if failures:
        with st.expander(f'실패 파일 {len(failures)}개 확인'):
            st.json(failures)
    modes=Counter(x.get('mode','?') for x in entries.values())
    st.caption(f'기존 인덱스 재사용 {modes.get("legacy",0)}개 · Studio 등록 {modes.get("studio",0)}개 · 실패 {len(failures)}개')
    st.warning('GPU 메모리 충돌을 피하기 위해 한 번에 하나의 임베딩 작업만 실행합니다. Qwen·Gemma 답변 생성과 동시에 실행하지 않는 것을 권장합니다.')
    st.caption(f'검색 인덱스 보관 위치: {STUDIO_DIR}')


def settings_tab(index):
    st.subheader('설정 및 진단')
    st.write('로컬 프로젝트:',str(ROOT))
    st.write('기존 인덱스:',str(INDEX_DIR))
    st.write('Studio 증분 인덱스:',str(STUDIO_DIR))
    st.write('로컬 검색 엔진: EmbeddingGemma 2 (CPU에서 질문 인코딩)')
    st.write('지원하는 자료:',', '.join(KINDS.values()))
    st.write('텍스트 폴더:',str(DOCUMENT_DIR))
    st.write('소스코드 폴더:',str(CODE_DIR))
    st.caption('Gemini는 외부 Google API를 사용합니다. 비공개 자료를 클라우드로 전송하기 전 동의 여부를 확인하세요.')
    st.divider()
    st.markdown('#### 현재 등록된 Ollama 모델')
    try:
        found=local_models_cached()
    except RuntimeError as exc:
        st.warning(str(exc))
        found=[]
    if found:st.code('\n'.join(found))
    else:st.warning('Ollama 서버가 응답하지 않거나 설치 모델을 조회할 수 없습니다. `ollama list`로 확인하세요.')
    st.markdown('#### 이전 AI 전문가 기능')
    for name,spec in EXPERTS.items():
        with st.expander(name):st.write(spec.description)
    st.info('시장 검색, 전체 오케스트레이터 자동 실행, Shader Graph 실제 파일 생성 등은 이 첫 버전에서 구현하지 않았습니다. 전문가 모드는 프롬프트 수준으로 통합된 상태입니다.')


def main():
    sync_chat()
    st.title('📚 Local Study AI Studio · v0.4.3')
    st.caption('빠른 시작 · 필요한 화면만 실행 · 인덱스 메모리 캐시')

    pages = ['🏠 홈', '💬 질문', '📁 자료실', '⚙️ 임베딩', '🛠 전문가 자료', '🔧 설정']
    with st.sidebar:
        page = st.radio('화면 이동', pages, key='active_page')

    # Home, indexing and settings do not need the large vector index in RAM.
    if page == '🏠 홈':
        st.subheader('시작하기')
        st.write('질문 또는 자료실 화면에 들어갈 때 기존 검색 인덱스를 읽습니다. 모델 변경과 설정 화면에서는 다시 읽지 않습니다.')
        a,b,c=st.columns(3)
        with a:
            st.info('💬 질문: 저장된 PDF·영상·코드를 의미 검색합니다.')
        with b:
            st.info('📁 자료실: 파일 등록 및 최대 50개씩 목록 보기')
        with c:
            st.info('⚙️ 임베딩: 종류별 신규·수정 파일만 처리')
        st.caption('왼쪽 메뉴에서 기능을 선택하세요. 이미 임베딩한 자료를 다시 계산하지 않습니다.')
        return
    if page == '⚙️ 임베딩':
        index_tab(None)
        return
    if page == '🔧 설정':
        settings_tab(None)
        if st.button('검색 메모리 캐시 비우기', help='다음 질문/자료실 열기에서 다시 로딩합니다.'):
            cached_catalog.clear()
            cached_corpus.clear()
            st.session_state.pop('active_corpus_token', None)
            st.success('검색 캐시를 비웠습니다.')
        return

    # The expert attachment editor is separate from the question view and
    # remains in session_state when navigating back to chat.
    if page == '🛠 전문가 자료':
        st.subheader('전문가 모드에 추가할 코드·표·메모')
        st.caption('여기 입력한 자료는 질문에만 사용되며 자동 임베딩되지 않습니다.')
        if 'extra_context' not in st.session_state:
            st.session_state.extra_context=st.session_state.get('saved_extra_context','')
        st.text_area('직접 입력', height=220,
                     placeholder='C# 코드, 게임 데이터, 기획서 일부 등을 붙여넣으세요.',
                     key='extra_context', on_change=save_extra_context)
        file = st.file_uploader('또는 텍스트·코드·CSV 파일 선택',
                                type=['txt','md','cs','csv','json'],key='context_file')
        if file:
            st.session_state['context_file_text'] = ('\n\n[첨부 파일: ' + file.name + ']\n'
                               + file.getvalue().decode('utf-8', errors='replace')[:20000])
            st.caption(f'{file.name} 파일의 앞부분 최대 20,000자를 추가했습니다.')
        else:
            st.session_state['context_file_text'] = ''
        st.info('질문 화면에서 작업 모드를 선택하고 질문하면 이 내용을 함께 전달합니다.')
        return

    # Only question, library, expert screens need a search index.
    try:
        token = corpus_token()
        if not token:
            raise FileNotFoundError('index/vectors.npy 또는 index/studio/vectors.npy 파일이 없습니다')
        with st.spinner('저장된 검색 인덱스를 불러오는 중... (새 인덱스일 때 1회)'):
            index = cached_corpus(token)
            catalog = cached_catalog(token)
    except Exception as exc:
        st.error(f'검색 인덱스를 읽지 못했습니다: {exc}')
        st.info('왼쪽 메뉴의 임베딩 화면에서 기존 자료를 확인하거나 증분 임베딩을 실행하세요.')
        return

    if page == '📁 자료실':
        library_tab(index, catalog)
        return

    # Only chat shows provider/model/query controls. No 10,000-item selectbox.
    with st.sidebar:
        st.divider()
        st.subheader('AI 모델 설정')
        provider = st.selectbox('답변 제공자', PROVIDERS)
        api_key = ''
        options = []
        if provider == 'Gemini API':
            try:
                api_key = get_gemini_api_key()
            except Exception as exc:
                st.error(str(exc))
            key_source = ('환경 변수' if os.environ.get('GEMINI_API_KEY') else
                          ('Windows 보안 저장소' if api_key else ''))
            st.caption('API 키: '+(key_source+'에서 불러옴' if api_key else '등록되지 않음'))
            with st.form('gemini_credential_form', clear_on_submit=True):
                new_key = st.text_input('새 Gemini API 키', type='password',
                    placeholder='새 키를 입력한 뒤 저장 버튼을 누르세요')
                save_key = st.form_submit_button('API 키 안전하게 저장',width='stretch')
            if save_key:
                try:
                    save_gemini_api_key(new_key)
                    cached_gemini_models.clear()
                    st.success('Gemini API 키 저장 완료')
                    st.rerun()
                except Exception as exc:
                    st.error(f'키 저장 실패: {exc}')
            if api_key and not os.environ.get('GEMINI_API_KEY'):
                if st.button('저장된 API 키 삭제',width='stretch'):
                    try:
                        delete_gemini_api_key()
                        cached_gemini_models.clear()
                        st.rerun()
                    except Exception as exc:st.error(f'키 삭제 실패: {exc}')
            if api_key.strip():
                if st.button('Gemini 모델 목록 새로고침',width='stretch'):
                    cached_gemini_models.clear()
                try:
                    options = cached_gemini_models(api_key.strip())
                except Exception as exc:
                    st.error(f'Gemini 모델 목록 조회 실패: {exc}')
                if not options:
                    st.warning('접근 가능한 Gemini generateContent 모델을 찾지 못했습니다.')
            else:
                st.info('API 키를 저장하면 API 모델 목록을 조회합니다.')
            st.caption('목록은 API가 보고한 generateContent 지원 모델입니다. 음성·이미지 전용 모델은 일반 대화와 호환되지 않을 수 있습니다.')
            st.caption('Gemini 선택 시 질문과 검색 근거 일부가 Google 서버로 전송됩니다.')
        else:
            try:
                downloaded = local_models_cached()
            except RuntimeError as exc:
                st.warning(str(exc))
                downloaded=[]
            options = local_model_options(provider,downloaded)
            if not downloaded:
                st.warning('Ollama 서버 연결 또는 설치 모델 목록을 확인하세요.')
            elif not options:
                st.warning(f'{provider} 계열 설치 모델이 없습니다.')
            else:
                st.caption(f'설치된 모델 {len(options)}개')
            if st.button('로컬 모델 목록 새로고침',width='stretch'):
                local_models_cached.clear()
                st.rerun()
        model = st.selectbox('답변 모델',options,index=0 if options and provider!='Gemini API' else None,
                             placeholder='선택 가능한 모델 없음',disabled=not options)
        st.divider()
        expert = st.selectbox('작업 모드',list(EXPERTS))
        st.caption(EXPERTS[expert].description)
        k = st.slider('검색 결과 개수',3,15,6)
        review=st.toggle('Xevn 근거 검토 (실험)',value=False,disabled=not xevn_available())
        st.caption('선택하면 상위 3개 결과의 발췌 앞 900자를 로컬 CPU에서 검토합니다. 정확도 향상은 아직 검증되지 않았습니다.')
        types = st.multiselect('자료 종류',list(KINDS),default=list(KINDS),
                               format_func=lambda kind: KINDS[kind])
        source_search = st.text_input('자료 파일명 찾기',key='source_search',
                                      placeholder='파일명 일부를 입력 (선택)')
        selected_source = None
        if source_search.strip():
            candidates = [rel for rel in catalog['files']
                          if source_search.casefold().strip() in rel.casefold()][:100]
            if candidates:
                selected_source = st.selectbox('검색 대상 파일 (최대 100개)',
                                               ['전체 자료'] + candidates)
                if selected_source == '전체 자료':
                    selected_source = None
            else:
                st.caption('검색어에 일치하는 파일이 없습니다. 전체 자료를 대상으로 검색합니다.')
        else:
            st.caption('기본은 전체 자료 검색. 파일명을 입력하면 후보를 표시합니다.')
        generate = st.toggle('AI 답변 생성',value=True,disabled=(model is None))
        if model is None:st.caption('답변 모델이 없어 검색 전용으로 동작합니다.')
        if st.button('대화 초기화',width='stretch'):
            st.session_state.messages=[]
            st.rerun()
        if st.session_state.messages:
            st.download_button('대화 JSON 내보내기',
                data=json.dumps(st.session_state.messages,ensure_ascii=False,indent=2),
                file_name='study-chat.json',mime='application/json',width='stretch')

    stats_panel(index,catalog)
    attachment = (st.session_state.get('saved_extra_context','') or '') + (st.session_state.get('context_file_text','') or '')
    chat_tab(index,provider,model,api_key,expert,k,types,selected_source,
             generate and bool(model),attachment,review)


if __name__=='__main__':
    main()

from __future__ import annotations
from collections.abc import Iterator
from .experts import BASE_SYSTEM, EXPERTS
from .search import related_text, source_description

PROVIDERS = ('Ollama (로컬)', 'Gemini API')


def installed_models() -> list[str]:
    """Use API capabilities, never a model-family/version name guess."""
    try:
        import ollama
        client = ollama.Client(timeout=8)
        result = []
        for item in client.list().models:
            if item.model and 'completion' in (client.show(item.model).capabilities or []):
                result.append(item.model)
        return sorted(set(result), key=str.casefold)
    except Exception:
        raise RuntimeError('Ollama 모델 조회에 실패했습니다. 로컬 서버 실행 상태를 확인하세요.') from None


def local_model_options(provider, downloaded):
    return list(downloaded) if provider == PROVIDERS[0] else []


def is_current_gemini_model(name, actions):
    # A returned generateContent capability is evidence; a version prefix is not.
    return bool(name) and 'generateContent' in (actions or [])


def gemini_rank(name):
    return name.casefold()


def available_gemini_models(api_key):
    if not api_key:
        return []
    try:
        from google import genai
        with genai.Client(api_key=api_key,http_options={'timeout':15000}) as client:
            result=[]
            for model in client.models.list():
                actions=getattr(model,'supported_actions',None)
                if actions is None:
                    actions=getattr(model,'supported_generation_methods',[])
                if is_current_gemini_model(model.name,actions):
                    result.append(model.name.removeprefix('models/'))
            return sorted(set(result),key=gemini_rank)
    except Exception:
        # SDK errors may contain request URLs or credentials. Do not echo them.
        raise RuntimeError('Gemini 모델 목록 조회에 실패했습니다. 키 권한과 연결을 확인하세요.') from None


def make_prompt(question: str, matches: list[dict], all_records: list[dict], expert: str, attachment: str=''):
    spec = EXPERTS[expert]
    context = []
    for i, item in enumerate(matches, 1):
        text = related_text(item, all_records)
        context.append(f'[자료 {i}] {source_description(item)} | 유형: {item.get("kind")}\n{text[:1850] or "(시각/음향 유사도 검색 결과. 정확한 말·소리 설명은 자료에서 검증되지 않음)"}')
        if item.get('evidence_label'):
            context[-1] += '\n[보조 모델의 미검증 근거 평가: '+item['evidence_label']+']'
    joined = '\n\n'.join(context)
    system = BASE_SYSTEM + '\n전문가 모드: ' + spec.instruction
    user = f'질문/요청: {question}\n\n[검색된 자료]\n{joined or "검색 결과 없음"}'
    if attachment:
        user += '\n\n[사용자가 직접 제공한 별도 자료]\n' + attachment[:16000]
    user += '\n\n근거 자료가 없는 사항은 반드시 표시해라. 임베딩 점수는 정답 확률이 아니다.'
    user += ' 보조 모델의 평가는 참고만 하고 실제 발췌문으로 답을 확인하라. 무관하거나 불충분한 근거로 단정하지 마라.'
    return system, user


def stream_answer(provider: str, model: str, system: str, user: str, images: list[bytes], gemini_key: str='') -> Iterator[str]:
    if provider == 'Gemini API':
        if not gemini_key:
            raise ValueError('Gemini API 키가 필요합니다. 사이드바에서 입력하거나 GEMINI_API_KEY 환경 변수를 설정하세요.')
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError('Gemini API 사용을 위해 pip install google-genai 를 실행하세요.') from exc
        client = None
        try:
            client = genai.Client(api_key=gemini_key,http_options={'timeout':60000})
            parts=[types.Part.from_text(text=user)]
            parts.extend(types.Part.from_bytes(data=img,mime_type='image/jpeg') for img in images)
            stream = client.models.generate_content_stream(
                model=model, contents=[types.Content(role='user',parts=parts)],
                config=types.GenerateContentConfig(system_instruction=system, temperature=0.15)
            )
            seen = False
            for part in stream:
                chunk = getattr(part,'text',None)
                if chunk:
                    seen=True
                    yield chunk
            if not seen:
                raise RuntimeError('Gemini 응답이 비어 있습니다. API 키, 모델 접근 권한 또는 응답 차단 상태를 확인하세요.')
        except Exception:
            raise RuntimeError('Gemini 응답에 실패했습니다. 모델의 입력·출력 형식, 키 권한 및 할당량을 확인하세요.') from None
        finally:
            if client is not None:
                client.close()
    else:
        import ollama
        # Vision capable local models; use user-provided model tag (no hidden substitution).
        messages=[{'role':'system','content':system},{'role':'user','content':user}]
        client=ollama.Client(timeout=360)
        capabilities=client.show(model).capabilities or []
        if images and 'vision' in capabilities:
            messages[-1]['images']=images
        options={'think':False} if 'thinking' in capabilities else {}
        stream=client.chat(model=model,messages=messages,stream=True,**options,options={'temperature':0.15,'num_ctx':8192})
        seen=False
        thinking_chars=0
        for part in stream:
            msg = part.get('message', {}) if isinstance(part,dict) else getattr(part,'message',None)
            if msg is None: continue
            content = msg.get('content','') if isinstance(msg,dict) else getattr(msg,'content','')
            thinking = msg.get('thinking','') if isinstance(msg,dict) else getattr(msg,'thinking','')
            thinking_chars+=len(thinking or '')
            if content:
                seen=True
                yield content
        if not seen:
            raise RuntimeError(f'{model}의 최종 답변이 비어 있습니다. Thinking 길이={thinking_chars}. Instruct 태그를 선택하거나 num_ctx/메모리를 점검하세요.')

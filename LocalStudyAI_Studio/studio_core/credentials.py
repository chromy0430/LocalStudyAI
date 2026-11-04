"""Store the user's Gemini API key in the OS credential vault, never in JSON."""
from __future__ import annotations
import os

SERVICE = 'LocalStudyAI_Studio'
ACCOUNT = 'Gemini_API_Key'


def _vault():
    # Refuse configurable plaintext/fallback keyring backends.
    from keyring.backends.Windows import WinVaultKeyring
    return WinVaultKeyring()


def get_gemini_api_key() -> str:
    """Environment value takes priority; Windows keyring persists across restarts."""
    env = os.environ.get('GEMINI_API_KEY', '').strip()
    if env:
        return env
    try:
        return (_vault().get_password(SERVICE, ACCOUNT) or '').strip()
    except Exception as exc:
        raise RuntimeError('Windows 자격 증명 관리자를 읽을 수 없습니다. keyring 설치/백엔드를 확인하세요.') from exc


def save_gemini_api_key(key: str):
    key = key.strip()
    if not key:
        raise ValueError('비어 있는 API 키는 저장할 수 없습니다')
    try:
        _vault().set_password(SERVICE, ACCOUNT, key)
    except Exception:
        raise RuntimeError('Windows 보안 저장소에 키를 저장하지 못했습니다.') from None


def delete_gemini_api_key():
    import keyring
    try:
        _vault().delete_password(SERVICE, ACCOUNT)
    except keyring.errors.PasswordDeleteError:
        pass
